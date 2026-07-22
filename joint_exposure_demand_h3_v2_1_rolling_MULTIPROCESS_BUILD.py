# =====================================================
# Demand v12c1-FUTURETCN-DECODER-STOPGRAD-Z
# Based on v12c0-READHAT-NOQ:
#   1) Keep z-conditioned decoder + active-underforecast loss + H diagnostics.
#   2) Keep READHAT support: default reads exposure_hat_for_demand_h3_nonrolling.csv from current working directory.
#   3) Remove quantile/pinball loss from training (lambda_q forced to 0 by default).
#   4) Remove QSHIFT reporting: p50_amxl = MC q50, p70_amxl = MC q70.
#      No P50(q50) / P70(q70).
#   5) Add future-horizon TCN before demand cross-attention.
#   6) Connect decoder features to z only through a stop-gradient decoder epinet.
# =====================================================

"""
Clean demand model with external predicted exposure hats only.

No internal exposure decoder.
No true future DPH is used as demand input.
Supported exposure modes:
  - instock_only
  - buybox_only
  - all3
"""

import os
import time
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# GPU / device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[joint_h3_rolling] using device: {DEVICE}")


def _move_batch_to_device(batch, device):
    """Move every tensor value in a collate'd batch dict to `device`.

    Non-tensor entries (asin strings, target_week date lists, etc.) are
    left untouched.
    """
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# =====================================================
# Boss-provided WAPE utilities (verbatim, do not modify)
# These are the standardized, company-approved WAPE/penalty
# calculations. All demand WAPE reporting in this file must
# route through calculate_wape_using_lp_oos2 + quick_error_check
# (and weekly_error_check for the per-horizon breakdown) instead
# of any locally rolled abs(pred-true)/true formula. Defining them
# here also satisfies the rolling runner's own
# "calculate_wape_using_lp_oos2 not in globals()" guard, so the
# file no longer depends on these being pasted into the notebook
# session beforehand.
# =====================================================

def calculate_wape_using_lp_oos2(df, quantiles, remove_oos_dp=False, source='lp'):

    print(f"Shape when read in is {df.shape}")

    if remove_oos_dp:
        if source == 'lp':
            df = df[df['oos_status'] == 0]
        if source == 'amxl':
            df = df[df['amxl_oos'] == 0]
    else:
        df = df.copy(deep=True)

    print(f"Shape after remove oos is {df.shape}")

    for quantile in quantiles:

        print(f"Working on quantile {quantile}")

        amxl_col = f'p{int(quantile*100)}_amxl'
        scot_col = f'p{int(quantile*100)}_scot'

        amxl_overbias_col = f'p{int(quantile*100)}_amxl_overbias'
        scot_overbias_col = f'p{int(quantile*100)}_scot_overbias'

        amxl_underbias_col = f'p{int(quantile*100)}_amxl_underbias'
        scot_underbias_col = f'p{int(quantile*100)}_scot_underbias'

        amxl_penalty_col = f'p{int(quantile*100)}_amxl_penalty'
        scot_penalty_col = f'p{int(quantile*100)}_scot_penalty'

        amxl_ob_wape_col = f'p{int(quantile*100)}_amxl_ob_wape'
        scot_ob_wape_col = f'p{int(quantile*100)}_scot_ob_wape'

        amxl_ub_wape_col = f'p{int(quantile*100)}_amxl_ub_wape'
        scot_ub_wape_col = f'p{int(quantile*100)}_scot_ub_wape'

        amxl_wape_col = f'p{int(quantile*100)}_amxl_wape'
        scot_wape_col = f'p{int(quantile*100)}_scot_wape'

        delta_wape_col = f'p{int(quantile*100)}_delta_wape'

        df = df.dropna(subset=[amxl_col, scot_col])

        print(f"Shape after remove when SCOT has null fcst is {df.shape}")

        df[amxl_wape_col] = np.nan
        df[scot_wape_col] = np.nan

        df[amxl_overbias_col] = np.where(
            (df[amxl_col] >= df['fbi_demand']),
            abs(df[amxl_col] - df['fbi_demand']) * (1 - quantile),
            0
        )
        df[scot_overbias_col] = np.where(
            (df[scot_col] >= df['fbi_demand']),
            abs(df[scot_col] - df['fbi_demand']) * (1 - quantile),
            0
        )
        df[amxl_ob_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_overbias_col] / df['fbi_demand'],
            np.nan
        )
        df[scot_ob_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_overbias_col] / df['fbi_demand'],
            np.nan
        )

        df[amxl_underbias_col] = np.where(
            (df[amxl_col] < df['fbi_demand']),
            abs(df[amxl_col] - df['fbi_demand']) * quantile,
            0
        )
        df[scot_underbias_col] = np.where(
            (df[scot_col] < df['fbi_demand']),
            abs(df[scot_col] - df['fbi_demand']) * quantile,
            0
        )
        df[amxl_ub_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_underbias_col] / df['fbi_demand'],
            np.nan
        )
        df[scot_ub_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_underbias_col] / df['fbi_demand'],
            np.nan
        )

        df[amxl_penalty_col] = df[amxl_overbias_col] + df[amxl_underbias_col]
        df[scot_penalty_col] = df[scot_overbias_col] + df[scot_underbias_col]

        df[amxl_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_penalty_col] / df['fbi_demand'],
            np.nan
        )

        df[scot_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_penalty_col] / df['fbi_demand'],
            np.nan
        )

        df[delta_wape_col] = df[amxl_wape_col] - df[scot_wape_col]

    return df


def quick_error_check(df, cols):
    original_output = df[cols].sum() / df['fbi_demand'].sum()

    if 'p50_amxl_penalty' in cols:
        penalty_diff = (df['p50_amxl_penalty'].sum() - df['p50_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()
    elif 'p70_amxl_penalty' in cols:
        penalty_diff = (df['p70_amxl_penalty'].sum() - df['p70_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()
    elif 'p90_amxl_penalty' in cols:
        penalty_diff = (df['p90_amxl_penalty'].sum() - df['p90_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()

    return original_output, penalty_diff


def weekly_error_check(df, cols, cols_type):
    """
    Parameters:
    df: DataFrame
    cols: list of columns to analyze (unused, kept for signature compatibility)
    cols_type: 'p50', 'p70', or 'p90' -- selects which quantile's per-horizon breakdown to build
    """
    if df.shape[0] > 0:

        if cols_type == 'p50':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p50_amxl_penalty': 'sum',
                'p50_scot_penalty': 'sum',
                'p50_amxl_overbias': 'sum',
                'p50_scot_overbias': 'sum',
                'p50_amxl_underbias': 'sum',
                'p50_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p50_amxl_penalty'] < result['p50_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p50_amxl_overbias'] < result['p50_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p50_amxl_underbias'] < result['p50_scot_underbias'], 'win', 'lose')
            result['p50_amxl_wape'] = result['p50_amxl_penalty'] / result['fbi_demand']
            result['p50_scot_wape'] = result['p50_scot_penalty'] / result['fbi_demand']
            result['p50_diff_bps'] = (result['p50_amxl_penalty'] - result['p50_scot_penalty']) * 10000 / result['fbi_demand']

        if cols_type == 'p70':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p70_amxl_penalty': 'sum',
                'p70_scot_penalty': 'sum',
                'p70_amxl_overbias': 'sum',
                'p70_scot_overbias': 'sum',
                'p70_amxl_underbias': 'sum',
                'p70_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p70_amxl_penalty'] < result['p70_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p70_amxl_overbias'] < result['p70_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p70_amxl_underbias'] < result['p70_scot_underbias'], 'win', 'lose')
            result['p70_amxl_wape'] = result['p70_amxl_penalty'] / result['fbi_demand']
            result['p70_scot_wape'] = result['p70_scot_penalty'] / result['fbi_demand']
            result['penalty_diff_bps'] = (result['p70_amxl_penalty'] - result['p70_scot_penalty']) * 10000 / result['fbi_demand']

        if cols_type == 'p90':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p90_amxl_penalty': 'sum',
                'p90_scot_penalty': 'sum',
                'p90_amxl_overbias': 'sum',
                'p90_scot_overbias': 'sum',
                'p90_amxl_underbias': 'sum',
                'p90_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p90_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum',
                'p90_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p90_amxl_penalty'] < result['p90_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p90_amxl_overbias'] < result['p90_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p90_amxl_underbias'] < result['p90_scot_underbias'], 'win', 'lose')
            result['p90_amxl_wape'] = result['p90_amxl_penalty'] / result['fbi_demand']
            result['p90_scot_wape'] = result['p90_scot_penalty'] / result['fbi_demand']
            result['penalty_diff_bps'] = (result['p90_amxl_penalty'] - result['p90_scot_penalty']) * 10000 / result['fbi_demand']
        return result
    else:
        print('df has zero rows')
        return pd.DataFrame()


# =====================================================
# 0. Sampling
# =====================================================

def prepare_data_sample(data_raw1, n_asins=5000):
    data_raw1 = data_raw1.copy()
    data_raw1["order_week"] = pd.to_datetime(data_raw1["order_week"])
    sample_asins = np.random.choice(
        data_raw1["asin"].unique(),
        size=min(n_asins, data_raw1["asin"].nunique()),
        replace=False
    )
    data_small = data_raw1[data_raw1["asin"].isin(sample_asins)].copy()
    print("Sample ASINs:", data_small["asin"].nunique())
    print("Sample rows:", len(data_small))
    return data_small



def prepare_data_from_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
):
    """
    Sample ASINs from data_raw1, then keep only ASINs also present in scot_df.

    n_asins=None means "use every ASIN" -- no sampling is done.
    """
    df = data_raw1.copy()
    scot = scot_df.copy()

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    scot["asin"] = scot["asin"].astype(str)

    unique_asins = df["asin"].dropna().unique()

    if n_asins is None:
        sample_asins = unique_asins
    else:
        rng = np.random.default_rng(seed)
        sample_asins = rng.choice(
            unique_asins,
            size=min(n_asins, len(unique_asins)),
            replace=False,
        )

    sample_asin_set = set(sample_asins)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    print("\n" + "=" * 80)
    print("SAMPLE-SCOT ASIN INTERSECTION")
    print("=" * 80)
    print("Sample ASINs:", len(sample_asin_set))
    print("SCOT ASINs:", len(scot_asin_set))
    print("Intersection ASINs:", len(intersect_asins))
    print("Sample ASINs missing in SCOT:", len(sample_asin_set - scot_asin_set))

    data_small = df[df["asin"].isin(intersect_asins)].copy()
    sample_asin_df = pd.DataFrame({"asin": list(sample_asins)})
    intersect_asin_df = pd.DataFrame({"asin": intersect_asins})

    print("Data rows after intersection:", len(data_small))
    print("Data ASINs after intersection:", data_small["asin"].nunique())

    return data_small, sample_asin_df, intersect_asin_df


def add_zero_rate_group(data_raw, zero_thresholds=(0.4, 0.7)):
    df = data_raw.copy()
    df["fbi_demand"] = pd.to_numeric(df["fbi_demand"], errors="coerce").fillna(0).clip(lower=0)
    asin_stats = (
        df.groupby("asin")
        .agg(
            zero_rate=("fbi_demand", lambda x: (x == 0).mean()),
            total_demand=("fbi_demand", "sum"),
            n_weeks=("fbi_demand", "count"),
        )
        .reset_index()
    )
    low, high = zero_thresholds
    def assign_group(z):
        if z < low: return "low_sparse"
        elif z < high: return "mid_sparse"
        else: return "high_sparse"
    asin_stats["zero_group"] = asin_stats["zero_rate"].apply(assign_group)
    df = df.merge(asin_stats[["asin", "zero_rate", "zero_group"]], on="asin", how="left")
    print("\nASIN counts by zero-rate group:")
    print(asin_stats.groupby("zero_group")["asin"].nunique().reset_index(name="n_asins"))
    return df, asin_stats


# =====================================================
# 1. Data loading
# =====================================================


def _infer_pkg_dimension_cols(df):
    """
    Infer package height, length, and width columns for package-volume diagnostics.
    Diagnostic only; not used as model input.
    """
    lower_map = {c.lower(): c for c in df.columns}

    candidates = {
        "height": [
            "pkg_height", "package_height", "pkg_h", "height",
            "item_height", "unit_height"
        ],
        "length": [
            "pkg_length", "package_length", "pkg_l", "length",
            "item_length", "unit_length"
        ],
        "width": [
            "pkg_width", "package_width", "pkg_w", "width",
            "item_width", "unit_width"
        ],
    }

    out = {}

    for dim_name, names in candidates.items():
        out[dim_name] = None
        for name in names:
            if name in lower_map:
                out[dim_name] = lower_map[name]
                break

    return out




def _get_1d_col(df, col):
    """
    Return one 1-D Series even if df has duplicate column names.
    """
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x



def _compute_total_dph_cap(df, q=0.995):
    """
    Compute a global cap from total_dph.

    For fast experiments, this uses the current modeling dataframe.
    For a stricter production backtest, compute this cap using training weeks only.
    """
    if "total_dph" not in df.columns:
        return np.inf

    s = pd.to_numeric(df["total_dph"], errors="coerce").fillna(0.0).clip(lower=0)

    if len(s) == 0 or s.sum() <= 0:
        return np.inf

    cap = float(s.quantile(q))

    if not np.isfinite(cap) or cap <= 0:
        return np.inf

    return cap


def _apply_dph_cap(df, cap):
    """
    Apply one total_dph-based cap to total_dph, buy_box_dph, and in_stock_dph.
    This stabilizes heavy-tailed exposure decoder targets.
    """
    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
            if np.isfinite(cap):
                df[c] = df[c].clip(upper=cap)
    return df



def _select_stock_decoder_extra_cols(data_raw):
    """
    Select additional features to help the external exposure covariates.

    These are NOT true future in_stock_dph. They are product / popularity / price / promo
    / package features that can help predict future exposure.

    We keep a conservative list to avoid leakage-prone realized future outcomes.
    """
    # Top-3 ablation choice for the external exposure / in-stock decoder.
    # These are the only additional stock_extra__ features added to future_context.
    candidate_cols = [
        "ind_promotion",
        "promotion_pricing_amount",
        "pricing_type",
    ]

    # Avoid realized target / future outcome columns.
    exclude_cols = {
        "fbi_demand",
        "order_units",
        "scot_oos",
        "in_stock_dph",
        "asin",
        "order_week",
    }

    cols = [
        c for c in candidate_cols
        if c in data_raw.columns and c not in exclude_cols
    ]

    return cols


def _encode_stock_decoder_extra_features(df, extra_cols):
    """
    Convert extra external-exposure related features to numeric features.

    Object/categorical columns are ordinal-encoded by pandas.factorize.
    This keeps the implementation lightweight and avoids requiring sklearn encoders.
    """
    out_cols = []

    for c in extra_cols:
        new_c = f"stock_extra__{c}"

        if c not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            val = pd.to_numeric(_get_1d_col(df, c), errors="coerce").fillna(0.0)

            # Conservative transforms by feature type.
            cl = c.lower()
            if (
                "count" in cl or "dph" in cl or "price" in cl
                or "amount" in cl or "rank" in cl or "score" in cl
                or "height" in cl or "length" in cl or "width" in cl
                or "weight" in cl or "wordcount" in cl
            ):
                val = np.log1p(val.clip(lower=0))

            # Scale robustly to avoid huge values.
            std = float(val.std()) if float(val.std()) > 1e-8 else 1.0
            mean = float(val.mean())
            df[new_c] = ((val - mean) / std).clip(-5, 5)

        else:
            codes, uniques = pd.factorize(_get_1d_col(df, c).astype(str).fillna("MISSING"))
            # normalize category code to roughly [0,1]
            denom = max(len(uniques) - 1, 1)
            df[new_c] = codes.astype(float) / denom

        out_cols.append(new_c)

    return df, out_cols



def _safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def _rolling_mean(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def _rolling_max(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).max().values


def _rolling_std(arr, window):
    return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values


def _rolling_positive_mean(arr, window):
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_quantile(arr, window, q):
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = np.quantile(vals, q) if len(vals) > 0 else 0.0
    return out


def _rolling_max_lag(arr, window):
    """Lag-safe rolling max excluding current step."""
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        out[i] = vals.max() if len(vals) > 0 else 0.0
    return out


def _zero_streak(active):
    out = np.zeros(len(active), dtype=np.float32)
    cur = 0
    for i, a in enumerate(active):
        if a > 0: cur = 0
        else: cur += 1
        out[i] = cur
    return out


def load_real_data(data_raw, dph_cap_q=0.995):
    _stage_t0 = time.perf_counter()
    print(f"[STAGE] load_real_data START | rows={len(data_raw):,} | asins={data_raw['asin'].nunique() if 'asin' in data_raw.columns else 'NA'}", flush=True)
    """
    34 history features.
    Feature index map:
      0  log1p(demand)
      1  active indicator
      2  distance since last active / 52
      3  sin(2π t/52)
      4  cos(2π t/52)
      5  promo_t
      6  sin(2π t/13)
      7  cos(2π t/13)
      8  hist_nonzero_mean_52_log   ← lag-fixed
      9  hist_nonzero_p75_52_log    ← lag-fixed
      10 recent_peak_13_log         ← lag-fixed
      11 in_stock_dph_lag_log
      12 oos
      13 active_rate_4
      14 active_rate_13
      15 oos_rate_4
      16 oos_rate_13
      17 instock_mean_4_log
      18 instock_mean_13_log
      19 zero_streak_scaled
      20 price_log
      21 positive_mean_4_log        ← lag-fixed
      22 positive_mean_13_log       ← lag-fixed
      23 positive_max_13_log        ← lag-fixed
      24 positive_std_13

      Added historical DPH funnel features:
      25 total_dph_log
      26 buy_box_dph_log
      27 total_dph_mean_4_log
      28 total_dph_mean_13_log
      29 buy_box_dph_mean_4_log
      30 buy_box_dph_mean_13_log
      31 buy_box_rate
      32 in_stock_rate
      33 in_stock_given_buybox
    """
    holiday_cols = [c for c in data_raw.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in data_raw.columns if c.startswith("distance_")]
    stock_extra_raw_cols = _select_stock_decoder_extra_cols(data_raw)
    pkg_cols = _infer_pkg_dimension_cols(data_raw)

    # ------------------------------------------------------------
    # Future-known context features.
    # We add business seasonality and major shopping-event proximity
    # BEFORE keep_cols is created, so these columns truly enter future_context.
    # ------------------------------------------------------------
    data_raw = data_raw.copy()
    data_raw["order_week"] = pd.to_datetime(data_raw["order_week"], errors="coerce")
    data_raw["order_month"] = data_raw["order_week"].dt.month.astype(float)
    data_raw["month_sin"] = np.sin(2 * np.pi * data_raw["order_month"] / 12.0)
    data_raw["month_cos"] = np.cos(2 * np.pi * data_raw["order_month"] / 12.0)

    data_raw["season_winter"] = data_raw["order_month"].isin([12, 1, 2]).astype(float)
    data_raw["season_spring"] = data_raw["order_month"].isin([3, 4, 5]).astype(float)
    data_raw["season_summer"] = data_raw["order_month"].isin([6, 7, 8]).astype(float)
    data_raw["season_fall"] = data_raw["order_month"].isin([9, 10, 11]).astype(float)

    seasonal_cols = [
        "order_month",
        "month_sin",
        "month_cos",
        "season_winter",
        "season_spring",
        "season_summer",
        "season_fall",
    ]

    # Major event proximity from distance_* columns.
    # This is robust to slightly different distance column names.
    event_keywords = [
        "black", "cyber", "prime", "christmas", "thanksgiving",
        "newyear", "new_year", "labor", "memorial",
    ]
    proximity_cols = []
    for c in distance_cols:
        c_lower = c.lower()
        if any(k in c_lower for k in event_keywords):
            new_c = f"{c}_proximity"
            data_raw[new_c] = (
                1.0 - pd.to_numeric(data_raw[c], errors="coerce").fillna(0.0).abs()
            ).clip(0.0, 1.0)
            proximity_cols.append(new_c)

    # Include holiday indicators, raw distance features, explicit season features,
    # and major-event proximity features.
    context_cols = ["our_price"] + holiday_cols + distance_cols + seasonal_cols + proximity_cols
    context_cols = list(dict.fromkeys(context_cols))

    base_cols = ["asin", "order_week", "fbi_demand", "scot_oos"]

    # Keep in_stock_dph for history encoder only.
    # It is intentionally excluded from future_context.
    # Keep DPH variables for history-only safe proxy features.
    # They are not used as raw future context.
    history_only_cols = ["in_stock_dph", "total_dph", "buy_box_dph"]

    extra_diag_cols = [c for c in pkg_cols.values() if c is not None]

    keep_cols = [
        c for c in base_cols + context_cols + history_only_cols + extra_diag_cols + stock_extra_raw_cols
        if c in data_raw.columns
    ]

    # Remove duplicate column names. Duplicates can happen because package columns
    # are used both for total_size diagnostics and stock-decoder extra features.
    keep_cols = list(dict.fromkeys(keep_cols))

    df = data_raw[keep_cols].copy()

    # Encode additional product / popularity / promo / size features for stock decoder.
    df, stock_extra_cols = _encode_stock_decoder_extra_features(df, stock_extra_raw_cols)

    # Add encoded stock-extra columns to future_context.
    # These features help the external exposure covariates.
    context_cols = context_cols + stock_extra_cols

    # Forecast-origin-safe historical DPH proxy features.
    # These columns are placeholders here and are filled inside DemandDataset
    # using only history up to each forecast origin.
    dph_proxy_cols = [
        "hist_total_dph_last_log",
        "hist_total_dph_mean4_log",
        "hist_total_dph_mean13_log",
        "hist_buy_box_dph_last_log",
        "hist_buy_box_dph_mean4_log",
        "hist_buy_box_dph_mean13_log",
        "hist_instock_dph_last_log",
        "hist_instock_dph_mean4_log",
        "hist_instock_dph_mean13_log",
    ]
    for c in dph_proxy_cols:
        df[c] = 0.0

    context_cols = context_cols + dph_proxy_cols
    df = df.rename(columns={"asin":"ASIN","order_week":"Week","fbi_demand":"Demand","scot_oos":"OOS"})

    h_col = pkg_cols.get("height")
    l_col = pkg_cols.get("length")
    w_col = pkg_cols.get("width")

    if h_col is not None and l_col is not None and w_col is not None:
        pkg_h = pd.to_numeric(_get_1d_col(df, h_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_l = pd.to_numeric(_get_1d_col(df, l_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_w = pd.to_numeric(_get_1d_col(df, w_col), errors="coerce").fillna(0).clip(lower=0)
        df["pkg_volume_raw"] = pkg_h * pkg_l * pkg_w
    else:
        df["pkg_volume_raw"] = np.nan

    df["Week"] = pd.to_datetime(df["Week"])
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0).clip(lower=0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)
    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    # Keep raw price for amount diagnostics, then use log price for model context.
    df["our_price_raw"] = df["our_price"].clip(lower=0)
    df["our_price"] = np.log1p(df["our_price_raw"])

    # Use historical in_stock_dph directly in the encoder; no lag shift.
    # Future in_stock_dph is not used in future_context.
    if "in_stock_dph" in df.columns:
        df["in_stock_dph"] = pd.to_numeric(df["in_stock_dph"], errors="coerce").fillna(0.0)
        df["in_stock_dph"] = df["in_stock_dph"].clip(lower=0)
    else:
        df["in_stock_dph"] = 0.0

    # Historical total_dph / buy_box_dph are used only as forecast-origin-safe summaries.
    for c in ["total_dph", "buy_box_dph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
        else:
            df[c] = 0.0

    # Cap heavy-tailed DPH targets using total_dph as a unified exposure scale cap.
    # This cap is applied before constructing decoder targets.
    dph_cap = _compute_total_dph_cap(df, q=dph_cap_q)
    df = _apply_dph_cap(df, dph_cap)
    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    # Distance-to-holiday features are future-known scalar calendar features.
    # Keep direction if raw values are signed: negative = before holiday, positive = after holiday.
    for c in distance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].clip(lower=-12, upper=12) / 12.0

    df = df.sort_values(["ASIN", "Week"]).reset_index(drop=True)

    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_window = np.roll(cur, -1); prev_window[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_window))
        df["promo_t"] = holiday_window
    else:
        df["promo_t"] = 0.0

    df["t"] = ((df["Week"] - df["Week"].min()).dt.days // 7).astype(int)

    data = {}
    _groups = df.groupby("ASIN", sort=False)
    _n_asins = df["ASIN"].nunique()
    _feat_t0 = time.perf_counter()
    print(f"[STAGE] per-ASIN feature construction START | {_n_asins:,} ASINs | SERIAL", flush=True)
    for _asin_i, (asin, group) in enumerate(_groups, start=1):
        if _asin_i == 1 or _asin_i % 2000 == 0 or _asin_i == _n_asins:
            _elapsed = time.perf_counter() - _feat_t0
            _rate = _asin_i / max(_elapsed, 1e-9)
            _eta = (_n_asins - _asin_i) / max(_rate, 1e-9)
            print(
                f"[PROGRESS] per-ASIN features {_asin_i:,}/{_n_asins:,} "
                f"({100*_asin_i/max(_n_asins,1):.1f}%) | elapsed={_elapsed/60:.1f}m | ETA={_eta/60:.1f}m",
                flush=True,
            )
        group = group.reset_index(drop=True)
        demand = group["Demand"].values.astype(float)
        oos    = group["OOS"].values.astype(float)
        weeks  = group["Week"].values
        t      = group["t"].values
        T      = len(demand)

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0: last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        in_stock_lag = group["in_stock_dph"].values.astype(float)
        instock_raw  = group["in_stock_dph"].values.astype(float)
        price_log    = group["our_price"].values.astype(float)
        price_raw    = group["our_price_raw"].values.astype(float)
        pkg_volume_raw = group["pkg_volume_raw"].values.astype(float)
        total_dph_raw = group["total_dph"].values.astype(float)
        buy_box_dph_raw = group["buy_box_dph"].values.astype(float)

        # All rolling features now exclude current step (leak-free)
        hist_nonzero_mean_52 = _rolling_positive_mean(demand, 52)
        hist_nonzero_p75_52  = _rolling_positive_quantile(demand, 52, 0.75)
        recent_peak_13       = _rolling_max_lag(demand, 13)

        active_rate_4   = _rolling_mean(b_t, 4)
        active_rate_13  = _rolling_mean(b_t, 13)
        oos_rate_4      = _rolling_mean(oos, 4)
        oos_rate_13     = _rolling_mean(oos, 13)
        instock_mean_4  = _rolling_mean(in_stock_lag, 4)
        instock_mean_13 = _rolling_mean(in_stock_lag, 13)

        total_dph_mean_4  = _rolling_mean(total_dph_raw, 4)
        total_dph_mean_13 = _rolling_mean(total_dph_raw, 13)
        buy_box_dph_mean_4  = _rolling_mean(buy_box_dph_raw, 4)
        buy_box_dph_mean_13 = _rolling_mean(buy_box_dph_raw, 13)

        buy_box_rate = buy_box_dph_raw / (total_dph_raw + 1.0)
        in_stock_rate = instock_raw / (total_dph_raw + 1.0)
        in_stock_given_buybox = instock_raw / (buy_box_dph_raw + 1.0)

        buy_box_rate = np.clip(buy_box_rate, 0.0, 10.0)
        in_stock_rate = np.clip(in_stock_rate, 0.0, 10.0)
        in_stock_given_buybox = np.clip(in_stock_given_buybox, 0.0, 10.0)

        zero_streak     = _zero_streak(b_t) / 52.0

        positive_mean_4  = _rolling_positive_mean(demand, 4)
        positive_mean_13 = _rolling_positive_mean(demand, 13)
        positive_max_13  = _rolling_max_lag(demand, 13)
        positive_std_13  = _rolling_std(np.log1p(demand), 13)

        features = np.stack([
            v_t,
            b_t,
            d_t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
            group["promo_t"].values.astype(float),
            np.sin(2 * np.pi * t / 13),
            np.cos(2 * np.pi * t / 13),
            np.log1p(hist_nonzero_mean_52),   # 8
            np.log1p(hist_nonzero_p75_52),    # 9
            np.log1p(recent_peak_13),         # 10
            np.log1p(in_stock_lag),
            oos,
            active_rate_4,
            active_rate_13,
            oos_rate_4,
            oos_rate_13,
            np.log1p(instock_mean_4),
            np.log1p(instock_mean_13),
            zero_streak,
            price_log,
            np.log1p(positive_mean_4),
            np.log1p(positive_mean_13),
            np.log1p(positive_max_13),
            positive_std_13,

            np.log1p(total_dph_raw),
            np.log1p(buy_box_dph_raw),
            np.log1p(total_dph_mean_4),
            np.log1p(total_dph_mean_13),
            np.log1p(buy_box_dph_mean_4),
            np.log1p(buy_box_dph_mean_13),
            buy_box_rate,
            in_stock_rate,
            in_stock_given_buybox,
        ], axis=1).astype(np.float32)

        future_context = group[context_cols].values.astype(np.float32)


        data[asin] = {
            "features": features,
            "future_context": future_context,
            "demand": demand.astype(np.float32),
            "week": weeks,
            "oos": oos.astype(np.float32),
            "price_raw": price_raw.astype(np.float32),
            "pkg_volume_raw": pkg_volume_raw.astype(np.float32),
            "instock_raw": instock_raw.astype(np.float32),
            "total_dph_raw": total_dph_raw.astype(np.float32),
            "buy_box_dph_raw": buy_box_dph_raw.astype(np.float32),
            "dph_proxy_context_idx": {
                c: context_cols.index(c) for c in dph_proxy_cols if c in context_cols
            },
        }

    print(f"[STAGE] per-ASIN feature construction DONE | {(time.perf_counter()-_feat_t0)/60:.2f} min", flush=True)
    print(f"[STAGE] load_real_data DONE | {(time.perf_counter()-_stage_t0)/60:.2f} min", flush=True)
    print("History encoder dim: 34")
    print(f"Package dimension columns for total_size: {pkg_cols}")
    print("History in_stock_dph: raw historical value, no lag shift")
    print("Future context excludes in_stock_dph")
    print("Future context includes distance_* calendar features")
    print("External exposure safe mode: demand uses external predicted DPH hats only")
    print("Safe historical DPH proxies: total/buy_box/in_stock last/mean4/mean13")
    print("History encoder includes DPH funnel features")
    print(f"DPH cap q: {dph_cap_q} | cap value: {dph_cap}")
    print(f"Context dim: {len(context_cols)}")
    return data, len(context_cols), context_cols


# =====================================================
# 2. Dataset
# =====================================================

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=3, mode="train", val_weeks=20):
        self.samples = []
        for asin, d in data.items():
            T = len(d["demand"])
            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                self.samples.append({
                    "x": torch.tensor(d["features"][start:start+history], dtype=torch.float32),
                    "future_context": torch.tensor(
                        self._make_future_context_with_dph_proxies(
                            d=d,
                            start=start,
                            history=history,
                            horizon=horizon,
                        ),
                        dtype=torch.float32),
                    "y": torch.tensor(d["demand"][start+history:start+history+horizon], dtype=torch.float32),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in d["week"][start+history:start+history+horizon]],
                    "oos": torch.tensor(d["oos"][start+history:start+history+horizon], dtype=torch.float32),
                    "our_price": torch.tensor(
                        d["price_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "pkg_volume": torch.tensor(
                        d["pkg_volume_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_instock": torch.tensor(
                        d["instock_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_total_dph": torch.tensor(
                        d["total_dph_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_buy_box_dph": torch.tensor(
                        d["buy_box_dph_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                })

    def _safe_hist_mean(self, arr, start, history, window):
        hist = arr[start:start+history]
        if len(hist) == 0:
            return 0.0
        hist = hist[-min(window, len(hist)):]
        return float(np.mean(hist))

    def _make_future_context_with_dph_proxies(self, d, start, history, horizon):
        """
        Fill historical DPH summary proxy features using only values up to forecast origin.
        These are repeated across the horizon and do not use future true DPH.
        """
        fc = d["future_context"][start+history:start+history+horizon].copy()
        idx = d.get("dph_proxy_context_idx", {})

        total_hist = d.get("total_dph_raw", None)
        buy_hist = d.get("buy_box_dph_raw", None)
        instock_hist = d.get("instock_raw", None)

        def fill(col, val):
            if col in idx:
                fc[:, idx[col]] = np.log1p(max(float(val), 0.0))

        if total_hist is not None:
            total_last = total_hist[start+history-1] if history > 0 else 0.0
            fill("hist_total_dph_last_log", total_last)
            fill("hist_total_dph_mean4_log", self._safe_hist_mean(total_hist, start, history, 4))
            fill("hist_total_dph_mean13_log", self._safe_hist_mean(total_hist, start, history, 13))

        if buy_hist is not None:
            buy_last = buy_hist[start+history-1] if history > 0 else 0.0
            fill("hist_buy_box_dph_last_log", buy_last)
            fill("hist_buy_box_dph_mean4_log", self._safe_hist_mean(buy_hist, start, history, 4))
            fill("hist_buy_box_dph_mean13_log", self._safe_hist_mean(buy_hist, start, history, 13))

        if instock_hist is not None:
            instock_last = instock_hist[start+history-1] if history > 0 else 0.0
            fill("hist_instock_dph_last_log", instock_last)
            fill("hist_instock_dph_mean4_log", self._safe_hist_mean(instock_hist, start, history, 4))
            fill("hist_instock_dph_mean13_log", self._safe_hist_mean(instock_hist, start, history, 13))

        return fc

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# =====================================================
# 3. Model
# =====================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparsePeakAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4, beta_peak=1.0, soft_mask_scale=3.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.beta_peak = beta_peak
        self.soft_mask_scale = soft_mask_scale

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(0.1)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x, b_t, peak_score):
        B, T, D = x.shape
        q = self.q_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_head)

        # Softly down-weight zero-demand weeks.
        sparse_mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        scores = scores - self.soft_mask_scale * sparse_mask.float()[:, None, None, :]

        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
        scores = scores + self.beta_peak * peak_norm[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1,2).contiguous().view(B,T,D)
        out  = self.out_proj(out)
        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    def __init__(self, input_dim=34, d_model=32, horizon=3):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        # Dilations include quarterly and annual scales.
        dilations = [1, 2, 4, 8, 13, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])

        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=1.0)
        self.final_norm  = nn.LayerNorm(d_model)

        self.base_head  = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))
        self.alpha_head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))

    def encode(self, x):
        """Return full encoder states and final state.

        H_enc: [B, T, d_model]
        h_t:   [B, d_model]
        b_t / peak_score are returned for decoder-side peak attention biases.
        """
        b_t        = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:,:,0]).clamp(min=0) + 1e-6)

        h = self.input_proj(x).permute(0,2,1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0,2,1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0,2,1)

        H_enc = self.sparse_attn(h.permute(0,2,1), b_t, peak_score)
        h_t   = self.final_norm(H_enc[:,-1,:])
        return H_enc, h_t, b_t, peak_score

    def forward(self, x):
        H_enc, h_t, b_t, peak_score = self.encode(x)
        mu    = F.softplus(self.base_head(h_t))
        alpha = F.softplus(self.alpha_head(h_t)) + 1e-4
        return mu, alpha, h_t


class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=32, context_dim=2, d_z=16, horizon=3):
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(d_phi + horizon * context_dim, 64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2 * d_z)
        )

    def forward(self, phi, future_context):
        B   = phi.shape[0]
        ctx = future_context.reshape(B, -1)
        out = self.net(torch.cat([phi, ctx], dim=-1))
        z_mean, z_logstd = out.chunk(2, dim=-1)
        z_std = F.softplus(z_logstd) + 1e-4
        return z_mean, z_std


class Epinet(nn.Module):
    def __init__(self, d_phi=32, d_z=16, horizon=3, prior_scale=0.3):
        super().__init__()
        self.d_z = d_z; self.horizon = horizon; self.prior_scale = prior_scale
        self.learnable = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2*horizon*d_z)
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(),
            nn.Linear(64, 2*horizon*d_z)
        )
        for p in self.prior.parameters(): p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl  = self.learnable(inp).view(-1, 2*self.horizon, self.d_z)
        sl  = torch.einsum("bhd,bd->bh", sl, z)
        sp  = self.prior(inp).view(-1, 2*self.horizon, self.d_z)
        sp  = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale
        out = sl + sp
        return out[:,:self.horizon], out[:,self.horizon:]



class HorizonTCNBlock(nn.Module):
    """Lightweight horizon-axis TCN block, matching the exposure decoder idea.

    Input / output shape: [B, H, D].  It lets each future horizon build its
    query from the complete 20-week future-context pattern before attending
    to historical encoder states.
    """
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        z = x.transpose(1, 2)
        z = self.drop(F.gelu(self.conv1(z)))
        z = self.drop(F.gelu(self.conv2(z)))
        z = z.transpose(1, 2)
        m = min(z.shape[1], residual.shape[1])
        return self.norm(residual[:, :m, :] + z[:, :m, :])


class DecoderEpinet(nn.Module):
    """z-conditioned epistemic residual over stop-gradient decoder features.

    The base decoder feature is detached before entering this module. Therefore:
      * the decoder epinet learns from the demand loss;
      * z affects the joint 20-week prediction;
      * this epistemic branch cannot update the base future TCN / cross-attention.

    This mirrors the encoder epinet use of phi = h_t.detach().
    """
    def __init__(self, d_decoder=32, context_dim=2, d_z=16,
                 hidden=64, prior_scale=0.20):
        super().__init__()
        self.d_z = d_z
        self.prior_scale = prior_scale
        in_dim = d_decoder + context_dim + d_z

        # Output d_z coefficients for mu and alpha at every horizon.
        self.learnable = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, 2 * d_z),
        )
        self.prior = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_z),
        )
        for p in self.prior.parameters():
            p.requires_grad = False

        # Start the learned residual at zero; fixed prior still supplies variation.
        nn.init.zeros_(self.learnable[-1].weight)
        nn.init.zeros_(self.learnable[-1].bias)

    def forward(self, decoder_feature, future_context, z):
        B, H, _ = decoder_feature.shape
        z_h = z[:, None, :].expand(B, H, -1)

        # Critical stop-gradient: z branch reads decoder information but cannot
        # reshape the deterministic decoder representation.
        epi_in = torch.cat([
            decoder_feature.detach(),
            future_context.detach(),
            z_h,
        ], dim=-1)

        learned = self.learnable(epi_in).view(B, H, 2, self.d_z)
        prior = self.prior(epi_in).view(B, H, 2, self.d_z)

        learned = torch.einsum('bhkd,bd->bhk', learned, z)
        prior = torch.einsum('bhkd,bd->bhk', prior, z) * self.prior_scale
        out = learned + prior
        return out[:, :, 0], out[:, :, 1]


class DecoderPeakCrossAttention(nn.Module):
    """Deterministic future-TCN + peak-aware cross-attention demand decoder.

    Important ENN separation:
      * z does NOT enter the base future TCN or the base attention query;
      * the resulting decoder state is later passed through stop-gradient to
        DecoderEpinet together with z.

    Thus the decoder can inform the z branch, while the z branch cannot update
    or distort the deterministic decoder representation.
    """
    def __init__(self, d_model=32, context_dim=2, horizon=3, n_heads=4,
                 dropout=0.1, active_bias=0.50, peak_bias=0.75,
                 future_tcn_dilations=(1, 2, 4)):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.horizon = horizon
        self.active_bias = active_bias
        self.peak_bias = peak_bias

        self.future_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.horizon_emb = nn.Embedding(horizon, d_model)

        # Same core idea as the exposure decoder: future context is first
        # modeled along the horizon axis, then used as cross-attention queries.
        self.future_tcn = nn.ModuleList([
            HorizonTCNBlock(
                d_model=d_model,
                kernel_size=3,
                dilation=d,
                dropout=dropout,
            )
            for d in future_tcn_dilations
        ])

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _future_peak_score(self, future_context):
        # The last three columns are already log1p(exposure_hat), so do not
        # apply a second log1p here.
        if future_context.shape[-1] >= 3:
            return future_context[:, :, -3:].clamp(min=0).mean(dim=-1)
        return future_context.new_zeros(future_context.shape[:2])

    def forward(self, H_enc, future_context, b_t=None, peak_score=None,
                return_attn=False):
        B, T, D = H_enc.shape
        H = future_context.shape[1]
        device = H_enc.device

        horizon_ids = torch.arange(H, device=device).clamp(max=self.horizon - 1)
        q0 = self.future_proj(future_context)
        q0 = q0 + self.horizon_emb(horizon_ids)[None, :, :]
        for block in self.future_tcn:
            q0 = block(q0)

        q = self.q_proj(q0).view(B, H, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(H_enc).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(H_enc).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        if b_t is not None:
            scores = scores + self.active_bias * b_t.float()[:, None, None, :]

        if peak_score is not None:
            peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
            f_peak = self._future_peak_score(future_context)
            f_peak = f_peak / (f_peak.max(dim=1, keepdim=True)[0] + 1e-6)
            scores = scores + (
                self.peak_bias
                * f_peak[:, None, :, None]
                * peak_norm[:, None, None, :]
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, H, D)
        out = self.out_proj(out)
        dec_h = self.norm(q0 + out)

        if return_attn:
            return dec_h, attn.mean(dim=1)
        return dec_h


class TCN_ENN(nn.Module):
    """Demand v12c1.

    Changes from v12c0:
      1. Future context passes through a horizon TCN before cross-attention,
         matching the exposure decoder design.
      2. The base decoder is deterministic: z no longer enters its attention query.
      3. Decoder features are connected to z through DecoderEpinet, but with
         stop-gradient, just like phi = h_t.detach() for the encoder epinet.

    There is still no internal exposure decoder. External total / buy-box /
    in-stock exposure hats remain the last three future-context columns.
    """
    def __init__(self, input_dim=34, context_dim=2, d_model=32,
                 d_z=16, horizon=3, prior_scale=0.3,
                 use_stock_decoder=False):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.context_dim = context_dim
        self.use_stock_decoder = False
        self.stock_decoder = None

        self.encoder = TCNSparseAttnEncoder(input_dim, d_model, horizon)
        self.z_generator = ContextZGenerator(d_model, context_dim, d_z, horizon)
        self.epinet = Epinet(d_model, d_z, horizon, prior_scale)

        self.peak_decoder = DecoderPeakCrossAttention(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            n_heads=4,
            dropout=0.1,
            future_tcn_dilations=(1, 2, 4),
        )

        # Deterministic peak path: no z here.
        deterministic_peak_in = d_model + context_dim
        self.peak_mu_head = nn.Sequential(
            nn.Linear(deterministic_peak_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.peak_gate_head = nn.Sequential(
            nn.Linear(deterministic_peak_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # z-conditioned decoder residual, fed by stop-gradient decoder states.
        self.decoder_epinet = DecoderEpinet(
            d_decoder=d_model,
            context_dim=context_dim,
            d_z=d_z,
            hidden=64,
            prior_scale=0.20,
        )
        self.decoder_epinet_scale = 0.35

        nn.init.zeros_(self.peak_mu_head[-1].weight)
        nn.init.constant_(self.peak_mu_head[-1].bias, -4.0)
        nn.init.zeros_(self.peak_gate_head[-1].weight)
        nn.init.constant_(self.peak_gate_head[-1].bias, -1.5)
        self.peak_decoder_scale = 1.0

    def _external_exposure_log_hat(self, future_context):
        if future_context.shape[-1] >= 3:
            return future_context[:, :, -3:].clamp(min=0.0)
        B, H, _ = future_context.shape
        return torch.zeros(
            B, H, 3,
            device=future_context.device,
            dtype=future_context.dtype,
        )

    def _augment_context_with_stock_hat(self, h_t, future_context, *args, **kwargs):
        return future_context, self._external_exposure_log_hat(future_context)

    def _deterministic_decoder_component(self, H_enc, future_context,
                                         b_t, peak_score):
        """Compute base decoder state and deterministic positive peak component."""
        dec_h = self.peak_decoder(
            H_enc,
            future_context,
            b_t=b_t,
            peak_score=peak_score,
        )
        dec_in = torch.cat([dec_h, future_context], dim=-1)
        peak_mu = F.softplus(self.peak_mu_head(dec_in).squeeze(-1))
        peak_gate = torch.sigmoid(self.peak_gate_head(dec_in).squeeze(-1))
        return dec_h, self.peak_decoder_scale * peak_mu, peak_gate

    def _combine_one_z(self, mu_base, alpha_base, phi, dec_h,
                       future_context, deterministic_peak_mu,
                       deterministic_peak_gate, z):
        """Create one coherent 20-week prediction for one shared z sample."""
        # Encoder epinet: phi was detached before this call.
        mu_e, al_e = self.epinet(phi, z)

        # Decoder epinet: decoder feature is detached internally, so this branch
        # cannot update future TCN / cross-attention parameters.
        dec_mu_e, dec_al_e = self.decoder_epinet(dec_h, future_context, z)
        dec_mu_e = self.decoder_epinet_scale * dec_mu_e
        dec_al_e = self.decoder_epinet_scale * dec_al_e

        mu_normal = F.softplus(mu_base + mu_e + dec_mu_e)
        mu = mu_normal + deterministic_peak_gate * deterministic_peak_mu
        alpha = F.softplus(alpha_base + al_e + dec_al_e) + 1e-4
        return mu, alpha

    def forward(self, x, future_context, nZ=8, *args, **kwargs):
        H_enc, h_t, b_t, peak_score = self.encoder.encode(x)
        mu_base = F.softplus(self.encoder.base_head(h_t))
        alpha_base = F.softplus(self.encoder.alpha_head(h_t)) + 1e-4

        # Encoder base feature is available to the epinet but cannot be updated
        # through the z branch.
        phi = h_t.detach()
        z_mean, z_std = self.z_generator(phi, future_context)
        z_reg = 0.001 * (z_mean**2 + z_std**2).mean()

        # Deterministic decoder is computed once per batch, not once per z.
        dec_h, peak_mu, peak_gate = self._deterministic_decoder_component(
            H_enc, future_context, b_t, peak_score,
        )

        preds = []
        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps
            mu, alpha = self._combine_one_z(
                mu_base=mu_base,
                alpha_base=alpha_base,
                phi=phi,
                dec_h=dec_h,
                future_context=future_context,
                deterministic_peak_mu=peak_mu,
                deterministic_peak_gate=peak_gate,
                z=z,
            )
            preds.append((mu, alpha))

        stock_log_hat = self._external_exposure_log_hat(future_context)
        return preds, z_reg, stock_log_hat

    def predict(self, x, future_context, M=50, return_stock=False,
                *args, **kwargs):
        self.eval()
        with torch.no_grad():
            H_enc, h_t, b_t, peak_score = self.encoder.encode(x)
            mu_base = F.softplus(self.encoder.base_head(h_t))
            alpha_base = F.softplus(self.encoder.alpha_head(h_t)) + 1e-4
            phi = h_t.detach()
            z_mean, z_std = self.z_generator(phi, future_context)

            dec_h, peak_mu, peak_gate = self._deterministic_decoder_component(
                H_enc, future_context, b_t, peak_score,
            )

            samples = []
            for _ in range(M):
                eps = torch.randn_like(z_mean)
                z = z_mean + z_std * eps
                mu, alpha = self._combine_one_z(
                    mu_base=mu_base,
                    alpha_base=alpha_base,
                    phi=phi,
                    dec_h=dec_h,
                    future_context=future_context,
                    deterministic_peak_mu=peak_mu,
                    deterministic_peak_gate=peak_gate,
                    z=z,
                )
                dist = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp(min=1e-4),
                    probs=(mu * alpha / (1 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                samples.append(dist.sample().float())

            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)
            stock_log_hat = self._external_exposure_log_hat(future_context)

        if return_stock:
            return p50, p70, stock_log_hat
        return p50, p70


# =====================================================
# 4. Loss
# =====================================================

def negbin_nll_elementwise(y, mu, alpha):
    eps = 1e-6
    r   = (1.0/alpha).clamp(min=eps)
    p   = (mu*alpha/(1+mu*alpha)).clamp(eps, 1-eps)
    return -(
        torch.lgamma(y+r) - torch.lgamma(r) - torch.lgamma(y+1)
        + r*torch.log(1-p) + y*torch.log(p)
    )


def tail_weighted_negbin_nll(y, mu, alpha, beta_tail=0.5):
    nll    = negbin_nll_elementwise(y, mu, alpha)
    weight = 1.0 + beta_tail * torch.log1p(y)
    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def pinball_elementwise(y, pred, q):
    d = y - pred
    return torch.max(q * d, (q - 1) * d)


def pinball(y, pred, q):
    return pinball_elementwise(y, pred, q).mean()


def weighted_pinball_loss(y, p50, p70, beta_active=2.0, beta_tail_q=0.30, tau_low=0.7, tau_high=0.9):
    """Active/tail-aware shifted quantile pinball loss.

    This is added on top of the existing NB NLL, not used as a replacement.
    It directly targets active-week magnitude calibration and upper-quantile
    underforecasting.
    """
    active = (y > 0).float()
    w = 1.0 + beta_active * active + beta_tail_q * torch.log1p(y.clamp(min=0))
    # v12c: p50 variable stores q70, p70 variable stores q90.
    l50 = pinball_elementwise(y, p50, tau_low)
    l70 = pinball_elementwise(y, p70, tau_high)
    return ((l50 + l70) * w).sum() / w.sum().clamp(min=1.0)


def active_underforecast_loss(y, mu, log_scale=True):
    """Penalize underforecasting only when the true future demand is active.

    This addresses the observed issue: occurrence is learned, but active magnitude
    is still too conservative.
    """
    active = (y > 0).float()
    if active.sum() <= 0:
        return y.new_tensor(0.0)
    if log_scale:
        under = torch.relu(torch.log1p(y.clamp(min=0)) - torch.log1p(mu.clamp(min=0)))
    else:
        under = torch.relu(y - mu)
    return (active * under).sum() / active.sum().clamp(min=1.0)


def active_overforecast_loss(y, mu, log_scale=True):
    """Penalize overforecasting only when the true future demand is active.

    Symmetric mirror of active_underforecast_loss -- same active (y > 0) mask,
    same log1p scale, just the opposite direction (mu > y instead of y > mu).
    Kept as a separate, independently-weighted term (rather than folded into
    a single symmetric loss) so lambda_under and lambda_over can be tuned or
    disabled independently for ablation.
    """
    active = (y > 0).float()
    if active.sum() <= 0:
        return y.new_tensor(0.0)
    if log_scale:
        over = torch.relu(torch.log1p(mu.clamp(min=0)) - torch.log1p(y.clamp(min=0)))
    else:
        over = torch.relu(mu - y)
    return (active * over).sum() / active.sum().clamp(min=1.0)


# =====================================================
# 5. Diagnostics
# =====================================================

def occurrence_probe_linear_nonlinear(h_ts, ys):
    """
    Probe whether future occurrence is linearly or nonlinearly readable from h_t.
    Targets:
      any_active: at least one positive demand in horizon
      next4_active: at least one positive demand in first 4 weeks
      active_rate_high: horizon active rate above median
    """
    targets = {
        "any_active": (ys > 0).any(axis=1),
        "next4_active": (ys[:, :min(4, ys.shape[1])] > 0).any(axis=1),
    }

    active_rate = (ys > 0).mean(axis=1)
    median_rate = np.median(active_rate)
    targets["active_rate_high"] = active_rate > median_rate

    rows = []

    for target_name, y_bin in targets.items():
        y_bin = y_bin.astype(int)

        if y_bin.sum() < 10 or (len(y_bin) - y_bin.sum()) < 10:
            rows.append({
                "target": target_name,
                "positive_rate": y_bin.mean(),
                "linear_auc": np.nan,
                "nonlinear_auc": np.nan,
                "nonlinear_gain": np.nan,
                "note": "skip: class imbalance",
            })
            continue

        try:
            linear_clf = LogisticRegression(max_iter=500, C=1.0)
            linear_clf.fit(h_ts, y_bin)
            linear_auc = roc_auc_score(y_bin, linear_clf.predict_proba(h_ts)[:, 1])
        except Exception:
            linear_auc = np.nan

        try:
            nonlinear_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            nonlinear_clf.fit(h_ts, y_bin)
            nonlinear_auc = roc_auc_score(y_bin, nonlinear_clf.predict_proba(h_ts)[:, 1])
        except Exception:
            nonlinear_auc = np.nan

        rows.append({
            "target": target_name,
            "positive_rate": y_bin.mean(),
            "linear_auc": linear_auc,
            "nonlinear_auc": nonlinear_auc,
            "nonlinear_gain": nonlinear_auc - linear_auc
                if np.isfinite(linear_auc) and np.isfinite(nonlinear_auc)
                else np.nan,
            "note": "",
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("OCCURRENCE PROBE: LINEAR VS NONLINEAR")
    print("=" * 60)
    print(out)

    print("\nHow to read:")
    print("  high linear AUC: occurrence signal is linearly readable from h_t")
    print("  nonlinear AUC >> linear AUC: h_t contains occurrence signal, but in nonlinear form")
    print("  both low: encoder may not capture occurrence well")

    return out



def diagnose_encoder(model, va_ld):
    """
    诊断 encoder（h_t）的质量：
    1. h_t 能区分活跃/非活跃样本的能力（AUC）
    2. h_t 对 magnitude 的预测力（R²）
    3. mu_base 和真实需求的对比
    """
    print("\n" + "="*60)
    print("ENCODER DIAGNOSIS")
    print("="*60)

    model.eval()
    h_ts, ys, mu_bases = [], [], []

    with torch.no_grad():
        for b in va_ld:
            mu_base, alpha_base, h_t = model.encoder(b["x"])
            h_ts.append(h_t.numpy())
            ys.append(b["y"].numpy())
            mu_bases.append(mu_base.numpy())

    h_ts     = np.concatenate(h_ts)      # [N, d_model]
    ys       = np.concatenate(ys)        # [N, horizon]
    mu_bases = np.concatenate(mu_bases)  # [N, horizon]

    occurrence_probe_df = occurrence_probe_linear_nonlinear(h_ts, ys)

    # 1. occurrence 判别能力
    has_active = (ys > 0).any(axis=1)
    if has_active.sum() > 10 and (~has_active).sum() > 10:
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(h_ts, has_active.astype(int))
            auc = roc_auc_score(has_active, clf.predict_proba(h_ts)[:,1])
            print(f"h_t → occurrence AUC: {auc:.3f}")
            if auc < 0.6:
                print("  ← 差：encoder 对 occurrence 判别能力不足")
            elif auc < 0.75:
                print("  ← 一般：有改进空间")
            else:
                print("  ← 好：encoder 对 occurrence 有判别能力")
        except Exception as e:
            print(f"AUC 计算失败: {e}")

    # 2. magnitude 预测力
    active_mask  = (ys > 0).any(axis=1)
    y_mean_active = ys[active_mask].mean(axis=1)
    h_active      = h_ts[active_mask]

    if len(h_active) > 20:
        try:
            reg = Ridge()
            reg.fit(h_active, np.log1p(y_mean_active))
            r2  = r2_score(np.log1p(y_mean_active), reg.predict(h_active))
            print(f"h_t → log(magnitude) R²: {r2:.3f}")
            if r2 < 0.1:
                print("  ← 差：encoder 对 magnitude 几乎没有预测力")
            elif r2 < 0.3:
                print("  ← 一般：有改进空间")
            else:
                print("  ← 好：encoder 对 magnitude 有预测力")
        except Exception as e:
            print(f"R² 计算失败: {e}")

    # 3. mu_base vs 真实需求
    active_weeks_mask = ys > 0
    if active_weeks_mask.sum() > 0:
        true_mean  = ys[active_weeks_mask].mean()
        mu_mean    = mu_bases[active_weeks_mask].mean()
        print(f"\nActive weeks comparison:")
        print(f"  true demand mean : {true_mean:.2f}")
        print(f"  mu_base mean     : {mu_mean:.2f}")
        print(f"  ratio (mu/true)  : {mu_mean/max(true_mean,1e-8):.3f}")
        if mu_mean / max(true_mean, 1e-8) < 0.3:
            print("  ← mu_base 严重低估，magnitude 学习有问题")
        elif mu_mean / max(true_mean, 1e-8) < 0.7:
            print("  ← mu_base 偏低，有改进空间")
        else:
            print("  ← mu_base 合理")

    # 4. z 的质量
    z_means, z_stds = [], []
    with torch.no_grad():
        for b in va_ld:
            _, _, h_t = model.encoder(b["x"])
            phi = h_t.detach()

            # Stock-decoder version:
            # z_generator expects future_context augmented with predicted stock_hat.
            if hasattr(model, "_augment_context_with_stock_hat"):
                fc_for_z, _ = model._augment_context_with_stock_hat(h_t, b["future_context"])
            else:
                fc_for_z = b["future_context"]

            zm, zs = model.z_generator(phi, fc_for_z)
            z_means.append(zm.numpy())
            z_stds.append(zs.numpy())

    z_means = np.concatenate(z_means)
    z_stds  = np.concatenate(z_stds)
    print(f"\nz quality:")
    print(f"  z_mean abs mean : {np.abs(z_means).mean():.3f} (should be small)")
    print(f"  z_std mean      : {z_stds.mean():.3f} (should be ~1)")
    if z_stds.mean() > 3.0:
        print("  ← z_std 过大，后验扩张，joint prediction 不稳定")
    elif z_stds.mean() < 0.1:
        print("  ← z_std 过小，z 失去不确定性表达能力")
    else:
        print("  ← z_std 合理")

    print("="*60)


def diagnose_training_batch(b, preds, epoch, bi, n_diag_batches=3):
    """Print diagnostics for the first few batches."""
    if bi >= n_diag_batches:
        return
    y = b["y"]
    active_cnt = (y > 0).sum().item()
    total_cnt  = y.numel()
    mu_mean    = torch.stack([mu for mu, _ in preds], dim=0).mean().item()
    y_active_mean = y[y > 0].mean().item() if active_cnt > 0 else 0.0
    print(
        f"  [batch {bi}] active={active_cnt}/{total_cnt} "
        f"({100*active_cnt/total_cnt:.1f}%) "
        f"mu_mean={mu_mean:.2f} "
        f"y_active_mean={y_active_mean:.2f}"
    )


# =====================================================
# 6. Training
# =====================================================

def train(
    model,
    tr_ld,
    va_ld,
    epochs=60,
    nZ=8,
    lr=1e-3,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.0,
    lambda_stock_mean_weight=0.0,
):
    """
    Train demand model with external predicted exposure hats already in future_context.
    No internal exposure decoder and no true future DPH are passed into the model.
    lambda_stock arguments are kept only for API compatibility and are ignored.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0

        for bi, b in enumerate(tr_ld):
            x = b["x"]
            fc = b["future_context"]
            y = b["y"]

            preds, z_reg, _ = model(x, fc, nZ=nZ)

            nll_loss = sum(
                tail_weighted_negbin_nll(y, mu, alpha, beta_tail=beta_tail)
                for mu, alpha in preds
            ) / nZ

            mu_stack = torch.stack([mu for mu, _ in preds], dim=1)
            # v12c0 NOQ: no pinball / quantile training loss.
            # Keep only NB NLL + active-underforecast + z regularization.
            q_loss = y.new_tensor(0.0)
            mu_mean_train = mu_stack.mean(dim=1)
            under_loss = active_underforecast_loss(y, mu_mean_train, log_scale=True)

            loss = nll_loss + lambda_under * under_loss + lambda_z_reg * z_reg

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

            if epoch == 0:
                diagnose_training_batch(b, preds, epoch, bi)

        sch.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va_ld:
                # v12c0 LOCALCSV FIX:
                # predict() returns only sampled quantiles (p50/p70).
                # Validation NLL needs mu/alpha, so use forward() and average over z samples.
                preds_val, _, _ = model(b["x"], b["future_context"], nZ=8)
                mu_mean_val = torch.stack([mu for mu, _ in preds_val], dim=1).mean(dim=1)
                alpha_mean_val = torch.stack([alpha for _, alpha in preds_val], dim=1).mean(dim=1)
                vl += tail_weighted_negbin_nll(b["y"], mu_mean_val, alpha_mean_val, beta_tail=beta_tail).item()
        vl /= max(1, len(va_ld))

        improved = vl < best_val
        if improved:
            best_val = vl
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | "
            f"train={tr_loss/max(1,len(tr_ld)):.4f} | "
            f"val={vl:.4f} | "
            f"beta_tail={beta_tail}"
            + (" *" if improved else "")
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_sd:
        model.load_state_dict(best_sd)
    print(f"Best val: {best_val:.4f}")


# =====================================================
# 7. Evaluation and forecast generation
# =====================================================

def evaluate(model, va_ld, M=100):
    all_y, all_p50, all_p70 = [], [], []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b["x"], b["future_context"], M=M)
            all_y.append(b["y"].numpy())
            all_p50.append(p50.numpy())
            all_p70.append(p70.numpy())
    y = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)
    yt = torch.tensor(y)
    return {
        # v12c0: reported p50/p70 are true q50/q70.
        "pinball50": pinball(yt, torch.tensor(p50), 0.5).item(),
        "pinball70": pinball(yt, torch.tensor(p70), 0.7).item(),
    }


def generate_forecast_df(model, va_ld, M=50):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70, stock_log_hat = model.predict(
                b["x"],
                b["future_context"],
                M=M,
                return_stock=True,
            )
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "fcst_week_index": h + 1,
                        "fbi_demand": b["y"][i, h].item(),
                        "our_price": b["our_price"][i, h].item(),
                        "true_amt": b["y"][i, h].item() * b["our_price"][i, h].item(),
                        "pkg_volume": b["pkg_volume"][i, h].item(),
                        "true_size": b["y"][i, h].item() * b["pkg_volume"][i, h].item(),

                        # True DPH values below are output-only diagnostics, never model inputs.
                        "true_future_total_dph": b["future_total_dph"][i, h].item() if "future_total_dph" in b else np.nan,
                        "true_future_buy_box_dph": b["future_buy_box_dph"][i, h].item() if "future_buy_box_dph" in b else np.nan,
                        "true_future_instock": b["future_instock"][i, h].item() if "future_instock" in b else np.nan,

                        # These are the external predicted exposure hats appended to future_context.
                        "pred_total_dph_hat": torch.expm1(stock_log_hat[i, h, 0]).item() if stock_log_hat is not None else np.nan,
                        "pred_buy_box_dph_hat": torch.expm1(stock_log_hat[i, h, 1]).item() if stock_log_hat is not None else np.nan,
                        "pred_instock_dph_hat": torch.expm1(stock_log_hat[i, h, 2]).item() if stock_log_hat is not None else np.nan,
                        "pred_total_dph_log_hat": stock_log_hat[i, h, 0].item() if stock_log_hat is not None else np.nan,
                        "pred_buy_box_dph_log_hat": stock_log_hat[i, h, 1].item() if stock_log_hat is not None else np.nan,
                        "pred_instock_log_hat": stock_log_hat[i, h, 2].item() if stock_log_hat is not None else np.nan,

                        "scot_oos": b["oos"][i, h].item(),
                        "oos": b["oos"][i, h].item(),
                        "oos_status": b["oos"][i, h].item(),
                        "p50_amxl": p50[i, h].item(),
                        "p70_amxl": p70[i, h].item(),
                    })
    return pd.DataFrame(rows)


def generate_diagnostic_df(model, va_ld, M=100, threshold=0.5):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b["x"], b["future_context"], M=M)
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    y_val = b["y"][i, h].item()
                    p50_val = p50[i, h].item()
                    p70_val = p70[i, h].item()
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "horizon": h + 1,
                        "y": y_val,
                        "p50": p50_val,
                        "p70": p70_val,
                        "true_active": int(y_val > 0),
                        "pred_active_p50": int(p50_val > threshold),
                        "pred_active_p70": int(p70_val > threshold),
                    })
    return pd.DataFrame(rows)


def underbias_diagnosis(diag_df, pred_col="p70", threshold=0.5):
    y    = diag_df["y"].values
    pred = diag_df[pred_col].values
    ta   = y > 0
    pa   = pred > threshold
    tp = np.sum(ta & pa); fp = np.sum(~ta & pa)
    fn = np.sum(ta & ~pa); tn = np.sum(~ta & ~pa)
    recall    = tp / max(1, tp+fn)
    precision = tp / max(1, tp+fp)
    f1        = 2*precision*recall / max(1e-8, precision+recall)
    total_under = np.maximum(y-pred, 0).sum()
    missed_under    = np.maximum(y[ta & ~pa] - pred[ta & ~pa], 0).sum()
    magnitude_under = np.maximum(y[ta & pa]  - pred[ta & pa],  0).sum()
    ratio = pred[ta & pa] / np.maximum(y[ta & pa], 1e-8) if (ta & pa).sum() > 0 else np.array([np.nan])
    return pd.DataFrame([{
        "pred_col": pred_col, "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "occurrence_recall": recall, "occurrence_precision": precision, "occurrence_f1": f1,
        "total_underbias": total_under,
        "underbias_rate": total_under / max(1e-8, y.sum()),
        "missed_active_share": missed_under / max(1e-8, total_under),
        "magnitude_under_share": magnitude_under / max(1e-8, total_under),
        "avg_pred_over_true_when_active_predicted": np.nanmean(ratio),
        "median_pred_over_true_when_active_predicted": np.nanmedian(ratio),
    }])


def magnitude_gap(diag_df):
    df = diag_df[diag_df["true_active"]==1].copy()
    if len(df) == 0: return pd.DataFrame()
    y, p50, p70 = df["y"].values, df["p50"].values, df["p70"].values
    out = pd.DataFrame([{
        "true_active_mean": y.mean(),
        "p50_active_mean": p50.mean(),
        "p70_active_mean": p70.mean(),
        "p50_pct_of_true": p50.mean()/max(y.mean(),1e-8),
        "p70_pct_of_true": p70.mean()/max(y.mean(),1e-8),
        "p50_gap": y.mean()-p50.mean(),
        "p70_gap": y.mean()-p70.mean(),
    }])
    print("\n[Magnitude Gap - Active weeks only]")
    print(out.T)
    return out


# =====================================================
# 8. Run
# =====================================================

def filter_extreme_asins(data_high, demand_col="fbi_demand", asin_col="asin", q=0.99):
    df = data_high.copy()
    df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce").fillna(0).clip(lower=0)
    pos = df.loc[df[demand_col]>0, demand_col]
    if len(pos) == 0: return df, pd.DataFrame(), np.nan
    cap = float(pos.quantile(q))
    asin_peak = df.groupby(asin_col)[demand_col].max().reset_index(name="asin_max")
    bad_asins = asin_peak.loc[asin_peak["asin_max"]>cap, asin_col]
    clean = df[~df[asin_col].isin(bad_asins)].copy()
    print(f"\nExtreme ASIN filter (p{int(q*100)}={cap:.1f}): removed {bad_asins.nunique()} ASINs")
    print(f"Clean ASINs: {clean[asin_col].nunique()} | Clean rows: {len(clean)}")
    return clean, asin_peak[asin_peak[asin_col].isin(bad_asins)], cap


def run_nb_high_sparse(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.05,
    lambda_stock_mean_weight=0.30,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
):
    print("="*70)
    print("NB-v2 HIGH-SPARSE | leak-fix + soft-mask + dilation13 + early-stop + z-reg")
    print("="*70)

    data_small, _ = add_zero_rate_group(
        prepare_data_sample(data_raw1, n_asins), zero_thresholds
    )
    data_high = data_small[data_small["zero_group"]=="high_sparse"].copy()

    if remove_extreme:
        data_high, _, _ = filter_extreme_asins(data_high, q=extreme_q)

    data, context_dim, context_cols = load_real_data(data_high, dph_cap_q=dph_cap_q)
    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs: {len(data)} | Zero rate: {(all_demand==0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val",   horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    print(f"Train: {len(tr_ds)} | Val: {len(va_ds)}")

    model = TCN_ENN(25, context_dim, d_model, d_z, horizon, prior_scale)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | lambda_under={lambda_under} | q_active_weight={q_active_weight} | q_tail_weight={q_tail_weight} | patience={patience}")

    train(model, tr_ld, va_ld,
          epochs=epochs, nZ=8, lr=1e-3,
          lambda_q=lambda_q, beta_tail=beta_tail,
          patience=patience, lambda_z_reg=lambda_z_reg, lambda_stock=lambda_stock, lambda_stock_mean_weight=lambda_stock_mean_weight)

    # Encoder diagnostics.
    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball: P50(q50)={metrics['pinball50']:.4f} | P70(q70)={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "high_sparse_nb_v2"

    diag_df  = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:"); print(diag_p50.T)
    print("\nUnderbias P70:"); print(diag_p70.T)

    return {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
    }



# =====================================================
# 9. Final WAPE summary
# =====================================================

def run_final_wape(result, remove_oos_dp=True, source="lp"):
    """
    Compute final boss-style WAPE from result["forecast_df"].

    This function expects these notebook functions to already exist:
      - calculate_wape_using_lp_oos2
      - quick_error_check
    """
    if "forecast_df" not in result:
        raise KeyError('result must contain "forecast_df".')

    if "calculate_wape_using_lp_oos2" not in globals():
        raise RuntimeError("calculate_wape_using_lp_oos2 is not defined.")

    if "quick_error_check" not in globals():
        raise RuntimeError("quick_error_check is not defined.")

    forecast_df = result["forecast_df"]

    wape_df = calculate_wape_using_lp_oos2(
        forecast_df,
        [0.5, 0.7],
        remove_oos_dp=remove_oos_dp,
        source=source,
    )

    cols_p50 = [
        "p50_amxl_penalty",
        "p50_scot_penalty",
        "p50_amxl_overbias",
        "p50_scot_overbias",
        "p50_amxl_underbias",
        "p50_scot_underbias",
        "fbi_demand",
    ]

    cols_p70 = [
        "p70_amxl_penalty",
        "p70_scot_penalty",
        "p70_amxl_overbias",
        "p70_scot_overbias",
        "p70_amxl_underbias",
        "p70_scot_underbias",
        "fbi_demand",
    ]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

    print("\n" + "=" * 80)
    print("FINAL WAPE SUMMARY")
    print("=" * 80)

    print("\nP50 WAPE")
    print(p50_wape)
    print("P50 penalty diff:", p50_penalty_diff)

    print("\nP70 WAPE")
    print(p70_wape)
    print("P70 penalty diff:", p70_penalty_diff)

    return {
        "wape_df": wape_df,
        "p50_wape": p50_wape,
        "p70_wape": p70_wape,
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
    }


def run_nb_high_sparse_with_wape(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.05,
    lambda_stock_mean_weight=0.30,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    remove_oos_dp=True,
):
    """
    Run the full experiment and print final WAPE.
    """
    result = run_nb_high_sparse(
        data_raw1=data_raw1,
        n_asins=n_asins,
        zero_thresholds=zero_thresholds,
        prior_scale=prior_scale,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
    )

    wape_outputs = run_final_wape(
        result,
        remove_oos_dp=remove_oos_dp,
        source="lp",
    )

    result["wape_outputs"] = wape_outputs

    return result



# =====================================================
# 10. Sparse-group WAPE diagnostics
# =====================================================

def attach_zero_group_to_joined_df(joined_df, asin_stats):
    """
    Attach zero_rate and zero_group to the joined AMXL-SCOT forecast dataframe.
    """
    if asin_stats is None or len(asin_stats) == 0:
        return joined_df.copy()

    out = joined_df.copy()
    stats = asin_stats.copy()

    out["asin"] = out["asin"].astype(str)
    stats["asin"] = stats["asin"].astype(str)

    keep = [c for c in ["asin", "zero_rate", "zero_group"] if c in stats.columns]

    if "zero_group" not in keep:
        return out

    out = out.merge(
        stats[keep].drop_duplicates("asin"),
        on="asin",
        how="left",
    )

    return out


def summarize_wape_by_sparse_group(wape_df, joined_df_with_group):
    """
    Summarize boss-style WAPE by zero_group using the already-generated wape_df.
    This is diagnostic only; the main result remains the overall WAPE.
    """
    if "zero_group" not in joined_df_with_group.columns:
        print("zero_group not found. Skip sparse-group WAPE diagnostics.")
        return pd.DataFrame()

    key_cols = ["asin", "order_week", "zero_rate", "zero_group"]
    group_map = joined_df_with_group[key_cols].drop_duplicates(["asin", "order_week"]).copy()

    work = wape_df.copy()
    work["asin"] = work["asin"].astype(str)
    work["order_week"] = pd.to_datetime(work["order_week"])
    group_map["asin"] = group_map["asin"].astype(str)
    group_map["order_week"] = pd.to_datetime(group_map["order_week"])

    work = work.merge(group_map, on=["asin", "order_week"], how="left")

    total_demand_all = work["fbi_demand"].sum()
    total_rows_all = len(work)
    total_asins_all = work["asin"].nunique()

    rows = []

    for group_name, g in work.groupby("zero_group", dropna=False):
        denom = g["fbi_demand"].sum()

        rows.append({
            "zero_group": group_name,
            "n_rows": len(g),
            "n_asins": g["asin"].nunique(),
            "total_fbi_demand": denom,
            "true_mean": g["fbi_demand"].mean(),
            "p50_amxl_penalty": g["p50_amxl_penalty"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_penalty": g["p50_scot_penalty"].sum() / denom if denom > 0 else np.nan,
            "p50_bps_improvement": (
                (g["p50_scot_penalty"].sum() - g["p50_amxl_penalty"].sum()) / denom * 10000
                if denom > 0 else np.nan
            ),
            "p70_amxl_penalty": g["p70_amxl_penalty"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_penalty": g["p70_scot_penalty"].sum() / denom if denom > 0 else np.nan,
            "p70_bps_improvement": (
                (g["p70_scot_penalty"].sum() - g["p70_amxl_penalty"].sum()) / denom * 10000
                if denom > 0 else np.nan
            ),
            "p50_amxl_underbias": g["p50_amxl_underbias"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_underbias": g["p50_scot_underbias"].sum() / denom if denom > 0 else np.nan,
            "p50_amxl_overbias": g["p50_amxl_overbias"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_overbias": g["p50_scot_overbias"].sum() / denom if denom > 0 else np.nan,
            "p70_amxl_underbias": g["p70_amxl_underbias"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_underbias": g["p70_scot_underbias"].sum() / denom if denom > 0 else np.nan,
            "p70_amxl_overbias": g["p70_amxl_overbias"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_overbias": g["p70_scot_overbias"].sum() / denom if denom > 0 else np.nan,
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("SPARSE-GROUP WAPE DIAGNOSTICS")
    print("=" * 80)

    display_cols = [
        "zero_group",
        "n_asins",
        "n_rows",
        "total_fbi_demand",
        "total_amt",
        "total_size",
        "demand_share",
        "avg_total_demand_per_asin",
        "true_mean",
        "true_zero_rate",
        "p50_amxl_penalty",
        "p50_scot_penalty",
        "p50_bps_improvement",
        "p70_amxl_penalty",
        "p70_scot_penalty",
        "p70_bps_improvement",
        "p50_amxl_underbias",
        "p50_scot_underbias",
        "p50_amxl_overbias",
        "p50_scot_overbias",
        "p70_amxl_underbias",
        "p70_scot_underbias",
        "p70_amxl_overbias",
        "p70_scot_overbias",
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    print(out[display_cols])

    return out



# =====================================================
# 10b. Sparse-group horizon decay diagnostics
# =====================================================

def _attach_zero_group_to_forecast_df_for_horizon_diag(forecast_df, asin_stats=None, data_raw1=None, zero_thresholds=(0.4, 0.7)):
    """Attach zero_rate / zero_group to forecast_df for horizon-level diagnostics."""
    out = forecast_df.copy()
    out["asin"] = out["asin"].astype(str)

    if "zero_group" in out.columns and "zero_rate" in out.columns:
        return out

    stats = None
    if asin_stats is not None and len(asin_stats) > 0:
        stats = asin_stats.copy()
    elif data_raw1 is not None and "asin" in data_raw1.columns and "fbi_demand" in data_raw1.columns:
        tmp = data_raw1.copy()
        tmp["asin"] = tmp["asin"].astype(str)
        stats = (
            tmp.groupby("asin", as_index=False)
            .agg(zero_rate=("fbi_demand", lambda x: (pd.to_numeric(x, errors="coerce").fillna(0) == 0).mean()))
        )
        low, high = zero_thresholds
        def _assign(z):
            if z < low:
                return "low_sparse"
            elif z < high:
                return "mid_sparse"
            return "high_sparse"
        stats["zero_group"] = stats["zero_rate"].apply(_assign)

    if stats is None or len(stats) == 0:
        print("No asin_stats/data_raw1 available. Sparse horizon diagnostics skipped.")
        return out

    stats["asin"] = stats["asin"].astype(str)
    keep = [c for c in ["asin", "zero_rate", "zero_group"] if c in stats.columns]
    out = out.merge(stats[keep].drop_duplicates("asin"), on="asin", how="left")
    return out


def summarize_sparse_horizon_decay(
    forecast_df,
    asin_stats=None,
    data_raw1=None,
    pred_cols=("p50_amxl", "p70_amxl"),
    true_col="fbi_demand",
    horizon_col="fcst_week_index",
    zero_thresholds=(0.4, 0.7),
    print_table=True,
):
    """Check whether predictions decay from h=1 to h=20 by sparse group.

    Returns:
      by_horizon: group x horizon true/pred/gap/WAPE table
      decay_summary: group-level h1/h20 decay and average gap summary

    Interpretation:
      pred_decay_pct < 0 means prediction declines over horizon.
      gap_mean = pred_mean - true_mean. Negative means under-prediction.
    """
    df = _attach_zero_group_to_forecast_df_for_horizon_diag(
        forecast_df, asin_stats=asin_stats, data_raw1=data_raw1, zero_thresholds=zero_thresholds
    )
    if "zero_group" not in df.columns:
        return {"by_horizon": pd.DataFrame(), "decay_summary": pd.DataFrame()}

    df = df.copy()
    df[horizon_col] = pd.to_numeric(df[horizon_col], errors="coerce").astype("Int64")
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce").fillna(0.0)
    for c in pred_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    rows = []
    group_cols = ["zero_group", horizon_col]
    for (zg, h), g in df.groupby(group_cols, dropna=False):
        base = {
            "zero_group": zg,
            "horizon": int(h) if pd.notna(h) else np.nan,
            "n_rows": len(g),
            "n_asins": g["asin"].nunique() if "asin" in g.columns else np.nan,
            "true_mean": g[true_col].mean(),
            "true_sum": g[true_col].sum(),
            "true_active_rate": (g[true_col] > 0).mean(),
        }
        for c in pred_cols:
            if c not in g.columns:
                continue
            err = g[c] - g[true_col]
            denom = g[true_col].sum()
            base[f"{c}_mean"] = g[c].mean()
            base[f"{c}_gap_mean"] = err.mean()
            base[f"{c}_abs_gap_mean"] = err.abs().mean()
            base[f"{c}_ratio_mean"] = g[c].mean() / max(g[true_col].mean(), 1e-8)
            base[f"{c}_wape"] = err.abs().sum() / denom if denom > 0 else np.nan
            base[f"{c}_underbias"] = np.maximum(g[true_col] - g[c], 0).sum() / denom if denom > 0 else np.nan
            base[f"{c}_overbias"] = np.maximum(g[c] - g[true_col], 0).sum() / denom if denom > 0 else np.nan
        rows.append(base)

    by_h = pd.DataFrame(rows).sort_values(["zero_group", "horizon"]).reset_index(drop=True)

    summary_rows = []
    for zg, g in by_h.groupby("zero_group", dropna=False):
        gg = g.sort_values("horizon")
        if len(gg) == 0:
            continue
        h1 = gg[gg["horizon"] == gg["horizon"].min()].iloc[0]
        hT = gg[gg["horizon"] == gg["horizon"].max()].iloc[0]
        row = {
            "zero_group": zg,
            "h_start": int(h1["horizon"]),
            "h_end": int(hT["horizon"]),
            "true_h1_mean": h1["true_mean"],
            "true_hEnd_mean": hT["true_mean"],
            "true_decay_pct": (hT["true_mean"] - h1["true_mean"]) / max(abs(h1["true_mean"]), 1e-8),
            "true_active_h1": h1["true_active_rate"],
            "true_active_hEnd": hT["true_active_rate"],
        }
        x = gg["horizon"].astype(float).values
        if len(x) >= 2:
            row["true_slope_per_h"] = float(np.polyfit(x, gg["true_mean"].astype(float).values, 1)[0])
        for c in pred_cols:
            mean_col = f"{c}_mean"
            gap_col = f"{c}_gap_mean"
            wape_col = f"{c}_wape"
            ub_col = f"{c}_underbias"
            ob_col = f"{c}_overbias"
            if mean_col not in gg.columns:
                continue
            row[f"{c}_h1_mean"] = h1[mean_col]
            row[f"{c}_hEnd_mean"] = hT[mean_col]
            row[f"{c}_decay_pct"] = (hT[mean_col] - h1[mean_col]) / max(abs(h1[mean_col]), 1e-8)
            row[f"{c}_avg_gap"] = gg[gap_col].mean() if gap_col in gg.columns else np.nan
            row[f"{c}_avg_abs_gap"] = gg[f"{c}_abs_gap_mean"].mean() if f"{c}_abs_gap_mean" in gg.columns else np.nan
            row[f"{c}_avg_wape"] = gg[wape_col].mean() if wape_col in gg.columns else np.nan
            row[f"{c}_avg_underbias"] = gg[ub_col].mean() if ub_col in gg.columns else np.nan
            row[f"{c}_avg_overbias"] = gg[ob_col].mean() if ob_col in gg.columns else np.nan
            if len(x) >= 2:
                row[f"{c}_slope_per_h"] = float(np.polyfit(x, gg[mean_col].astype(float).values, 1)[0])
                row[f"{c}_decays"] = bool(row[f"{c}_slope_per_h"] < 0)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if print_table:
        print("\n" + "=" * 100)
        print("SPARSE-GROUP HORIZON DECAY / GAP DIAGNOSTICS")
        print("=" * 100)
        show_cols = [
            "zero_group", "true_h1_mean", "true_hEnd_mean", "true_decay_pct", "true_slope_per_h",
            "p50_amxl_h1_mean", "p50_amxl_hEnd_mean", "p50_amxl_decay_pct", "p50_amxl_slope_per_h",
            "p50_amxl_avg_gap", "p50_amxl_avg_wape", "p50_amxl_avg_underbias", "p50_amxl_decays",
            "p70_amxl_h1_mean", "p70_amxl_hEnd_mean", "p70_amxl_decay_pct", "p70_amxl_slope_per_h",
            "p70_amxl_avg_gap", "p70_amxl_avg_wape", "p70_amxl_avg_underbias", "p70_amxl_decays",
        ]
        show_cols = [c for c in show_cols if c in summary.columns]
        print(summary[show_cols].round(4).to_string(index=False))

        print("\nBy-horizon compact view:")
        compact_cols = [
            "zero_group", "horizon", "true_mean", "true_active_rate",
            "p50_amxl_mean", "p50_amxl_gap_mean", "p50_amxl_wape",
            "p70_amxl_mean", "p70_amxl_gap_mean", "p70_amxl_wape",
        ]
        compact_cols = [c for c in compact_cols if c in by_h.columns]
        print(by_h[compact_cols].round(4).to_string(index=False))

    return {"by_horizon": by_h, "decay_summary": summary}



def summarize_h1_h20_magnitude_diagnostics(
    forecast_df,
    asin_stats=None,
    data_raw1=None,
    pred_cols=("p50_amxl", "p70_amxl"),
    true_col="fbi_demand",
    horizon_col="fcst_week_index",
    zero_thresholds=(0.4, 0.7),
    top_k_bad_h=8,
    print_table=True,
):
    """Horizon-by-horizon magnitude diagnostics for h=1..H.

    This is stricter than the decay table: it explicitly reports which horizons
    have the largest WAPE / underforecast bias, both overall and by sparse group.
    It also reports active-only true/pred means, which directly targets the
    active-magnitude underforecast issue.
    """
    df = _attach_zero_group_to_forecast_df_for_horizon_diag(
        forecast_df, asin_stats=asin_stats, data_raw1=data_raw1, zero_thresholds=zero_thresholds
    ).copy()
    if horizon_col not in df.columns or true_col not in df.columns:
        return {"by_horizon": pd.DataFrame(), "worst_horizons": pd.DataFrame()}

    df[horizon_col] = pd.to_numeric(df[horizon_col], errors="coerce")
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce").fillna(0.0)
    if "zero_group" not in df.columns:
        df["zero_group"] = "all"
    for c in pred_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    rows = []
    # include both overall and sparse-group breakdown
    group_specs = [("all", df)]
    for zg, g in df.groupby("zero_group", dropna=False):
        group_specs.append((str(zg), g))

    for group_name, gd in group_specs:
        for h, g in gd.groupby(horizon_col, dropna=False):
            if pd.isna(h):
                continue
            y = g[true_col].astype(float)
            active = y > 0
            base = {
                "group": group_name,
                "horizon": int(h),
                "n_rows": len(g),
                "n_asins": g["asin"].nunique() if "asin" in g.columns else np.nan,
                "true_mean": float(y.mean()),
                "true_sum": float(y.sum()),
                "true_active_rate": float(active.mean()),
                "true_active_mean": float(y[active].mean()) if active.any() else 0.0,
            }
            denom = max(float(y.sum()), 1e-8)
            for c in pred_cols:
                if c not in g.columns:
                    continue
                pred = g[c].astype(float)
                err = pred - y
                under = np.maximum(y - pred, 0)
                over = np.maximum(pred - y, 0)
                base[f"{c}_mean"] = float(pred.mean())
                base[f"{c}_gap"] = float(err.mean())
                base[f"{c}_ratio_pred_true"] = float(pred.mean() / max(y.mean(), 1e-8))
                base[f"{c}_wape"] = float(np.abs(err).sum() / denom)
                base[f"{c}_underbias"] = float(under.sum() / denom)
                base[f"{c}_overbias"] = float(over.sum() / denom)
                base[f"{c}_active_mean"] = float(pred[active].mean()) if active.any() else 0.0
                base[f"{c}_active_gap"] = float((pred[active] - y[active]).mean()) if active.any() else 0.0
                base[f"{c}_active_ratio_pred_true"] = float(pred[active].mean() / max(y[active].mean(), 1e-8)) if active.any() else np.nan
            rows.append(base)

    by_h = pd.DataFrame(rows).sort_values(["group", "horizon"]).reset_index(drop=True)

    worst_rows = []
    for group_name, gg in by_h.groupby("group", dropna=False):
        for c in pred_cols:
            wape_col = f"{c}_wape"
            ub_col = f"{c}_underbias"
            active_gap_col = f"{c}_active_gap"
            if wape_col in gg.columns:
                tmp = gg.sort_values(wape_col, ascending=False).head(top_k_bad_h)
                for _, r in tmp.iterrows():
                    worst_rows.append({
                        "group": group_name,
                        "pred_col": c,
                        "bad_by": "wape",
                        "horizon": int(r["horizon"]),
                        "value": float(r[wape_col]),
                        "true_mean": float(r["true_mean"]),
                        "pred_mean": float(r.get(f"{c}_mean", np.nan)),
                        "gap": float(r.get(f"{c}_gap", np.nan)),
                        "underbias": float(r.get(ub_col, np.nan)),
                        "active_gap": float(r.get(active_gap_col, np.nan)),
                    })
            if ub_col in gg.columns:
                tmp = gg.sort_values(ub_col, ascending=False).head(top_k_bad_h)
                for _, r in tmp.iterrows():
                    worst_rows.append({
                        "group": group_name,
                        "pred_col": c,
                        "bad_by": "underbias",
                        "horizon": int(r["horizon"]),
                        "value": float(r[ub_col]),
                        "true_mean": float(r["true_mean"]),
                        "pred_mean": float(r.get(f"{c}_mean", np.nan)),
                        "gap": float(r.get(f"{c}_gap", np.nan)),
                        "underbias": float(r.get(ub_col, np.nan)),
                        "active_gap": float(r.get(active_gap_col, np.nan)),
                    })
    worst = pd.DataFrame(worst_rows)

    if print_table:
        print("\n" + "=" * 100)
        print("H1-H20 MAGNITUDE DIAGNOSTICS: WHICH HORIZONS ARE BAD?")
        print("=" * 100)
        show_cols = [
            "group", "horizon", "true_mean", "true_active_rate", "true_active_mean",
            "p50_amxl_mean", "p50_amxl_gap", "p50_amxl_wape", "p50_amxl_underbias", "p50_amxl_active_ratio_pred_true",
            "p70_amxl_mean", "p70_amxl_gap", "p70_amxl_wape", "p70_amxl_underbias", "p70_amxl_active_ratio_pred_true",
        ]
        show_cols = [c for c in show_cols if c in by_h.columns]
        print(by_h[show_cols].round(4).to_string(index=False))
        if len(worst):
            print("\nWorst horizons by WAPE / underbias:")
            print(worst.round(4).to_string(index=False))

    return {"by_horizon": by_h, "worst_horizons": worst}


# =====================================================
# 10. Real SCOT alignment and WAPE
# =====================================================



def _canonical_asin_join_key(series):
    """Create a stable ASIN join key without changing model features."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\\.0+$", "", regex=True)

    # Normalize values that arrived through CSV as scientific notation/float.
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        rounded = numeric.loc[numeric_mask].round()
        integer_like = (numeric.loc[numeric_mask] - rounded).abs() < 1e-6
        integer_index = integer_like[integer_like].index
        if len(integer_index):
            s.loc[integer_index] = rounded.loc[integer_index].astype("Int64").astype("string")

    return s.astype(str)


def _normalize_week_to_sunday(series):
    dt = pd.to_datetime(series, errors="coerce").dt.normalize()
    return dt + pd.to_timedelta((6 - dt.dt.dayofweek) % 7, unit="D")


def _print_join_key_debug(forecast_df, scot):
    """Print raw/canonical ASIN and week overlap before any merge."""
    forecast_raw_asins = set(forecast_df["asin"].dropna().astype(str).unique())
    scot_raw_asins = set(scot["asin"].dropna().astype(str).unique())

    forecast_canonical = set(
        _canonical_asin_join_key(forecast_df["asin"]).dropna().unique()
    )
    scot_canonical = set(
        _canonical_asin_join_key(scot["asin"]).dropna().unique()
    )

    forecast_weeks = set(
        pd.to_datetime(forecast_df["order_week"], errors="coerce")
        .dropna().dt.normalize().unique()
    )
    scot_weeks = set(
        pd.to_datetime(scot["order_week"], errors="coerce")
        .dropna().dt.normalize().unique()
    )
    forecast_sunday_weeks = set(
        _normalize_week_to_sunday(forecast_df["order_week"])
        .dropna().unique()
    )
    scot_sunday_weeks = set(
        _normalize_week_to_sunday(scot["order_week"])
        .dropna().unique()
    )

    debug = {
        "forecast_asin_dtype": str(forecast_df["asin"].dtype),
        "scot_asin_dtype": str(scot["asin"].dtype),
        "forecast_asin_sample": forecast_df["asin"].head(5).astype(str).tolist(),
        "scot_asin_sample": scot["asin"].head(5).astype(str).tolist(),
        "raw_asin_intersection": len(forecast_raw_asins & scot_raw_asins),
        "canonical_asin_intersection": len(forecast_canonical & scot_canonical),
        "raw_week_intersection": len(forecast_weeks & scot_weeks),
        "sunday_week_intersection": len(forecast_sunday_weeks & scot_sunday_weeks),
        "forecast_week_values": sorted(str(pd.Timestamp(x).date()) for x in forecast_weeks)[:10],
        "scot_week_values": sorted(str(pd.Timestamp(x).date()) for x in scot_weeks)[:10],
    }

    print("\\n" + "=" * 88)
    print("JOIN KEY DEBUG — BEFORE SCOT MERGE")
    print("=" * 88)
    for key, value in debug.items():
        print(f"{key}: {value}")

    return debug


def run_high_sparse_scot_alignment_wape(
    result,
    scot_df,
    data_raw1=None,
    asin_stats=None,
    remove_oos_dp=True,
    source="lp",
):
    """
    Align real SCOT forecasts to result["forecast_df"] and compute WAPE.
    """
    if "calculate_wape_using_lp_oos2" not in globals():
        raise RuntimeError("calculate_wape_using_lp_oos2 is not defined.")

    if "quick_error_check" not in globals():
        raise RuntimeError("quick_error_check is not defined.")

    forecast_df = result["forecast_df"].copy()
    forecast_df.columns = [c.strip() for c in forecast_df.columns]
    forecast_df["asin"] = forecast_df["asin"].astype(str)
    forecast_df["order_week"] = pd.to_datetime(forecast_df["order_week"])

    scot = scot_df.copy()
    scot.columns = [c.strip() for c in scot.columns]

    for c in ["asin", "order_week", "forecast_qty_p50", "forecast_qty_p70", "forecast_qty_p90"]:
        if c not in scot.columns:
            raise ValueError(f"Missing SCOT column: {c}")

    scot["asin"] = scot["asin"].astype(str)
    scot["order_week"] = pd.to_datetime(scot["order_week"])
    scot["forecast_qty_p50"] = pd.to_numeric(scot["forecast_qty_p50"], errors="coerce")
    scot["forecast_qty_p70"] = pd.to_numeric(scot["forecast_qty_p70"], errors="coerce")
    scot["forecast_qty_p90"] = pd.to_numeric(scot["forecast_qty_p90"], errors="coerce")

    if "fcst_start_week" in scot.columns:
        scot["fcst_start_week"] = pd.to_datetime(scot["fcst_start_week"])

    print("\n" + "=" * 80)
    print("NB FORECAST WINDOW")
    print("=" * 80)
    print("NB rows:", len(forecast_df))
    print("NB ASINs:", forecast_df["asin"].nunique())
    print("NB weeks:", forecast_df["order_week"].min(), "to", forecast_df["order_week"].max())
    print("NB week count:", forecast_df["order_week"].nunique())

    print("\n" + "=" * 80)
    print("REAL SCOT FORECAST FILE")
    print("=" * 80)
    print("SCOT rows:", len(scot))
    print("SCOT ASINs:", scot["asin"].nunique())
    print("SCOT weeks:", scot["order_week"].min(), "to", scot["order_week"].max())
    print("SCOT week count:", scot["order_week"].nunique())

    if "fcst_start_week" in scot.columns:
        print("\nSCOT fcst_start_week counts:")
        print(scot["fcst_start_week"].value_counts().sort_index())

    alignment_debug = _print_join_key_debug(forecast_df, scot)

    # Auto-repair only when diagnostics prove the raw key format is the issue.
    if (
        alignment_debug["raw_asin_intersection"] == 0
        and alignment_debug["canonical_asin_intersection"] > 0
    ):
        print(
            "ASIN raw formats differ but canonical ASINs overlap. "
            "Using canonical ASIN join keys."
        )
        forecast_df["asin"] = _canonical_asin_join_key(forecast_df["asin"])
        scot["asin"] = _canonical_asin_join_key(scot["asin"])

    # Auto-repair week anchors only when raw weeks have no overlap and mapping
    # both sides to Sunday produces overlap. This handles Saturday snapshot
    # labels versus Sunday SCOT order_week labels without blindly adding a day.
    if (
        alignment_debug["raw_week_intersection"] == 0
        and alignment_debug["sunday_week_intersection"] > 0
    ):
        print(
            "Raw order_week anchors differ but Sunday-normalized weeks overlap. "
            "Normalizing both join keys to Sunday."
        )
        forecast_df["order_week"] = _normalize_week_to_sunday(
            forecast_df["order_week"]
        )
        scot["order_week"] = _normalize_week_to_sunday(scot["order_week"])

    scot_keep = (
        scot[["asin", "order_week", "forecast_qty_p50", "forecast_qty_p70", "forecast_qty_p90"]]
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            forecast_qty_p50=("forecast_qty_p50", "mean"),
            forecast_qty_p70=("forecast_qty_p70", "mean"),
            forecast_qty_p90=("forecast_qty_p90", "mean"),
        )
    )

    forecast_df_scot_real = forecast_df.merge(
        scot_keep,
        on=["asin", "order_week"],
        how="inner",
    )

    if forecast_df_scot_real.empty:
        raise RuntimeError(
            "No ASIN-week rows matched after raw and diagnostic-safe key "
            "normalization. Review the JOIN KEY DEBUG block above."
        )

    row_match_rate = len(forecast_df_scot_real) / max(len(forecast_df), 1)
    asin_match_rate = (
        forecast_df_scot_real["asin"].nunique()
        / max(forecast_df["asin"].nunique(), 1)
    )

    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    print("NB forecast rows:", len(forecast_df))
    print("After SCOT merge rows:", len(forecast_df_scot_real))
    print("Matched ASINs:", forecast_df_scot_real["asin"].nunique())
    print("Matched weeks:", forecast_df_scot_real["order_week"].min(), "to",
          forecast_df_scot_real["order_week"].max())
    print("Matched week count:", forecast_df_scot_real["order_week"].nunique())
    print("Row match rate:", row_match_rate)
    print("ASIN match rate:", asin_match_rate)

    print("\n" + "=" * 80)
    print("ASIN SELECTION CHECK")
    print("=" * 80)
    print("Selected NB ASINs:", forecast_df["asin"].nunique())
    print("Matched ASINs with SCOT:", forecast_df_scot_real["asin"].nunique())
    print(
        "Missing ASINs after SCOT merge:",
        forecast_df["asin"].nunique() - forecast_df_scot_real["asin"].nunique(),
    )

    forecast_df_scot_real["p50_scot"] = forecast_df_scot_real["forecast_qty_p50"]
    forecast_df_scot_real["p70_scot"] = np.maximum(
        forecast_df_scot_real["forecast_qty_p70"],
        forecast_df_scot_real["forecast_qty_p50"],
    )
    forecast_df_scot_real["p90_scot"] = np.maximum(
        forecast_df_scot_real["forecast_qty_p90"],
        forecast_df_scot_real["p70_scot"],
    )

    mean_check = pd.DataFrame([{
        "n_rows": len(forecast_df_scot_real),
        "n_asins": forecast_df_scot_real["asin"].nunique(),
        "true_mean": forecast_df_scot_real["fbi_demand"].mean(),
        "total_amt": (
            forecast_df_scot_real["true_amt"].sum()
            if "true_amt" in forecast_df_scot_real.columns
            else np.nan
        ),
        "total_size": (
            forecast_df_scot_real["true_size"].sum()
            if "true_size" in forecast_df_scot_real.columns
            else np.nan
        ),
        "amxl_p50_mean": forecast_df_scot_real["p50_amxl"].mean(),
        "amxl_p70_mean": forecast_df_scot_real["p70_amxl"].mean(),
        "amxl_p90_mean": forecast_df_scot_real["p90_amxl"].mean(),
        "real_scot_p50_mean": forecast_df_scot_real["p50_scot"].mean(),
        "real_scot_p70_mean": forecast_df_scot_real["p70_scot"].mean(),
        "real_scot_p90_mean": forecast_df_scot_real["p90_scot"].mean(),
        "true_zero_rate": (forecast_df_scot_real["fbi_demand"] == 0).mean(),
        "true_active_ratio": (forecast_df_scot_real["fbi_demand"] > 0).mean(),
    }])

    print("\n" + "=" * 80)
    print("FORECAST MEAN CHECK")
    print("=" * 80)
    print(mean_check.T)

    wape_df = calculate_wape_using_lp_oos2(
        forecast_df_scot_real,
        [0.5, 0.7, 0.9],
        remove_oos_dp=remove_oos_dp,
        source=source,
    )

    if asin_stats is None and "asin_stats" in result:
        asin_stats = result["asin_stats"]

    forecast_df_scot_real_with_group = attach_zero_group_to_joined_df(
        forecast_df_scot_real,
        asin_stats,
    )

    sparse_group_wape = summarize_wape_by_sparse_group(
        wape_df,
        forecast_df_scot_real_with_group,
    )

    cols_p50 = [
        "p50_amxl_penalty", "p50_scot_penalty",
        "p50_amxl_overbias", "p50_scot_overbias",
        "p50_amxl_underbias", "p50_scot_underbias",
        "fbi_demand",
    ]

    cols_p70 = [
        "p70_amxl_penalty", "p70_scot_penalty",
        "p70_amxl_overbias", "p70_scot_overbias",
        "p70_amxl_underbias", "p70_scot_underbias",
        "fbi_demand",
    ]

    cols_p90 = [
        "p90_amxl_penalty", "p90_scot_penalty",
        "p90_amxl_overbias", "p90_scot_overbias",
        "p90_amxl_underbias", "p90_scot_underbias",
        "fbi_demand",
    ]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)
    p90_wape, p90_penalty_diff = quick_error_check(wape_df, cols_p90)

    # Per-horizon (H1/H2/H3) breakdown, same boss-approved WAPE formula,
    # just grouped by fcst_week_index instead of summed over everything.
    h_wape_p50 = weekly_error_check(wape_df, cols_p50, cols_type="p50")
    h_wape_p70 = weekly_error_check(wape_df, cols_p70, cols_type="p70")
    h_wape_p90 = weekly_error_check(wape_df, cols_p90, cols_type="p90")

    print("\n" + "=" * 80)
    print("FINAL WAPE WITH REAL SCOT")
    print("=" * 80)
    print("\nP50 WAPE:")
    print(p50_wape)
    print("P50 penalty diff AMXL - SCOT:", p50_penalty_diff)
    print("\nP70 WAPE:")
    print(p70_wape)
    print("P70 penalty diff AMXL - SCOT:", p70_penalty_diff)
    print("\nP90 WAPE:")
    print(p90_wape)
    print("P90 penalty diff AMXL - SCOT:", p90_penalty_diff)

    print("\n" + "=" * 80)
    print("WAPE BY HORIZON (H1/H2/H3)")
    print("=" * 80)
    print("\nP50 by horizon:")
    print(h_wape_p50[["fcst_week_index", "p50_amxl_wape", "p50_scot_wape", "p50_diff_bps"]].to_string(index=False))
    print("\nP70 by horizon:")
    print(h_wape_p70[["fcst_week_index", "p70_amxl_wape", "p70_scot_wape", "penalty_diff_bps"]].to_string(index=False))
    print("\nP90 by horizon:")
    print(h_wape_p90[["fcst_week_index", "p90_amxl_wape", "p90_scot_wape", "penalty_diff_bps"]].to_string(index=False))

    return {
        "alignment_debug": alignment_debug,
        "forecast_df_scot_real": forecast_df_scot_real,
        "forecast_df_scot_real_with_group": forecast_df_scot_real_with_group,
        "wape_df": wape_df,
        "sparse_group_wape": sparse_group_wape,
        "mean_check": mean_check,
        "p50_wape": p50_wape,
        "p70_wape": p70_wape,
        "p90_wape": p90_wape,
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
        "p90_penalty_diff": p90_penalty_diff,
        "h_wape_p50": h_wape_p50,
        "h_wape_p70": h_wape_p70,
        "h_wape_p90": h_wape_p90,
    }


# =====================================================
# 11. Train on sample-SCOT intersection
# =====================================================

def run_nb_high_sparse_from_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.05,
    lambda_stock_mean_weight=0.30,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    run_wape=True,
    remove_oos_dp=True,
):
    """
    Sample 5000 from data_raw1, keep SCOT intersection, train high_sparse, and compute WAPE.
    """
    print("=" * 80)
    print("LEGACY NB HIGH-SPARSE | SAMPLE 5000 THEN KEEP SCOT INTERSECTION")
    print("=" * 80)

    data_small_raw, sample_asin_df, intersect_asin_df = (
        prepare_data_from_sample_scot_intersection(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
        )
    )

    data_small, asin_stats = add_zero_rate_group(data_small_raw, zero_thresholds)
    data_high = data_small[data_small["zero_group"] == "high_sparse"].copy()

    print("\n" + "=" * 80)
    print("HIGH-SPARSE AFTER SCOT INTERSECTION")
    print("=" * 80)
    print("High-sparse ASINs:", data_high["asin"].nunique())
    print("High-sparse rows:", len(data_high))

    if remove_extreme:
        data_high, removed_extreme, extreme_cap = filter_extreme_asins(
            data_high,
            q=extreme_q,
        )
    else:
        removed_extreme = pd.DataFrame()
        extreme_cap = np.nan

    data, context_dim, context_cols = load_real_data(data_high, dph_cap_q=dph_cap_q)

    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs used for training: {len(data)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val", horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN(
        input_dim=34,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | lambda_under={lambda_under} | q_active_weight={q_active_weight} | q_tail_weight={q_tail_weight} | patience={patience}")

    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        lambda_under=lambda_under,
        q_active_weight=q_active_weight,
        q_tail_weight=q_tail_weight,
        lambda_stock=lambda_stock,
        lambda_stock_mean_weight=lambda_stock_mean_weight,
    )

    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball: P50(q50)={metrics['pinball50']:.4f} | P70(q70)={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "high_sparse_sample_scot_intersection"

    diag_df = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:")
    print(diag_p50.T)
    print("\nUnderbias P70:")
    print(diag_p70.T)

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "data_small": data_small,
        "data_high": data_high,
        "asin_stats": asin_stats,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
    }

    # v12 diagnostics: h=1..H magnitude gap / underforecast check.
    result["sparse_horizon_outputs"] = summarize_sparse_horizon_decay(
        forecast_df=forecast_df,
        asin_stats=asin_stats,
        data_raw1=data_raw1,
        pred_cols=("p50_amxl", "p70_amxl"),
        true_col="fbi_demand",
        horizon_col="fcst_week_index",
        zero_thresholds=zero_thresholds,
        print_table=True,
    )
    result["horizon_mag_outputs"] = summarize_h1_h20_magnitude_diagnostics(
        forecast_df=forecast_df,
        asin_stats=asin_stats,
        data_raw1=data_raw1,
        pred_cols=("p50_amxl", "p70_amxl"),
        true_col="fbi_demand",
        horizon_col="fcst_week_index",
        zero_thresholds=zero_thresholds,
        print_table=True,
    )

    if run_wape:
        result["real_scot_outputs"] = run_high_sparse_scot_alignment_wape(
            result=result,
            scot_df=scot_df,
            data_raw1=data_raw1,
            asin_stats=asin_stats,
            remove_oos_dp=remove_oos_dp,
            source="lp",
        )

    return result



# =====================================================
# 12. Train on all sample-SCOT intersection ASINs
# =====================================================

def run_nb_all_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.05,
    lambda_stock_mean_weight=0.30,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    run_wape=True,
    remove_oos_dp=True,
):
    """
    Main experiment:
      1. sample 5000 ASINs from data_raw1
      2. keep ASINs also present in scot_df
      3. assign sparse labels for diagnostics only
      4. train one model on all intersection ASINs
      5. align with real SCOT and compute overall + sparse-group WAPE
    """
    print("=" * 80)
    print("NB ALL-ASIN | SAMPLE 5000 THEN KEEP SCOT INTERSECTION")
    print("=" * 80)

    data_intersection_raw, sample_asin_df, intersect_asin_df = (
        prepare_data_from_sample_scot_intersection(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
        )
    )

    # Sparse labels are for diagnostics only. No filtering by group.
    data_labeled, asin_stats = add_zero_rate_group(
        data_intersection_raw,
        zero_thresholds,
    )

    print("\n" + "=" * 80)
    print("TRAINING SET AFTER SCOT INTERSECTION")
    print("=" * 80)
    print("Training ASINs:", data_labeled["asin"].nunique())
    print("Training rows:", len(data_labeled))

    print("\nSparse-group labels for diagnostics only:")
    print(
        data_labeled
        .groupby("zero_group")["asin"]
        .nunique()
        .reset_index(name="n_asins")
    )

    data_train = data_labeled.copy()

    if remove_extreme:
        data_train, removed_extreme, extreme_cap = filter_extreme_asins(
            data_train,
            q=extreme_q,
        )
    else:
        removed_extreme = pd.DataFrame()
        extreme_cap = np.nan

    data, context_dim, context_cols = load_real_data(data_train)

    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs used for training: {len(data)}")
    print(f"Overall zero rate: {(all_demand == 0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val", horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN(
        input_dim=34,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | lambda_under={lambda_under} | q_active_weight={q_active_weight} | q_tail_weight={q_tail_weight} | patience={patience}")

    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        lambda_under=lambda_under,
        q_active_weight=q_active_weight,
        q_tail_weight=q_tail_weight,
        lambda_stock=lambda_stock,
        lambda_stock_mean_weight=lambda_stock_mean_weight,
    )

    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball: P50(q50)={metrics['pinball50']:.4f} | P70(q70)={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "all_sample_scot_intersection"

    diag_df = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:")
    print(diag_p50.T)

    print("\nUnderbias P70:")
    print(diag_p70.T)

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "data_intersection_raw": data_intersection_raw,
        "data_labeled": data_labeled,
        "data_train": data_train,
        "asin_stats": asin_stats,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
    }

    # v12 diagnostics: h=1..H magnitude gap / underforecast check.
    result["sparse_horizon_outputs"] = summarize_sparse_horizon_decay(
        forecast_df=forecast_df,
        asin_stats=asin_stats,
        data_raw1=data_raw1,
        pred_cols=("p50_amxl", "p70_amxl"),
        true_col="fbi_demand",
        horizon_col="fcst_week_index",
        zero_thresholds=zero_thresholds,
        print_table=True,
    )
    result["horizon_mag_outputs"] = summarize_h1_h20_magnitude_diagnostics(
        forecast_df=forecast_df,
        asin_stats=asin_stats,
        data_raw1=data_raw1,
        pred_cols=("p50_amxl", "p70_amxl"),
        true_col="fbi_demand",
        horizon_col="fcst_week_index",
        zero_thresholds=zero_thresholds,
        print_table=True,
    )

    if run_wape:
        result["real_scot_outputs"] = run_high_sparse_scot_alignment_wape(
            result=result,
            scot_df=scot_df,
            data_raw1=data_raw1,
            asin_stats=asin_stats,
            remove_oos_dp=remove_oos_dp,
            source="lp",
        )

    return result



# =====================================================

# ============================================================
# External exposure-3 injection into demand future_context
# ============================================================

_ORIGINAL_LOAD_REAL_DATA_BEFORE_EXTERNAL_EXP3 = load_real_data


def load_exposure_hat_for_demand_csv(csv_path):
    """
    Load a saved exposure_hat_for_demand CSV generated by exposure v25 SAVEHAT.
    This returns a dataframe that can be passed as exposure_result_or_hat.
    """
    import pandas as pd
    import numpy as np
    hat = pd.read_csv(csv_path)
    if "asin" not in hat.columns or "order_week" not in hat.columns:
        raise ValueError(f"CSV must contain asin and order_week. Available columns: {hat.columns.tolist()}")
    hat["asin"] = hat["asin"].astype(str)
    hat["order_week"] = pd.to_datetime(hat["order_week"])
    for c in ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph", "pred_in_stock_dph"]:
        if c in hat.columns:
            hat[c] = pd.to_numeric(hat[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    print(f"\nLoaded exposure hat CSV: {csv_path}")
    print(f"Rows: {len(hat):,} | ASINs: {hat['asin'].nunique():,} | Weeks: {hat['order_week'].nunique():,}")
    return hat

def _extract_external_exposure3_hat(result_or_hat):
    """
    Extract a clean dataframe containing external predicted exposure hats.

    This version is duplicate-column safe.

    Priority:
      1. calibrated level columns:
           pred_total_dph_calib
           pred_buy_box_dph_calib
           pred_in_stock_dph_calib / pred_instock_dph_calib
      2. normal level columns:
           pred_total_dph
           pred_buy_box_dph
           pred_instock_dph / pred_in_stock_dph
      3. attention level columns:
           attn_total_dph
           attn_buy_box_dph
           attn_instock_dph / attn_in_stock_dph
      4. log columns:
           external_total_dph_hat_log
           external_buy_box_dph_hat_log
           external_instock_dph_hat_log
    """
    source = None

    # Allow passing a saved CSV path directly.
    if isinstance(result_or_hat, (str, bytes)):
        hat = load_exposure_hat_for_demand_csv(result_or_hat)
        source = f"csv_path:{result_or_hat}"

    elif isinstance(result_or_hat, dict):
        if "exposure_hat_for_demand_calib" in result_or_hat:
            hat = result_or_hat["exposure_hat_for_demand_calib"].copy()
            source = "dict['exposure_hat_for_demand_calib']"

        elif "exposure_hat_for_demand" in result_or_hat:
            hat = result_or_hat["exposure_hat_for_demand"].copy()
            source = "dict['exposure_hat_for_demand']"

        elif "result_focus" in result_or_hat and isinstance(result_or_hat["result_focus"], dict):
            rf = result_or_hat["result_focus"]

            if "exposure_hat_for_demand_calib" in rf:
                hat = rf["exposure_hat_for_demand_calib"].copy()
                source = "dict['result_focus']['exposure_hat_for_demand_calib']"
            elif "exposure_hat_for_demand" in rf:
                hat = rf["exposure_hat_for_demand"].copy()
                source = "dict['result_focus']['exposure_hat_for_demand']"
            elif "attn_df" in rf:
                hat = rf["attn_df"].copy()
                source = "dict['result_focus']['attn_df']"
            else:
                raise ValueError("result_focus has no exposure_hat_for_demand / exposure_hat_for_demand_calib / attn_df.")

        elif "attn_df" in result_or_hat:
            hat = result_or_hat["attn_df"].copy()
            source = "dict['attn_df']"

        else:
            raise ValueError(
                "Cannot find exposure hat dataframe in dict. "
                "Expected exposure_hat_for_demand_calib, exposure_hat_for_demand, result_focus, or attn_df."
            )
    else:
        hat = result_or_hat.copy()
        source = "direct dataframe input"

    hat = hat.copy()

    if "asin" not in hat.columns or "order_week" not in hat.columns:
        raise ValueError("External exposure hat must contain asin and order_week.")

    def _first_existing_col(df, cols):
        for c in cols:
            if c in df.columns:
                x = df[c]
                # If duplicate column names still exist for any reason, take the first one.
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[:, 0]
                return x, c
        return None, None

    total_s, total_src = _first_existing_col(
        hat,
        [
            "pred_total_dph_calib",
            "pred_total_dph",
            "attn_total_dph",
            "external_total_dph_hat_log",
        ],
    )

    buy_s, buy_src = _first_existing_col(
        hat,
        [
            "pred_buy_box_dph_calib",
            "pred_buy_box_dph",
            "attn_buy_box_dph",
            "external_buy_box_dph_hat_log",
        ],
    )

    instock_s, instock_src = _first_existing_col(
        hat,
        [
            "pred_in_stock_dph_calib",
            "pred_instock_dph_calib",
            "pred_instock_dph",
            "pred_in_stock_dph",
            "attn_instock_dph",
            "attn_in_stock_dph",
            "external_instock_dph_hat_log",
        ],
    )

    missing = []
    if total_s is None:
        missing.append("pred_total_dph")
    if buy_s is None:
        missing.append("pred_buy_box_dph")
    if instock_s is None:
        missing.append("pred_instock_dph")

    if missing:
        raise ValueError(
            "External exposure hat is missing required prediction columns: "
            f"{missing}. Available columns: {hat.columns.tolist()}"
        )

    clean = pd.DataFrame({
        "asin": hat["asin"].astype(str),
        "order_week": pd.to_datetime(hat["order_week"]),
    })

    # If source is log column, convert back to level.
    if total_src == "external_total_dph_hat_log":
        clean["pred_total_dph"] = np.expm1(pd.to_numeric(total_s, errors="coerce").fillna(0.0))
    else:
        clean["pred_total_dph"] = pd.to_numeric(total_s, errors="coerce").fillna(0.0)

    if buy_src == "external_buy_box_dph_hat_log":
        clean["pred_buy_box_dph"] = np.expm1(pd.to_numeric(buy_s, errors="coerce").fillna(0.0))
    else:
        clean["pred_buy_box_dph"] = pd.to_numeric(buy_s, errors="coerce").fillna(0.0)

    if instock_src == "external_instock_dph_hat_log":
        clean["pred_instock_dph"] = np.expm1(pd.to_numeric(instock_s, errors="coerce").fillna(0.0))
    else:
        clean["pred_instock_dph"] = pd.to_numeric(instock_s, errors="coerce").fillna(0.0)

    for c in ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]:
        clean[c] = clean[c].fillna(0.0).clip(lower=0.0)

    # Safety: one ASIN-week row.
    clean = (
        clean.groupby(["asin", "order_week"], as_index=False)
        .agg(
            pred_total_dph=("pred_total_dph", "mean"),
            pred_buy_box_dph=("pred_buy_box_dph", "mean"),
            pred_instock_dph=("pred_instock_dph", "mean"),
        )
    )

    print("\nExternal exposure hat source:", source)
    print("Selected total column:", total_src)
    print("Selected buy_box column:", buy_src)
    print("Selected instock column:", instock_src)

    return clean, source


def attach_external_exposure3_to_raw_data(
    data_raw1,
    exposure3_hat=None,
    exposure_mode="all3",
):
    """
    Attach external predicted exposure funnel to data_raw1.

    exposure_mode:
      "instock_only":
          use only predicted in_stock DPH hat; total/buy_box hats are set to 0

      "buybox_only":
          use only predicted buy_box DPH hat; total/in_stock hats are set to 0

      "all3":
          use predicted total + buy_box + in_stock hats

    Output columns:
      attn_pred_total_dph
      attn_pred_buy_box_dph
      attn_pred_instock_dph

    These columns are then picked up by the overridden load_real_data and DemandDataset.
    """
    valid_modes = {"instock_only", "buybox_only", "all3"}
    if exposure_mode not in valid_modes:
        raise ValueError(f"exposure_mode must be one of {sorted(valid_modes)}, got {exposure_mode}")

    df = data_raw1.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    if exposure3_hat is None:
        raise ValueError(
            f"exposure3_hat cannot be None when exposure_mode='{exposure_mode}'. "
            "This clean version only supports predicted external exposure hats."
        )

    hat, source = _extract_external_exposure3_hat(exposure3_hat)

    # Select which external hats are allowed to enter demand model.
    use_total = exposure_mode == "all3"
    use_buy = exposure_mode in {"all3", "buybox_only"}
    use_instock = exposure_mode in {"all3", "instock_only"}
    uses_true_future_exposure = False

    if not use_total:
        hat["pred_total_dph"] = 0.0
    if not use_buy:
        hat["pred_buy_box_dph"] = 0.0
    if not use_instock:
        hat["pred_instock_dph"] = 0.0

    out = df.merge(
        hat.rename(
            columns={
                "pred_total_dph": "attn_pred_total_dph",
                "pred_buy_box_dph": "attn_pred_buy_box_dph",
                "pred_instock_dph": "attn_pred_instock_dph",
            }
        ),
        on=["asin", "order_week"],
        how="left",
    )

    for c in [
        "attn_pred_total_dph",
        "attn_pred_buy_box_dph",
        "attn_pred_instock_dph",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    out["attn_pred_total_log"] = np.log1p(out["attn_pred_total_dph"])
    out["attn_pred_buy_box_log"] = np.log1p(out["attn_pred_buy_box_dph"])
    out["attn_pred_instock_log"] = np.log1p(out["attn_pred_instock_dph"])

    print("\n" + "=" * 100)
    print("EXTERNAL EXPOSURE HATS ATTACHED TO DEMAND DATA")
    print("=" * 100)
    print("Source:", source)
    print("exposure_mode:", exposure_mode)
    print("Using total hat:", use_total)
    print("Using buy_box hat:", use_buy)
    print("Using instock hat:", use_instock)

    print("\nDemand model receives:")
    if use_total:
        print("  log1p(attn_pred_total_dph)")
    if use_buy:
        print("  log1p(attn_pred_buy_box_dph)")
    if use_instock:
        print("  log1p(attn_pred_instock_dph)")

    if uses_true_future_exposure:
        print("WARNING: This mode uses TRUE future in_stock_dph. Use only as oracle upper-bound test.")
    else:
        print("No true future exposure is used as input.")

    print("\nHat summaries after mode selection:")
    print(
        out[
            [
                "attn_pred_total_dph",
                "attn_pred_buy_box_dph",
                "attn_pred_instock_dph",
            ]
        ].describe().round(4).to_string()
    )

    return out



def run_external_exposure3_in_old_decoder_style(
    data_raw1,
    scot_df,
    exposure3_hat=None,
    exposure_mode="all3",
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.0,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    lambda_stock=0.0,
    lambda_stock_mean_weight=0.0,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    run_wape=True,
    remove_oos_dp=True,
):
    """
    Demand model with external predicted exposure-3.

    Use this when you already have the three DPH hats from another pipeline:
      exposure_hat_for_demand_calib
      exposure_hat_for_demand_e2e_attn
      exposure_hat_for_demand
      or any dataframe with pred_total_dph / pred_buy_box_dph / pred_instock_dph.

    This function injects the three hats into the demand model's future context.
    """
    print("\n" + "=" * 100)
    print("DEMAND MODEL WITH EXTERNAL EXPOSURE HATS")
    print("=" * 100)
    print("exposure_mode:", exposure_mode)

    data_with_external_exp3 = attach_external_exposure3_to_raw_data(
        data_raw1=data_raw1,
        exposure3_hat=exposure3_hat,
        exposure_mode=exposure_mode,
    )

    return run_nb_all_sample_scot_intersection(
        data_raw1=data_with_external_exp3,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        zero_thresholds=zero_thresholds,
        prior_scale=prior_scale,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        lambda_under=lambda_under,
        q_active_weight=q_active_weight,
        q_tail_weight=q_tail_weight,
        lambda_stock=lambda_stock,
        lambda_stock_mean_weight=lambda_stock_mean_weight,
        dph_cap_q=dph_cap_q,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
        run_wape=run_wape,
        remove_oos_dp=remove_oos_dp,
    )



def load_real_data(data_raw, dph_cap_q=0.995):
    """
    Override original load_real_data to inject external exposure-3 hats into future_context.

    Added future context columns:
      external_total_dph_hat_log
      external_buy_box_dph_hat_log
      external_instock_dph_hat_log

    These are predicted future covariates, not true future DPH.
    """
    data, context_dim, context_cols = _ORIGINAL_LOAD_REAL_DATA_BEFORE_EXTERNAL_EXP3(
        data_raw=data_raw,
        dph_cap_q=dph_cap_q,
    )

    required = [
        "asin",
        "order_week",
        "attn_pred_total_log",
        "attn_pred_buy_box_log",
        "attn_pred_instock_log",
    ]

    if not all(c in data_raw.columns for c in required):
        print("\nExternal exposure-3 columns not found. Using original future_context.")
        return data, context_dim, context_cols

    ext = data_raw[required].copy()
    ext["asin"] = ext["asin"].astype(str)
    ext["order_week"] = pd.to_datetime(ext["order_week"])

    for c in ["attn_pred_total_log", "attn_pred_buy_box_log", "attn_pred_instock_log"]:
        ext[c] = pd.to_numeric(ext[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    ext = (
        ext.sort_values(["asin", "order_week"])
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            attn_pred_total_log=("attn_pred_total_log", "mean"),
            attn_pred_buy_box_log=("attn_pred_buy_box_log", "mean"),
            attn_pred_instock_log=("attn_pred_instock_log", "mean"),
        )
    )

    new_cols = [
        "external_total_dph_hat_log",
        "external_buy_box_dph_hat_log",
        "external_instock_dph_hat_log",
    ]

    added_any = False

    for asin, d in data.items():
        sub = ext[ext["asin"] == str(asin)].sort_values("order_week")

        if len(sub) != len(d["week"]):
            # Align by week to be safe.
            week_df = pd.DataFrame({"order_week": pd.to_datetime(d["week"])})
            sub = week_df.merge(
                sub.drop(columns=["asin"]),
                on="order_week",
                how="left",
            )

        arr = sub[[
            "attn_pred_total_log",
            "attn_pred_buy_box_log",
            "attn_pred_instock_log",
        ]].fillna(0.0).values.astype(np.float32)

        old_fc = d["future_context"]
        d["future_context"] = np.concatenate([old_fc, arr], axis=1)
        added_any = True

    if added_any:
        context_cols = context_cols + new_cols
        context_dim = len(context_cols)

        print("\n" + "=" * 100)
        print("EXTERNAL EXPOSURE-3 HATS ADDED TO FUTURE_CONTEXT")
        print("=" * 100)
        print("Added context cols:", new_cols)
        print("New context dim:", context_dim)

    return data, context_dim, context_cols


# ============================================================
# Clean usage helpers (NO auto-run)
# ============================================================

def run_demand_with_predicted_exposure_all3(
    data_raw1,
    scot_df,
    exposure_result_or_hat,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    remove_oos_dp=True,
):
    """
    Recommended production-style run.

    exposure_result_or_hat can be either:
      1. exposure_result dict with key 'exposure_hat_for_demand', or
      2. exposure_hat_for_demand dataframe from the exposure model.

    Uses all three predicted exposure hats:
      pred_total_dph, pred_buy_box_dph, pred_instock_dph.
    """
    return run_external_exposure3_in_old_decoder_style(
        data_raw1=data_raw1,
        scot_df=scot_df,
        exposure3_hat=exposure_result_or_hat,
        exposure_mode="all3",
        n_asins=n_asins,
        seed=seed,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=0.0,
        beta_tail=0.5,
        patience=5,
        lambda_z_reg=1.0,
        lambda_stock=0.0,
        lambda_stock_mean_weight=0.0,
        remove_extreme=True,
        extreme_q=0.99,
        run_wape=True,
        remove_oos_dp=remove_oos_dp,
    )


def run_demand_with_predicted_exposure_instock_only(
    data_raw1,
    scot_df,
    exposure_result_or_hat,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    remove_oos_dp=True,
):
    """
    Comparison run: use only predicted in_stock_dph hat.
    """
    return run_external_exposure3_in_old_decoder_style(
        data_raw1=data_raw1,
        scot_df=scot_df,
        exposure3_hat=exposure_result_or_hat,
        exposure_mode="instock_only",
        n_asins=n_asins,
        seed=seed,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=0.0,
        beta_tail=0.5,
        patience=5,
        lambda_z_reg=1.0,
        lambda_stock=0.0,
        lambda_stock_mean_weight=0.0,
        remove_extreme=True,
        extreme_q=0.99,
        run_wape=True,
        remove_oos_dp=remove_oos_dp,
    )



def run_demand_with_predicted_exposure_buybox_only(
    data_raw1,
    scot_df,
    exposure_result_or_hat,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    remove_oos_dp=True,
):
    """
    Comparison run: use only predicted buy_box_dph hat.
    """
    return run_external_exposure3_in_old_decoder_style(
        data_raw1=data_raw1,
        scot_df=scot_df,
        exposure3_hat=exposure_result_or_hat,
        exposure_mode="buybox_only",
        n_asins=n_asins,
        seed=seed,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=0.0,
        beta_tail=0.5,
        patience=5,
        lambda_z_reg=1.0,
        lambda_stock=0.0,
        lambda_stock_mean_weight=0.0,
        remove_extreme=True,
        extreme_q=0.99,
        run_wape=True,
        remove_oos_dp=remove_oos_dp,
    )

"""
USAGE IN JUPYTER
----------------
%run -i demand_external_exposure3_clean_3modes.py

# After running your exposure model:
# exposure_result = run_exposure_v2_final_scot_5000(...)
# exposure_hat_for_demand = exposure_result["exposure_hat_for_demand"]

# Mode 1: predicted in-stock only
demand_result_instock = run_demand_with_predicted_exposure_instock_only(
    data_raw1=data_raw1,
    scot_df=scot_df,
    exposure_result_or_hat=exposure_hat_for_demand,
    n_asins=5000,
    epochs=60,
    history=52,
    horizon=3,
)

# Mode 2: predicted buy-box only
demand_result_buybox = run_demand_with_predicted_exposure_buybox_only(
    data_raw1=data_raw1,
    scot_df=scot_df,
    exposure_result_or_hat=exposure_hat_for_demand,
    n_asins=5000,
    epochs=60,
    history=52,
    horizon=3,
)

# Mode 3: predicted total + buy-box + in-stock
demand_result_all3 = run_demand_with_predicted_exposure_all3(
    data_raw1=data_raw1,
    scot_df=scot_df,
    exposure_result_or_hat=exposure_hat_for_demand,
    n_asins=5000,
    epochs=60,
    history=52,
    horizon=3,
)
"""


# ============================================================
# PACKAGE-AWARE ASIN RELATION GRAPH CONTEXT PATCH
# Added by ChatGPT: package-comparable peer graph features.
# Design:
#   - graph neighbors = same category_code + relaxed package-size/weight comparable
#   - graph features are computed at forecast origin from historical values only
#   - future target weeks are never used to construct graph context
#   - features are injected into future_context so encoder/decoder/z can use them
# ============================================================

GRAPH_CONTEXT_COLS = [
    "graph_peer_total_mean13_log",
    "graph_peer_buybox_mean13_log",
    "graph_peer_instock_mean13_log",
    "graph_peer_demand_mean13_log",
    "graph_peer_active_rate13",
    "graph_peer_zero_rate13",
    "graph_peer_count_log",
    "graph_peer_rank_prior",
    "graph_same_hbt_peer_rate",
    "graph_top10_peer_rate",
]


def _graph_mode_str(x, default="MISSING"):
    try:
        s = pd.Series(x).astype(str).replace({"nan": default, "None": default, "": default})
        if len(s) == 0:
            return default
        return str(s.mode().iloc[0]) if len(s.mode()) else default
    except Exception:
        return default


def _graph_safe_num(x, fill=0.0):
    try:
        return pd.to_numeric(x, errors="coerce").fillna(fill)
    except Exception:
        return pd.Series(fill, index=getattr(x, 'index', None))


def _graph_build_meta_from_raw(data_raw):
    """Static product metadata used for graph neighbor construction."""
    df = data_raw.copy()
    asin_col = "asin" if "asin" in df.columns else ("ASIN" if "ASIN" in df.columns else None)
    if asin_col is None:
        return {}
    df[asin_col] = df[asin_col].astype(str)
    if "order_week" in df.columns:
        df["order_week"] = pd.to_datetime(df["order_week"], errors="coerce")
        df = df.sort_values([asin_col, "order_week"])

    for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight", "our_price", "ind_top10_brand"]:
        if c not in df.columns:
            df[c] = np.nan if c.startswith("pkg_") else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "category_code" not in df.columns:
        df["category_code"] = "UNKNOWN"
    if "hbt" not in df.columns:
        df["hbt"] = "MISSING"

    meta = {}
    for asin, g in df.groupby(asin_col):
        last = g.iloc[-1]
        dims = {}
        for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight"]:
            vals = pd.to_numeric(g[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v = float(vals.iloc[-1]) if len(vals) else np.nan
            dims[c] = v
            dims[f"log_{c}"] = float(np.log1p(max(v, 0.0))) if np.isfinite(v) else np.nan
        vol = dims.get("pkg_height", np.nan) * dims.get("pkg_length", np.nan) * dims.get("pkg_width", np.nan)
        dims["pkg_volume"] = vol
        dims["log_pkg_volume"] = float(np.log1p(max(vol, 0.0))) if np.isfinite(vol) else np.nan
        price_vals = pd.to_numeric(g["our_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        price = float(price_vals.iloc[-1]) if len(price_vals) else 0.0
        meta[str(asin)] = {
            "category_code": _graph_mode_str(g["category_code"], "UNKNOWN"),
            "hbt": _graph_mode_str(g["hbt"], "MISSING").lower(),
            "ind_top10_brand": float(pd.to_numeric(g["ind_top10_brand"], errors="coerce").fillna(0.0).iloc[-1]),
            "log_our_price": float(np.log1p(max(price, 0.0))),
            **dims,
        }
    return meta


def _graph_pkg_relaxed_similar(mi, mj, max_mean_log_gap=0.40, max_volume_log_gap=0.75, max_weight_log_gap=0.55):
    """Relaxed package comparability: same physical scale, not exact duplicate."""
    dim_cols = ["log_pkg_height", "log_pkg_length", "log_pkg_width", "log_pkg_weight"]
    gaps = []
    for c in dim_cols:
        a, b = mi.get(c, np.nan), mj.get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            gaps.append(abs(float(a) - float(b)))
    if len(gaps) == 0:
        return False
    mean_gap = float(np.mean(gaps))
    vol_gap = abs(float(mi.get("log_pkg_volume", np.nan)) - float(mj.get("log_pkg_volume", np.nan))) \
        if np.isfinite(mi.get("log_pkg_volume", np.nan)) and np.isfinite(mj.get("log_pkg_volume", np.nan)) else mean_gap
    wt_gap = abs(float(mi.get("log_pkg_weight", np.nan)) - float(mj.get("log_pkg_weight", np.nan))) \
        if np.isfinite(mi.get("log_pkg_weight", np.nan)) and np.isfinite(mj.get("log_pkg_weight", np.nan)) else mean_gap
    return (mean_gap <= max_mean_log_gap) and (vol_gap <= max_volume_log_gap) and (wt_gap <= max_weight_log_gap)


def _graph_recent_mean(arr, end, window=13):
    x = np.asarray(arr[max(0, end-window):end], dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.clip(x, 0, None)))


def _graph_strength_for_asin(d, end):
    return (
        0.30 * np.log1p(_graph_recent_mean(d.get("total_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("buy_box_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13)) +
        0.20 * np.log1p(_graph_recent_mean(d.get("demand", []), end, 13))
    )


def _graph_percentile_rank(value, values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) <= 1:
        return 0.5
    return float((np.sum(vals < value) + 0.5 * np.sum(vals == value)) / len(vals))


def _graph_add_context_cols_to_data(data, context_cols, data_raw=None):
    """Append zero graph columns to every future_context and attach product metadata."""
    context_cols = list(context_cols)
    add_cols = [c for c in GRAPH_CONTEXT_COLS if c not in context_cols]
    if len(add_cols) > 0:
        for d in data.values():
            zeros = np.zeros((d["future_context"].shape[0], len(add_cols)), dtype=np.float32)
            d["future_context"] = np.concatenate([d["future_context"], zeros], axis=1)
        context_cols = context_cols + add_cols

    meta = _graph_build_meta_from_raw(data_raw) if data_raw is not None else {}
    for asin, d in data.items():
        d["context_cols"] = context_cols
        d["graph_context_idx"] = {c: context_cols.index(c) for c in GRAPH_CONTEXT_COLS if c in context_cols}
        d["graph_meta"] = meta.get(str(asin), {
            "category_code": "UNKNOWN", "hbt": "missing", "ind_top10_brand": 0.0,
            "log_our_price": 0.0, "log_pkg_height": np.nan, "log_pkg_length": np.nan,
            "log_pkg_width": np.nan, "log_pkg_weight": np.nan, "log_pkg_volume": np.nan,
        })
    return data, len(context_cols), context_cols


def _graph_build_neighbor_map(data, min_neighbors=3):
    asins = list(data.keys())
    by_cat = {}
    for a in asins:
        cat = data[a].get("graph_meta", {}).get("category_code", "UNKNOWN")
        by_cat.setdefault(cat, []).append(a)

    nbrs = {}
    for a in asins:
        mi = data[a].get("graph_meta", {})
        cand = []
        for b in by_cat.get(mi.get("category_code", "UNKNOWN"), []):
            if b == a:
                continue
            mj = data[b].get("graph_meta", {})
            if _graph_pkg_relaxed_similar(mi, mj):
                cand.append(b)
        if len(cand) < min_neighbors:
            cand = [b for b in by_cat.get(mi.get("category_code", "UNKNOWN"), []) if b != a]
        nbrs[a] = cand
    return nbrs


class _GraphContextMixin:
    def _init_graph_context(self, min_graph_neighbors=3):
        self.graph_neighbor_map = _graph_build_neighbor_map(self.data, min_neighbors=min_graph_neighbors)
        self._graph_context_cache = {}
        counts = [len(v) for v in self.graph_neighbor_map.values()]
        if len(counts):
            print("Package-aware graph context enabled | ASINs:", len(counts),
                  "| median neighbors:", int(np.median(counts)),
                  "| mean neighbors:", round(float(np.mean(counts)), 2),
                  "| min/max:", int(np.min(counts)), int(np.max(counts)))

    def _compute_graph_context_vec(self, asin, end):
        key = (str(asin), int(end))
        if key in self._graph_context_cache:
            return self._graph_context_cache[key]
        d_i = self.data[asin]
        idx = d_i.get("graph_context_idx", {})
        if len(idx) == 0:
            vec = np.zeros(len(GRAPH_CONTEXT_COLS), dtype=np.float32)
            self._graph_context_cache[key] = vec
            return vec
        nbrs = self.graph_neighbor_map.get(asin, [])
        if len(nbrs) == 0:
            nbrs = [asin]

        vals_total, vals_buy, vals_inst, vals_dem, active_rates = [], [], [], [], []
        strengths = []
        same_hbt, top10 = [], []
        hbt_i = d_i.get("graph_meta", {}).get("hbt", "missing")
        for b in nbrs:
            d = self.data[b]
            vals_total.append(_graph_recent_mean(d.get("total_dph", []), end, 13))
            vals_buy.append(_graph_recent_mean(d.get("buy_box_dph", []), end, 13))
            vals_inst.append(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13))
            vals_dem.append(_graph_recent_mean(d.get("demand", []), end, 13))
            x = np.asarray(d.get("in_stock_dph", d.get("instock_raw", []))[max(0, end-13):end], dtype=float)
            active_rates.append(float(np.mean(x > 0)) if len(x) else 0.0)
            strengths.append(_graph_strength_for_asin(d, end))
            same_hbt.append(1.0 if d.get("graph_meta", {}).get("hbt", "missing") == hbt_i else 0.0)
            top10.append(float(d.get("graph_meta", {}).get("ind_top10_brand", 0.0)))

        own_strength = _graph_strength_for_asin(d_i, end)
        all_strengths = strengths + [own_strength]
        rank_prior = _graph_percentile_rank(own_strength, all_strengths)
        peer_count = max(len(nbrs), 1)
        vec_map = {
            "graph_peer_total_mean13_log": np.log1p(np.mean(vals_total) if len(vals_total) else 0.0),
            "graph_peer_buybox_mean13_log": np.log1p(np.mean(vals_buy) if len(vals_buy) else 0.0),
            "graph_peer_instock_mean13_log": np.log1p(np.mean(vals_inst) if len(vals_inst) else 0.0),
            "graph_peer_demand_mean13_log": np.log1p(np.mean(vals_dem) if len(vals_dem) else 0.0),
            "graph_peer_active_rate13": float(np.mean(active_rates) if len(active_rates) else 0.0),
            "graph_peer_zero_rate13": float(1.0 - np.mean(active_rates) if len(active_rates) else 1.0),
            "graph_peer_count_log": np.log1p(peer_count),
            "graph_peer_rank_prior": rank_prior,
            "graph_same_hbt_peer_rate": float(np.mean(same_hbt) if len(same_hbt) else 0.0),
            "graph_top10_peer_rate": float(np.mean(top10) if len(top10) else 0.0),
        }
        vec = np.array([vec_map[c] for c in GRAPH_CONTEXT_COLS], dtype=np.float32)
        self._graph_context_cache[key] = vec
        return vec

    def _inject_graph_context(self, fc, d, asin, end):
        idx = d.get("graph_context_idx", {})
        if len(idx) == 0 or fc is None or len(fc) == 0:
            return fc
        base_vec = self._compute_graph_context_vec(asin, end)
        H = fc.shape[0]
        for step_h in range(H):
            # keep graph prior strongest near origin but still available at long horizon
            h_decay = 0.65 + 0.35 * np.exp(-0.06 * step_h)
            for k, col in enumerate(GRAPH_CONTEXT_COLS):
                if col in idx:
                    fc[step_h, idx[col]] = float(base_vec[k]) * h_decay
        return fc


# ---- Demand-specific overrides ----
_ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT = load_real_data
_ORIGINAL_DEMAND_DATASET_BEFORE_GRAPH_CONTEXT = DemandDataset


# =====================================================
# Parallel sample construction for DemandDataset.__init__
# =====================================================
# Building every training window's tensors is CPU-bound, pure-Python work
# (dict + torch.tensor() construction per window) -- threads don't help here
# because of the GIL, only separate processes do. This relies on the "fork"
# start method (default on Linux): a module-level global is set to the
# half-built dataset instance right before the worker pool is created, so
# each forked worker inherits it via copy-on-write for free, with no
# per-task pickling of the (potentially large) `data` dict. Workers call the
# exact same `_make_future_context_with_dph_proxies` / `_inject_graph_context`
# instance methods as the serial path -- there is no duplicated/rewritten
# logic here, so results are identical to the single-process version, just
# computed across multiple cores. If anything about the parallel path fails
# (pickling, platform without fork, etc.) it falls back to the plain serial
# loop automatically.
_MP_BUILD_DATASET = None


def _mp_build_dataset_worker(asin_starts):
    asin, starts, history, horizon = asin_starts
    ds = _MP_BUILD_DATASET
    d = ds.data[asin]
    out = []
    for start in starts:
        fc = ds._make_future_context_with_dph_proxies(
            d=d, start=start, history=history, horizon=horizon,
        )
        fc = ds._inject_graph_context(fc, d, asin, start + history)
        out.append({
            "x": np.asarray(d["features"][start:start+history], dtype=np.float32),
            "future_context": np.asarray(fc, dtype=np.float32),
            "y": np.asarray(d["demand"][start+history:start+history+horizon], dtype=np.float32),
            "asin": asin,
            "target_week": [str(w)[:10] for w in d["week"][start+history:start+history+horizon]],
            "oos": np.asarray(d["oos"][start+history:start+history+horizon], dtype=np.float32),
            "our_price": np.asarray(d["price_raw"][start+history:start+history+horizon], dtype=np.float32),
            "pkg_volume": np.asarray(d["pkg_volume_raw"][start+history:start+history+horizon], dtype=np.float32),
            "future_instock": np.asarray(d["instock_raw"][start+history:start+history+horizon], dtype=np.float32),
            "future_total_dph": np.asarray(d["total_dph_raw"][start+history:start+history+horizon], dtype=np.float32),
            "future_buy_box_dph": np.asarray(d["buy_box_dph_raw"][start+history:start+history+horizon], dtype=np.float32),
        })
    return out


def _mp_row_to_tensors(raw):
    return {
        "x": torch.from_numpy(raw["x"]),
        "future_context": torch.from_numpy(raw["future_context"]),
        "y": torch.from_numpy(raw["y"]),
        "asin": raw["asin"],
        "target_week": raw["target_week"],
        "oos": torch.from_numpy(raw["oos"]),
        "our_price": torch.from_numpy(raw["our_price"]),
        "pkg_volume": torch.from_numpy(raw["pkg_volume"]),
        "future_instock": torch.from_numpy(raw["future_instock"]),
        "future_total_dph": torch.from_numpy(raw["future_total_dph"]),
        "future_buy_box_dph": torch.from_numpy(raw["future_buy_box_dph"]),
    }


EXTERNAL_HAT_COLS = [
    "external_total_dph_hat_log",
    "external_buy_box_dph_hat_log",
    "external_instock_dph_hat_log",
]


def _reorder_future_context_keep_hats_last(data, context_cols):
    context_cols = list(context_cols)
    if not all(c in context_cols for c in EXTERNAL_HAT_COLS):
        return data, len(context_cols), context_cols
    non_hat_cols = [c for c in context_cols if c not in EXTERNAL_HAT_COLS]
    new_cols = non_hat_cols + EXTERNAL_HAT_COLS
    old_idx = [context_cols.index(c) for c in new_cols]
    for d in data.values():
        d["future_context"] = d["future_context"][:, old_idx]
        d["dph_proxy_context_idx"] = {c: new_cols.index(c) for c in d.get("dph_proxy_context_idx", {}) if c in new_cols}
        d["graph_context_idx"] = {c: new_cols.index(c) for c in GRAPH_CONTEXT_COLS if c in new_cols}
    return data, len(new_cols), new_cols


def load_real_data(data_raw, dph_cap_q=0.995):
    _wrapper_t0 = time.perf_counter()
    print("[STAGE] graph-context wrapper START", flush=True)
    data, context_dim, context_cols = _ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT(data_raw, dph_cap_q=dph_cap_q)
    data, context_dim, context_cols = _graph_add_context_cols_to_data(data, context_cols, data_raw=data_raw)
    data, context_dim, context_cols = _reorder_future_context_keep_hats_last(data, context_cols)
    print("\n" + "=" * 100)
    print("PACKAGE-AWARE RELATION GRAPH FEATURES ADDED TO DEMAND FUTURE_CONTEXT")
    print("Graph cols:", GRAPH_CONTEXT_COLS)
    print("External exposure hats kept as last 3 columns:", all(c in context_cols[-3:] for c in EXTERNAL_HAT_COLS))
    print("New context dim:", context_dim)
    print("=" * 100)
    print(f"[STAGE] graph-context wrapper DONE | {(time.perf_counter()-_wrapper_t0)/60:.2f} min", flush=True)
    return data, context_dim, context_cols


class DemandDataset(_GraphContextMixin, _ORIGINAL_DEMAND_DATASET_BEFORE_GRAPH_CONTEXT):
    def __init__(self, data, history=52, horizon=3, mode="train", val_weeks=20,
                 min_graph_neighbors=3, num_build_workers=None):
        _ds_t0 = time.perf_counter()
        print(f"[STAGE] DemandDataset START | mode={mode} | ASINs={len(data):,} | history={history} | horizon={horizon}", flush=True)
        self.data = data
        self.history = history
        self.horizon = horizon
        self.val_weeks = val_weeks
        self._init_graph_context(min_graph_neighbors=min_graph_neighbors)
        self.samples = []

        asin_starts = []
        for asin, d in data.items():
            T = len(d["demand"])
            if mode == "train":
                starts = list(range(max(0, T - val_weeks - horizon - history + 1)))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []
            if starts:
                asin_starts.append((asin, starts, history, horizon))

        if num_build_workers is None:
            num_build_workers = max(1, (os.cpu_count() or 1) - 1)

        built_in_parallel = False
        if num_build_workers > 1 and len(asin_starts) >= 2 and mp.get_start_method(allow_none=True) != "spawn":
            try:
                global _MP_BUILD_DATASET
                _MP_BUILD_DATASET = self  # inherited via fork copy-on-write, not pickled per task
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=min(num_build_workers, len(asin_starts))) as pool:
                    for raw_chunk in pool.imap(_mp_build_dataset_worker, asin_starts, chunksize=4):
                        for raw in raw_chunk:
                            self.samples.append(_mp_row_to_tensors(raw))
                built_in_parallel = True
            except Exception as e:
                print(f"Parallel DemandDataset build failed ({type(e).__name__}: {e}); "
                      f"falling back to serial construction.")
                self.samples = []
            finally:
                _MP_BUILD_DATASET = None

        if not built_in_parallel:
            for asin, starts, history, horizon in asin_starts:
                d = data[asin]
                for start in starts:
                    fc = self._make_future_context_with_dph_proxies(
                        d=d, start=start, history=history, horizon=horizon,
                    )
                    fc = self._inject_graph_context(fc, d, asin, start + history)
                    self.samples.append({
                        "x": torch.tensor(d["features"][start:start+history], dtype=torch.float32),
                        "future_context": torch.tensor(fc, dtype=torch.float32),
                        "y": torch.tensor(d["demand"][start+history:start+history+horizon], dtype=torch.float32),
                        "asin": asin,
                        "target_week": [str(w)[:10] for w in d["week"][start+history:start+history+horizon]],
                        "oos": torch.tensor(d["oos"][start+history:start+history+horizon], dtype=torch.float32),
                        "our_price": torch.tensor(
                            d["price_raw"][start+history:start+history+horizon], dtype=torch.float32),
                        "pkg_volume": torch.tensor(
                            d["pkg_volume_raw"][start+history:start+history+horizon], dtype=torch.float32),
                        "future_instock": torch.tensor(
                            d["instock_raw"][start+history:start+history+horizon], dtype=torch.float32),
                        "future_total_dph": torch.tensor(
                            d["total_dph_raw"][start+history:start+history+horizon], dtype=torch.float32),
                        "future_buy_box_dph": torch.tensor(
                            d["buy_box_dph_raw"][start+history:start+history+horizon], dtype=torch.float32),
                    })

        print(f"[STAGE] DemandDataset DONE | mode={mode} | samples={len(self.samples):,} | {(time.perf_counter()-_ds_t0)/60:.2f} min", flush=True)


def run_demand_with_predicted_exposure_all_modes_graph(
    data_raw1,
    scot_df,
    exposure_result_or_hat,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    remove_oos_dp=True,
):
    """Run buybox_only, instock_only, and all3 with graph context enabled."""
    out = {}
    out["buybox_graph"] = run_demand_with_predicted_exposure_buybox_only(
        data_raw1=data_raw1, scot_df=scot_df, exposure_result_or_hat=exposure_result_or_hat,
        n_asins=n_asins, seed=seed, epochs=epochs, history=history, horizon=horizon,
        d_model=d_model, d_z=d_z, batch_size=batch_size, M_eval=M_eval, remove_oos_dp=remove_oos_dp,
    )
    out["instock_graph"] = run_demand_with_predicted_exposure_instock_only(
        data_raw1=data_raw1, scot_df=scot_df, exposure_result_or_hat=exposure_result_or_hat,
        n_asins=n_asins, seed=seed, epochs=epochs, history=history, horizon=horizon,
        d_model=d_model, d_z=d_z, batch_size=batch_size, M_eval=M_eval, remove_oos_dp=remove_oos_dp,
    )
    out["all3_graph"] = run_demand_with_predicted_exposure_all3(
        data_raw1=data_raw1, scot_df=scot_df, exposure_result_or_hat=exposure_result_or_hat,
        n_asins=n_asins, seed=seed, epochs=epochs, history=history, horizon=horizon,
        d_model=d_model, d_z=d_z, batch_size=batch_size, M_eval=M_eval, remove_oos_dp=remove_oos_dp,
    )
    return out


def summarize_graph_context_from_demand_result(result):
    cols = result.get("context_cols", []) if isinstance(result, dict) else []
    present = [c for c in GRAPH_CONTEXT_COLS if c in cols]
    print("Graph context cols present:", present)
    print("n_graph_cols:", len(present), "| context_dim:", len(cols))
    print("external hats last3:", cols[-3:] if len(cols) >= 3 else cols)
    return {"graph_cols": present, "n_graph_cols": len(present), "context_dim": len(cols), "last3": cols[-3:] if len(cols) >= 3 else cols}

# ============================================================
# V12c3 H3 recommended current run: instock-only demand
# ============================================================

def run_demand_current_best_instock_only(
    data_raw1,
    scot_df,
    exposure_result_or_hat=None,
    exposure_hat_csv_path="exposure_hat_for_demand_h3_nonrolling.csv",
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.08,
    lambda_under=0.0,
    q_active_weight=2.0,
    q_tail_weight=0.30,
    beta_tail=0.5,
    remove_oos_dp=True,
):
    """Recommended v12 run.

    The exposure input can be supplied in either form:
      1) exposure_result_or_hat: an exposure result dict / DataFrame / CSV path; or
      2) exposure_hat_csv_path: a CSV path.

    If neither is explicitly supplied, this function automatically reads
    ``exposure_hat_for_demand_h3_nonrolling.csv`` from the current working directory.
    """
    # Local-CSV-friendly behavior: allow the commonly used keyword
    # exposure_hat_csv_path directly on this main entry point.
    if exposure_result_or_hat is None:
        exposure_result_or_hat = exposure_hat_csv_path

    print(f"Demand exposure source: {exposure_result_or_hat}")
    print(f"Demand horizon: {horizon} weeks")
    print("SCOT evaluation alignment: only forecast weeks/horizons present in the 3-week demand forecast are joined and scored.")

    return run_external_exposure3_in_old_decoder_style(
        data_raw1=data_raw1,
        scot_df=scot_df,
        exposure3_hat=exposure_result_or_hat,
        exposure_mode="instock_only",
        n_asins=n_asins,
        seed=seed,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=5,
        lambda_z_reg=1.0,
        lambda_under=lambda_under,
        q_active_weight=q_active_weight,
        q_tail_weight=q_tail_weight,
        lambda_stock=0.0,
        lambda_stock_mean_weight=0.0,
        remove_extreme=True,
        extreme_q=0.99,
        run_wape=True,
        remove_oos_dp=remove_oos_dp,
    )

def run_demand_current_best_instock_only_from_hat_csv(
    data_raw1,
    scot_df,
    exposure_hat_csv_path="exposure_hat_for_demand_h3_nonrolling.csv",
    **kwargs,
):
    """Convenience wrapper: run current best demand using saved exposure hat CSV.

    Default path matches the exposure v27.3 HIER AUTOSAVE LOCALCSV output,
    so you can omit exposure_hat_csv_path when the CSV is in the current
    notebook working directory.
    """
    return run_demand_current_best_instock_only(
        data_raw1=data_raw1,
        scot_df=scot_df,
        exposure_hat_csv_path=exposure_hat_csv_path,
        **kwargs,
    )


# ============================================================================
# JOINT EXPOSURE + DEMAND H3 END-TO-END V1 (NON-ROLLING)
# Shared history encoder + two separate future decoders.
# Demand branch consumes predicted exposure from the exposure branch.
# ============================================================================

class JointExposureDemandH3(nn.Module):
    """Experimental joint H3 model.

    Structure
    ---------
    shared demand-history encoder
        -> exposure decoder -> total/buy-box/in-stock H1-H3
        -> demand decoder, conditioned on predicted exposure -> NB demand H1-H3

    By default ``detach_exposure_for_demand=False`` for true end-to-end training:
    demand loss can update the exposure decoder through predicted exposure.
    Set it to True for the safer semi-joint ablation.
    """
    def __init__(self, input_dim, context_dim, d_model=32, d_z=16,
                 horizon=3, prior_scale=0.3,
                 detach_exposure_for_demand=False):
        super().__init__()
        if horizon != 3:
            raise ValueError("This experimental file is intentionally H3 only.")
        if context_dim < 3:
            raise ValueError("future_context must reserve its final 3 columns for exposure hats.")

        self.horizon = horizon
        self.context_dim = context_dim
        self.base_context_dim = context_dim - 3
        self.d_z = d_z
        self.detach_exposure_for_demand = bool(detach_exposure_for_demand)

        # One shared history encoder for both tasks.
        self.encoder = TCNSparseAttnEncoder(input_dim, d_model, horizon)

        # Exposure decoder: independent future decoder/head.
        self.exposure_decoder = DecoderPeakCrossAttention(
            d_model=d_model,
            context_dim=self.base_context_dim,
            horizon=horizon,
            n_heads=4,
            dropout=0.1,
            future_tcn_dilations=(1, 2),
        )
        exp_in = d_model + self.base_context_dim
        self.exp_total_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.exp_ratio_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )
        self.exp_active_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3)
        )

        # Demand latent-index and uncertainty branches.
        self.z_generator = ContextZGenerator(d_model, context_dim, d_z, horizon)
        self.epinet = Epinet(d_model, d_z, horizon, prior_scale)

        # Separate demand decoder.
        self.demand_decoder = DecoderPeakCrossAttention(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            n_heads=4,
            dropout=0.1,
            future_tcn_dilations=(1, 2),
        )
        demand_dec_in = d_model + context_dim
        self.peak_mu_head = nn.Sequential(
            nn.Linear(demand_dec_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.peak_gate_head = nn.Sequential(
            nn.Linear(demand_dec_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.decoder_epinet = DecoderEpinet(
            d_decoder=d_model,
            context_dim=context_dim,
            d_z=d_z,
            hidden=64,
            prior_scale=0.20,
        )
        self.decoder_epinet_scale = 0.35

        nn.init.constant_(self.exp_total_head[-1].bias, 5.5)
        nn.init.zeros_(self.exp_ratio_head[-1].weight)
        nn.init.constant_(self.exp_ratio_head[-1].bias, 0.7)
        nn.init.zeros_(self.peak_mu_head[-1].weight)
        nn.init.constant_(self.peak_mu_head[-1].bias, -4.0)
        nn.init.zeros_(self.peak_gate_head[-1].weight)
        nn.init.constant_(self.peak_gate_head[-1].bias, -1.5)

    def _base_future_context(self, future_context):
        # Final 3 external-hat slots are intentionally excluded from exposure prediction.
        return future_context[:, :, :self.base_context_dim]

    def predict_exposure(self, H_enc, b_t, peak_score, future_context):
        fc_base = self._base_future_context(future_context)
        dec_h = self.exposure_decoder(
            H_enc, fc_base, b_t=b_t, peak_score=peak_score
        )
        z = torch.cat([dec_h, fc_base], dim=-1)

        total = F.softplus(self.exp_total_head(z).squeeze(-1))
        ratios = torch.sigmoid(self.exp_ratio_head(z))
        buy = total * ratios[..., 0]
        instock = total * ratios[..., 1]
        exposure = torch.stack([total, buy, instock], dim=-1)
        active_logits = self.exp_active_head(z)
        return exposure, active_logits, dec_h

    def _demand_context(self, future_context, exposure_hat):
        exp_for_demand = (
            exposure_hat.detach() if self.detach_exposure_for_demand else exposure_hat
        )
        fc = future_context.clone()
        fc[:, :, -3:] = torch.log1p(exp_for_demand.clamp_min(0.0))
        return fc

    def _combine_one_z(self, mu_base, alpha_base, phi, dec_h,
                       demand_context, peak_mu, peak_gate, z):
        mu_e, al_e = self.epinet(phi, z)
        dec_mu_e, dec_al_e = self.decoder_epinet(dec_h, demand_context, z)
        dec_mu_e = self.decoder_epinet_scale * dec_mu_e
        dec_al_e = self.decoder_epinet_scale * dec_al_e

        mu_normal = F.softplus(mu_base + mu_e + dec_mu_e)
        mu = mu_normal + peak_gate * peak_mu
        alpha = F.softplus(alpha_base + al_e + dec_al_e) + 1e-4
        return mu, alpha

    def forward(self, x, future_context, nZ=8):
        H_enc, h_t, b_t, peak_score = self.encoder.encode(x)

        exposure_hat, exposure_active_logits, exposure_dec_h = self.predict_exposure(
            H_enc, b_t, peak_score, future_context
        )
        demand_context = self._demand_context(future_context, exposure_hat)

        mu_base = F.softplus(self.encoder.base_head(h_t))
        alpha_base = F.softplus(self.encoder.alpha_head(h_t)) + 1e-4
        phi = h_t.detach()
        z_mean, z_std = self.z_generator(phi, demand_context)
        z_reg = 0.001 * (z_mean.square() + z_std.square()).mean()

        demand_dec_h = self.demand_decoder(
            H_enc, demand_context, b_t=b_t, peak_score=peak_score
        )
        demand_dec_in = torch.cat([demand_dec_h, demand_context], dim=-1)
        peak_mu = F.softplus(self.peak_mu_head(demand_dec_in).squeeze(-1))
        peak_gate = torch.sigmoid(self.peak_gate_head(demand_dec_in).squeeze(-1))

        preds = []
        for _ in range(nZ):
            z = z_mean + z_std * torch.randn_like(z_mean)
            preds.append(self._combine_one_z(
                mu_base, alpha_base, phi, demand_dec_h,
                demand_context, peak_mu, peak_gate, z
            ))

        return {
            "demand_preds": preds,
            "z_reg": z_reg,
            "exposure_hat": exposure_hat,
            "exposure_active_logits": exposure_active_logits,
            "demand_context": demand_context,
        }

    def predict(self, x, future_context, M=100):
        self.eval()
        with torch.no_grad():
            out = self.forward(x, future_context, nZ=M)
            samples = []
            for mu, alpha in out["demand_preds"]:
                dist = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp_min(1e-4),
                    probs=(mu * alpha / (1.0 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                samples.append(dist.sample().float())
            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = torch.maximum(samples.quantile(0.7, dim=1), p50)
            p90 = torch.maximum(samples.quantile(0.9, dim=1), p70)
            return p50, p70, p90, out["exposure_hat"]


def joint_exposure_loss(exposure_hat, active_logits, true_exposure):
    """Gentle H3 exposure loss on log magnitude + occurrence + hierarchy."""
    true_exposure = true_exposure.clamp_min(0.0)
    pred_log = torch.log1p(exposure_hat.clamp_min(0.0))
    true_log = torch.log1p(true_exposure)

    mag = F.smooth_l1_loss(pred_log, true_log)
    active_target = (true_exposure > 0).float()
    active = F.binary_cross_entropy_with_logits(active_logits, active_target)

    # Mild mean calibration per channel/horizon.
    pred_mean = exposure_hat.mean(dim=0)
    true_mean = true_exposure.mean(dim=0)
    mean_calib = torch.abs(
        torch.log1p(pred_mean) - torch.log1p(true_mean)
    ).mean()

    # Hierarchical constraints are already architectural; keep only a tiny numerical penalty.
    hierarchy = (
        F.relu(exposure_hat[..., 1] - exposure_hat[..., 0]).mean()
        + F.relu(exposure_hat[..., 2] - exposure_hat[..., 0]).mean()
    )
    return mag + 0.20 * active + 0.10 * mean_calib + 0.01 * hierarchy


def train_joint_exposure_demand_h3(
    model, tr_ld, va_ld, epochs=60, lr=1e-3, nZ=8,
    beta_tail=0.5, lambda_under=0.0, lambda_over=0.0, lambda_z_reg=1.0,
    lambda_exposure=0.50, patience=6,
):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        sums = {"total": 0.0, "demand": 0.0, "exposure": 0.0}

        for b in tr_ld:
            b = _move_batch_to_device(b, DEVICE)
            true_exp = torch.stack([
                b["future_total_dph"],
                b["future_buy_box_dph"],
                b["future_instock"],
            ], dim=-1)
            out = model(b["x"], b["future_context"], nZ=nZ)

            demand_nll = sum(
                tail_weighted_negbin_nll(b["y"], mu, alpha, beta_tail=beta_tail)
                for mu, alpha in out["demand_preds"]
            ) / nZ
            demand_loss = demand_nll + lambda_z_reg * out["z_reg"]

            # Optional asymmetric bias losses. Defaults are both zero, so the
            # model is trained without an explicit under/over-forecast preference.
            if lambda_under > 0.0 or lambda_over > 0.0:
                mu_mean = torch.stack(
                    [m for m, _ in out["demand_preds"]], dim=1
                ).mean(dim=1)
                if lambda_under > 0.0:
                    demand_loss = demand_loss + lambda_under * active_underforecast_loss(
                        b["y"], mu_mean, log_scale=True
                    )
                if lambda_over > 0.0:
                    demand_loss = demand_loss + lambda_over * active_overforecast_loss(
                        b["y"], mu_mean, log_scale=True
                    )

            exp_loss = joint_exposure_loss(
                out["exposure_hat"], out["exposure_active_logits"], true_exp
            )
            loss = demand_loss + lambda_exposure * exp_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sums["total"] += float(loss.item())
            sums["demand"] += float(demand_loss.item())
            sums["exposure"] += float(exp_loss.item())

        sch.step()

        model.eval()
        val_total = 0.0
        val_demand = 0.0
        val_exp = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = _move_batch_to_device(b, DEVICE)
                true_exp = torch.stack([
                    b["future_total_dph"], b["future_buy_box_dph"], b["future_instock"]
                ], dim=-1)
                out = model(b["x"], b["future_context"], nZ=min(8, nZ))
                mu_mean = torch.stack([m for m, _ in out["demand_preds"]], dim=1).mean(dim=1)
                al_mean = torch.stack([a for _, a in out["demand_preds"]], dim=1).mean(dim=1)
                dloss = tail_weighted_negbin_nll(
                    b["y"], mu_mean, al_mean, beta_tail=beta_tail
                )
                eloss = joint_exposure_loss(
                    out["exposure_hat"], out["exposure_active_logits"], true_exp
                )
                val_demand += float(dloss.item())
                val_exp += float(eloss.item())
                val_total += float((dloss + lambda_exposure * eloss).item())

        ntr = max(1, len(tr_ld)); nva = max(1, len(va_ld))
        val_total /= nva; val_demand /= nva; val_exp /= nva
        improved = val_total < best_val
        if improved:
            best_val = val_total
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | total={sums['total']/ntr:.4f} | "
            f"demand={sums['demand']/ntr:.4f} | exposure={sums['exposure']/ntr:.4f} | "
            f"val_total={val_total:.4f} | val_demand={val_demand:.4f} | "
            f"val_exposure={val_exp:.4f}" + (" *" if improved else "")
        )
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"Best joint val: {best_val:.4f}")


def generate_joint_forecast_df(model, va_ld, M=100):
    rows = []
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = _move_batch_to_device(b, DEVICE)
            p50, p70, p90, exp_hat = model.predict(b["x"], b["future_context"], M=M)
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "fcst_week_index": h + 1,
                        "fbi_demand": b["y"][i, h].item(),
                        "our_price": b["our_price"][i, h].item(),
                        "true_amt": b["y"][i, h].item() * b["our_price"][i, h].item(),
                        "pkg_volume": b["pkg_volume"][i, h].item(),
                        "true_size": b["y"][i, h].item() * b["pkg_volume"][i, h].item(),
                        "true_future_total_dph": b["future_total_dph"][i, h].item(),
                        "true_future_buy_box_dph": b["future_buy_box_dph"][i, h].item(),
                        "true_future_instock": b["future_instock"][i, h].item(),
                        "pred_total_dph_hat": exp_hat[i, h, 0].item(),
                        "pred_buy_box_dph_hat": exp_hat[i, h, 1].item(),
                        "pred_instock_dph_hat": exp_hat[i, h, 2].item(),
                        "scot_oos": b["oos"][i, h].item(),
                        "oos": b["oos"][i, h].item(),
                        "oos_status": b["oos"][i, h].item(),
                        "p50_amxl": p50[i, h].item(),
                        "p70_amxl": p70[i, h].item(),
                        "p90_amxl": p90[i, h].item(),
                    })
    return pd.DataFrame(rows)


def _joint_exposure_metrics(df):
    out = []
    for name, true_col, pred_col in [
        ("total", "true_future_total_dph", "pred_total_dph_hat"),
        ("buy_box", "true_future_buy_box_dph", "pred_buy_box_dph_hat"),
        ("in_stock", "true_future_instock", "pred_instock_dph_hat"),
    ]:
        y = df[true_col].to_numpy(float)
        p = df[pred_col].to_numpy(float)
        denom = np.abs(y).sum() + 1e-8
        out.append({
            "channel": name,
            "wape": np.abs(p-y).sum()/denom,
            "ratio": p.sum()/(y.sum()+1e-8),
            "corr": np.corrcoef(y, p)[0,1] if np.std(y)>1e-8 and np.std(p)>1e-8 else np.nan,
            "true_mean": y.mean(),
            "pred_mean": p.mean(),
        })
    return pd.DataFrame(out)


def run_joint_exposure_demand_h3_end2end(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    beta_tail=0.5,
    patience=6,
    lambda_under=0.0,
    lambda_over=0.0,
    lambda_exposure=0.50,
    detach_exposure_for_demand=False,
    remove_extreme=True,
    extreme_q=0.99,
    output_csv="joint_exposure_demand_h3_forecast_two_step_wape_aligned.csv",
    remove_oos_dp=True,
):
    """Run non-rolling H3 joint model on one snapshot.

    V1 keeps the existing demand preprocessing and SCOT cohort logic, but it no
    longer reads an exposure CSV. Future exposure is predicted internally.
    """
    if horizon != 3:
        raise ValueError("Use horizon=3 for this file.")

    print("=" * 88)
    print("JOINT EXPOSURE + DEMAND H3 END-TO-END V1.2 | TWO-STEP WAPE-ALIGNED")
    print("Shared encoder: YES | Separate decoders: YES")
    print(f"Demand gradient through exposure_hat: {not detach_exposure_for_demand}")
    print("=" * 88)

    if scot_df is None:
        raise ValueError(
            "scot_df is required: WAPE is reported exclusively via the "
            "boss-approved calculate_wape_using_lp_oos2/quick_error_check "
            "pipeline, which needs real SCOT p50/p70 forecasts to compare "
            "against. Pass scot_df instead of running without it."
        )

    raw = data_raw1.copy()
    # Reserve the last three future-context columns without reading an external CSV.
    # NOTE: load_real_data's external-exposure attach step requires these exact
    # _log-suffixed names (see `required` in attach_external_exposure3_to_raw_data's
    # override) to append the 3 dedicated hat slots. Using _dph here silently skips
    # that step, and JointExposureDemandH3 then treats whatever real feature ends up
    # last in context_cols (e.g. a graph-peer column) as the reserved hat slot instead.
    for c in ["attn_pred_total_log", "attn_pred_buy_box_log", "attn_pred_instock_log"]:
        raw[c] = 0.0

    data_small_raw, sample_asin_df, intersect_asin_df = prepare_data_from_sample_scot_intersection(
        data_raw1=raw, scot_df=scot_df, n_asins=n_asins, seed=seed
    )

    data_small, asin_stats = add_zero_rate_group(data_small_raw, (0.4, 0.7))
    # Match the two-step H3 demand model exactly:
    # zero_group is diagnostic only; do not filter to high_sparse.
    data_use = data_small.copy()
    print("\n" + "=" * 80)
    print("JOINT COHORT MATCHED TO TWO-STEP H3")
    print("=" * 80)
    print("Sampled ASINs:", len(sample_asin_df))
    print("ASINs after SCOT intersection:", len(intersect_asin_df))
    print("ASINs before extreme filtering:", data_use["asin"].nunique())
    print("Sparse groups are diagnostics only; no high_sparse-only filtering.")

    if remove_extreme:
        data_use, removed_extreme, extreme_cap = filter_extreme_asins(data_use, q=extreme_q)
    else:
        removed_extreme, extreme_cap = pd.DataFrame(), np.nan

    data, context_dim, context_cols = load_real_data(data_use, dph_cap_q=0.995)
    if context_cols[-3:] != EXTERNAL_HAT_COLS:
        raise RuntimeError(
            "load_real_data did not append the reserved exposure-hat "
            f"placeholder columns as the last 3 context columns "
            f"(got {context_cols[-3:]}, expected {EXTERNAL_HAT_COLS}). "
            "JointExposureDemandH3 assumes context_dim - 3 == base context "
            "and would silently overwrite whatever real feature ended up "
            "last instead. Check the placeholder column names set above "
            "match the _log-suffixed names load_real_data requires."
        )
    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val", horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, pin_memory=(DEVICE.type == "cuda"))
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError("No train/validation samples. Check history/horizon and data length.")

    input_dim = int(tr_ds[0]["x"].shape[-1])
    print(f"ASINs used: {len(data):,} | Train={len(tr_ds):,} | Val={len(va_ds):,}")
    print(f"input_dim={input_dim} | context_dim={context_dim} | H={horizon}")

    model = JointExposureDemandH3(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=0.3,
        detach_exposure_for_demand=detach_exposure_for_demand,
    ).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training on: {DEVICE}")

    train_joint_exposure_demand_h3(
        model, tr_ld, va_ld,
        epochs=epochs, lr=1e-3, nZ=8,
        beta_tail=beta_tail,
        lambda_under=lambda_under,
        lambda_over=lambda_over,
        lambda_z_reg=1.0,
        lambda_exposure=lambda_exposure,
        patience=patience,
    )

    forecast_df = generate_joint_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "all_sample_scot_intersection"
    forecast_df.to_csv(output_csv, index=False)

    exposure_metrics = _joint_exposure_metrics(forecast_df)
    exposure_diag = compute_exposure_hat_diagnostics(forecast_df)

    print("\nJOINT EXPOSURE METRICS (overall, legacy summary)")
    print(exposure_metrics.round(4).to_string(index=False))
    print_exposure_hat_diagnostics(
        exposure_diag,
        title="EXPOSURE HAT DIAGNOSTICS | NON-ROLLING",
    )
    print(f"Saved joint forecast: {output_csv}")

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "exposure_metrics": exposure_metrics,
        "exposure_hat_diagnostics": exposure_diag,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "asin_stats": asin_stats,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
    }

    # Demand WAPE is reported exclusively through the boss-approved
    # calculate_wape_using_lp_oos2 / quick_error_check / weekly_error_check
    # pipeline below -- no locally rolled abs(pred-true)/true formula.
    # Cohort and evaluation order match the standalone two-step H3 model:
    # sample n_asins -> SCOT intersection -> all sparse groups -> same extreme
    # filter -> H1-H3 validation rows -> same OOS-DP filter -> same real-SCOT
    # join -> same calculate_wape_using_lp_oos2 / quick_error_check WAPE.
    result["real_scot_outputs"] = run_high_sparse_scot_alignment_wape(
        result=result,
        scot_df=scot_df,
        data_raw1=data_raw1,
        asin_stats=asin_stats,
        remove_oos_dp=remove_oos_dp,
        source="lp",
    )
    result["final_wape"] = result["real_scot_outputs"]

    return result


# ============================================================================
# NON-ROLLING USAGE (MODEL DEFINITION PRESERVED)
# ============================================================================
# joint_result_h3 = run_joint_exposure_demand_h3_end2end(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     epochs=60,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,  # true end-to-end
#     output_csv="joint_exposure_demand_h3_forecast_two_step_wape_aligned.csv",
# )





# ============================================================================
# EXPOSURE HAT DIAGNOSTICS: OVERALL / HORIZON / CUT-HORIZON
# ============================================================================

def _safe_exposure_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_exposure_active_auc(y_true, score):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return np.nan

    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    active = (y_true > 0).astype(int)

    if len(np.unique(active)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(active, score))
    except Exception:
        return np.nan


def _exposure_metric_row(
    sub,
    channel,
    true_col,
    pred_col,
):
    valid = sub[
        ["asin", true_col, pred_col]
    ].dropna()

    if valid.empty:
        return {
            "channel": channel,
            "n_rows": 0,
            "n_asins": 0,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "pred_true_ratio": np.nan,
            "wape": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "overbias": np.nan,
            "underbias": np.nan,
            "true_zero_rate": np.nan,
            "pred_zero_rate": np.nan,
            "active_auc": np.nan,
        }

    y_true = pd.to_numeric(
        valid[true_col], errors="coerce"
    ).to_numpy(dtype=float)
    y_pred = pd.to_numeric(
        valid[pred_col], errors="coerce"
    ).to_numpy(dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) == 0:
        return {
            "channel": channel,
            "n_rows": 0,
            "n_asins": 0,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "pred_true_ratio": np.nan,
            "wape": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "overbias": np.nan,
            "underbias": np.nan,
            "true_zero_rate": np.nan,
            "pred_zero_rate": np.nan,
            "active_auc": np.nan,
        }

    error = y_pred - y_true
    denom = np.abs(y_true).sum() + 1e-8

    return {
        "channel": channel,
        "n_rows": len(y_true),
        "n_asins": valid.loc[finite, "asin"].astype(str).nunique(),
        "true_mean": y_true.mean(),
        "pred_mean": y_pred.mean(),
        "pred_true_ratio": y_pred.sum() / (y_true.sum() + 1e-8),
        "wape": np.abs(error).sum() / denom,
        "mae": np.abs(error).mean(),
        "corr": _safe_exposure_corr(y_true, y_pred),
        "overbias": np.maximum(error, 0.0).sum() / denom,
        "underbias": np.maximum(-error, 0.0).sum() / denom,
        "true_zero_rate": np.mean(y_true <= 0),
        "pred_zero_rate": np.mean(y_pred <= 0),
        "active_auc": _safe_exposure_active_auc(y_true, y_pred),
    }


def compute_exposure_hat_diagnostics(
    forecast_df,
    data_cut=None,
):
    """
    Compute exposure_hat diagnostics for total, buy-box, and in-stock.

    Returns:
      overall_df
      by_horizon_df
      by_cut_horizon_df
    """
    df = forecast_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    channel_specs = [
        (
            "total",
            "true_future_total_dph",
            "pred_total_dph_hat",
        ),
        (
            "buy_box",
            "true_future_buy_box_dph",
            "pred_buy_box_dph_hat",
        ),
        (
            "in_stock",
            "true_future_instock",
            "pred_instock_dph_hat",
        ),
    ]

    required = ["asin", "fcst_week_index"]
    for _, true_col, pred_col in channel_specs:
        required.extend([true_col, pred_col])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing exposure diagnostic columns: {missing}"
        )

    df["asin"] = df["asin"].astype(str)
    df["fcst_week_index"] = pd.to_numeric(
        df["fcst_week_index"], errors="coerce"
    )

    if data_cut is not None:
        df["data_cut"] = pd.Timestamp(data_cut)

    overall_rows = []
    by_h_rows = []
    by_cut_h_rows = []

    for channel, true_col, pred_col in channel_specs:
        overall_rows.append(
            _exposure_metric_row(
                df,
                channel,
                true_col,
                pred_col,
            )
        )

    horizons = sorted(
        int(h)
        for h in df["fcst_week_index"].dropna().unique()
    )

    for horizon in horizons:
        sub_h = df[df["fcst_week_index"] == horizon]

        for channel, true_col, pred_col in channel_specs:
            row = _exposure_metric_row(
                sub_h,
                channel,
                true_col,
                pred_col,
            )
            row["horizon"] = horizon
            by_h_rows.append(row)

    if "data_cut" in df.columns:
        for cut_value, sub_cut in df.groupby(
            "data_cut", dropna=False
        ):
            for horizon in horizons:
                sub_h = sub_cut[
                    sub_cut["fcst_week_index"] == horizon
                ]

                for channel, true_col, pred_col in channel_specs:
                    row = _exposure_metric_row(
                        sub_h,
                        channel,
                        true_col,
                        pred_col,
                    )
                    row["data_cut"] = cut_value
                    row["horizon"] = horizon
                    by_cut_h_rows.append(row)

    return {
        "overall": pd.DataFrame(overall_rows),
        "by_horizon": pd.DataFrame(by_h_rows),
        "by_cut_horizon": pd.DataFrame(by_cut_h_rows),
    }


def print_exposure_hat_diagnostics(
    diagnostics,
    title="EXPOSURE HAT DIAGNOSTICS",
):
    overall_df = diagnostics["overall"]
    by_horizon_df = diagnostics["by_horizon"]

    print("\n" + "=" * 100)
    print(title + " — OVERALL")
    print("=" * 100)
    print(
        overall_df[
            [
                "channel",
                "n_rows",
                "n_asins",
                "true_mean",
                "pred_mean",
                "pred_true_ratio",
                "wape",
                "corr",
                "overbias",
                "underbias",
                "active_auc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print(title + " — BY HORIZON")
    print("=" * 100)
    print(
        by_horizon_df[
            [
                "horizon",
                "channel",
                "n_rows",
                "n_asins",
                "true_mean",
                "pred_mean",
                "pred_true_ratio",
                "wape",
                "corr",
                "overbias",
                "underbias",
                "active_auc",
            ]
        ]
        .sort_values(["horizon", "channel"])
        .round(4)
        .to_string(index=False)
    )


# ============================================================================
# ROLLING S3 RUNNER: JOINT H3 + MATCHED SCOT P50/P70
# ============================================================================

import io
import re
from pathlib import Path
import boto3


ROLLING_S3_BUCKET = "amxl-asin-forecast590184089576"
ROLLING_DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)
ROLLING_SCOT_PREFIX = "amxl-asin-forecast-intern/scotforecast/"


def _list_s3_keys(bucket, prefix, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)

    return keys



def list_joint_rolling_snapshot_pairs(
    bucket=ROLLING_S3_BUCKET,
    data_prefix=ROLLING_DATA_PREFIX,
    scot_prefix=ROLLING_SCOT_PREFIX,
    s3_client=None,
):
    """
    Build a leakage-controlled rolling backtest pairing.

    For origin snapshot date D (Saturday):
      - SCOT FCD is D + 1 day (Sunday)
      - evaluation snapshot is D + 21 days

    Why D + 21?
      The evaluation snapshot's final observed Sunday is the SCOT H3 week.
      Therefore the existing non-rolling validation logic (last 3 weeks)
      produces exactly:
        H1 = FCD
        H2 = FCD + 7 days
        H3 = FCD + 14 days

    Example:
      origin snapshot: 2025-10-04, max observed week 2025-09-28
      SCOT FCD:        2025-10-05
      eval snapshot:   2025-10-25, max observed week 2025-10-19
      model val H1-H3: 2025-10-05, 2025-10-12, 2025-10-19
    """
    s3_client = s3_client or boto3.client("s3")

    data_keys = _list_s3_keys(bucket, data_prefix, s3_client)
    scot_keys = _list_s3_keys(bucket, scot_prefix, s3_client)

    data_pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )
    scot_pattern = re.compile(
        r"from(\d{4}-\d{2}-\d{2})_20weeks__"
        r"headbody_scot_fcst_(?:no_refresh|refresh)\.parquet$"
    )

    data_by_date = {}
    for key in data_keys:
        match = data_pattern.search(key)
        if match:
            data_by_date[pd.Timestamp(match.group(1)).normalize()] = key

    scot_by_date = {}
    for key in scot_keys:
        match = scot_pattern.search(key)
        if not match:
            continue

        fcd = pd.Timestamp(match.group(1)).normalize()
        previous = scot_by_date.get(fcd)
        if previous is None or (
            "no_refresh" in key and "no_refresh" not in previous
        ):
            scot_by_date[fcd] = key

    rows = []
    for origin_cut, origin_key in sorted(data_by_date.items()):
        scot_fcd = origin_cut + pd.Timedelta(days=1)
        eval_cut = origin_cut + pd.Timedelta(days=21)

        rows.append({
            "data_cut": origin_cut,
            "scot_fcd": scot_fcd,
            "eval_cut": eval_cut,
            "data_key": origin_key,
            "eval_data_key": data_by_date.get(eval_cut),
            "scot_key": scot_by_date.get(scot_fcd),
            "has_eval_snapshot": eval_cut in data_by_date,
            "has_scot": scot_fcd in scot_by_date,
        })

    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs["is_complete_pair"] = (
            pairs["has_eval_snapshot"] & pairs["has_scot"]
        )
    else:
        pairs["is_complete_pair"] = pd.Series(dtype=bool)

    print("=" * 96)
    print("ROLLING ORIGIN / EVALUATION SNAPSHOT / SCOT PAIR CHECK")
    print("=" * 96)
    print("Feature snapshots:", len(data_by_date))
    print("SCOT forecast files:", len(scot_by_date))
    print(
        "Complete rolling pairs:",
        int(pairs["is_complete_pair"].sum()) if len(pairs) else 0,
    )
    print(
        "Missing evaluation snapshot:",
        int((~pairs["has_eval_snapshot"]).sum()) if len(pairs) else 0,
    )
    print(
        "Missing SCOT file:",
        int((~pairs["has_scot"]).sum()) if len(pairs) else 0,
    )

    return pairs


def _read_s3_csv(bucket, key, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    _read_t0 = time.perf_counter()
    print(f"[STAGE] parsing CSV bytes START | bytes={len(body):,}", flush=True)
    _df_read = pd.read_csv(io.BytesIO(body))
    print(f"[STAGE] parsing CSV bytes DONE | rows={len(_df_read):,} | cols={len(_df_read.columns)} | {(time.perf_counter()-_read_t0):.1f}s", flush=True)
    return _df_read


def _read_s3_parquet(bucket, key, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def _extract_wape_metrics(wape_object, prefix):
    """
    Flatten the P50/P70 object returned by the existing two-step evaluator.

    The original quick_error_check may return a Series, dict, DataFrame,
    or scalar depending on the notebook helper implementation.
    """
    output = {}

    if wape_object is None:
        return output

    if isinstance(wape_object, pd.DataFrame):
        if len(wape_object) == 1:
            for key, value in wape_object.iloc[0].items():
                if np.isscalar(value):
                    output[f"{prefix}_{key}"] = value
        else:
            output[f"{prefix}_repr"] = wape_object.to_json(orient="records")
        return output

    if isinstance(wape_object, pd.Series):
        for key, value in wape_object.items():
            if np.isscalar(value):
                output[f"{prefix}_{key}"] = value
        return output

    if isinstance(wape_object, dict):
        for key, value in wape_object.items():
            if np.isscalar(value):
                output[f"{prefix}_{key}"] = value
            else:
                output[f"{prefix}_{key}"] = str(value)
        return output

    if np.isscalar(wape_object):
        output[prefix] = wape_object
    else:
        output[f"{prefix}_repr"] = str(wape_object)

    return output


def _safe_file_token(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def run_joint_h3_s3_rolling_scot_p50_p70(
    n_asins=5000,
    seed=42,
    max_snapshots=None,
    select_latest=False,
    snapshot_stride=1,
    start_date=None,
    end_date=None,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    beta_tail=0.5,
    patience=6,
    lambda_under=0.0,
    lambda_over=0.0,
    lambda_exposure=0.50,
    detach_exposure_for_demand=False,
    remove_extreme=True,
    extreme_q=0.99,
    remove_oos_dp=True,
    output_root="joint_true_rolling_h3_original_wape_exposure_diag",
    resume_existing=True,
    continue_on_error=True,
    bucket=ROLLING_S3_BUCKET,
    data_prefix=ROLLING_DATA_PREFIX,
    scot_prefix=ROLLING_SCOT_PREFIX,
):
    """
    Rolling Joint H3 evaluation.

    For every matched cut:
      1. Read one feature snapshot CSV.
      2. Read the SCOT parquet whose FCD is snapshot date + 1 day.
      3. Randomly sample 5000 ASINs from that snapshot.
      4. Keep the sampled-ASIN / SCOT-ASIN intersection.
      5. Train the unchanged Joint H3 V1.2 model.
      6. Evaluate only P50 and P70 with the exact two-step WAPE function.
      7. Save per-cut predictions, aligned rows, and one rolling summary.

    Model architecture and losses are unchanged from the non-rolling V1.2 file.
    """
    if horizon != 3:
        raise ValueError("This rolling experiment is H3 only.")

    missing_wape_helpers = [
        name
        for name in [
            "calculate_wape_using_lp_oos2",
            "quick_error_check",
        ]
        if name not in globals()
    ]
    if missing_wape_helpers:
        raise RuntimeError(
            "The rolling file intentionally does not implement WAPE. "
            "Load the same existing helper functions used by the two-stage "
            "model before running. Missing: "
            + ", ".join(missing_wape_helpers)
        )

    output_root = Path(output_root)
    per_cut_dir = output_root / "per_cut"
    aligned_dir = output_root / "aligned_scot"
    per_cut_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")

    pairs = list_joint_rolling_snapshot_pairs(
        bucket=bucket,
        data_prefix=data_prefix,
        scot_prefix=scot_prefix,
        s3_client=s3_client,
    )

    pairs = pairs[pairs["is_complete_pair"]].copy()

    if start_date is not None:
        pairs = pairs[
            pairs["data_cut"] >= pd.Timestamp(start_date).normalize()
        ]
    if end_date is not None:
        pairs = pairs[
            pairs["data_cut"] <= pd.Timestamp(end_date).normalize()
        ]

    pairs = pairs.sort_values("data_cut").iloc[::max(1, int(snapshot_stride))]

    if max_snapshots is not None:
        max_snapshots = int(max_snapshots)
        pairs = (
            pairs.tail(max_snapshots)
            if select_latest
            else pairs.head(max_snapshots)
        )

    pairs = pairs.reset_index(drop=True)

    print("\n" + "=" * 88)
    print("JOINT H3 TRUE ROLLING + ORIGINAL WAPE + EXPOSURE DIAGNOSTICS")
    print("=" * 88)
    print("Cuts selected:", len(pairs))
    print("Random ASIN sample per cut:", "ALL" if n_asins is None else n_asins)
    print("Seed:", seed)
    print("Output root:", output_root.resolve())
    print("Model unchanged from non-rolling V1.2: YES")
    print("Metrics retained: P50 and P70 only")
    print(
        "WAPE: calls original calculate_wape_using_lp_oos2 "
        "+ quick_error_check; no local WAPE formula"
    )

    summary_rows = []
    all_forecasts = []
    all_aligned = []
    all_exposure_overall = []
    all_exposure_by_horizon = []
    all_exposure_by_cut_horizon = []

    for run_index, row in pairs.iterrows():
        data_cut = pd.Timestamp(row["data_cut"])
        scot_fcd = pd.Timestamp(row["scot_fcd"])
        cut_token = _safe_file_token(data_cut)

        forecast_path = (
            per_cut_dir
            / f"joint_h3_forecast_cut_{cut_token}.csv"
        )
        aligned_path = (
            aligned_dir
            / f"joint_h3_scot_aligned_cut_{cut_token}.csv"
        )

        print("\n" + "#" * 100)
        print(
            f"ROLLING CUT {run_index + 1}/{len(pairs)} | "
            f"data_cut={data_cut.date()} | "
            f"SCOT_FCD={scot_fcd.date()}"
        )
        print("#" * 100)

        try:
            origin_raw = _read_s3_csv(
                bucket,
                row["data_key"],
                s3_client,
            )
            eval_raw = _read_s3_csv(
                bucket,
                row["eval_data_key"],
                s3_client,
            )
            scot_df = _read_s3_parquet(
                bucket,
                row["scot_key"],
                s3_client,
            )

            for frame in [origin_raw, eval_raw, scot_df]:
                frame["asin"] = frame["asin"].astype(str).str.strip()

            origin_raw["order_week"] = pd.to_datetime(
                origin_raw["order_week"], errors="coerce"
            )
            eval_raw["order_week"] = pd.to_datetime(
                eval_raw["order_week"], errors="coerce"
            )
            scot_df["order_week"] = pd.to_datetime(
                scot_df["order_week"], errors="coerce"
            )

            origin_asins = origin_raw["asin"].dropna().unique()
            if n_asins is None:
                sampled_origin_asins = origin_asins
            else:
                rng = np.random.default_rng(seed)
                sampled_origin_asins = rng.choice(
                    origin_asins,
                    size=min(n_asins, len(origin_asins)),
                    replace=False,
                )

            sampled_set = set(sampled_origin_asins)
            scot_set = set(scot_df["asin"].dropna().unique())
            eval_set = set(eval_raw["asin"].dropna().unique())
            rolling_asins = sorted(sampled_set & scot_set & eval_set)

            if not rolling_asins:
                raise RuntimeError(
                    "No ASINs remain after origin-sample, SCOT, and "
                    "evaluation-snapshot intersection."
                )

            # This is the missing rolling merge step:
            # use the later evaluation snapshot to supply the actual H1-H3
            # rows, but restrict it to ASINs sampled at the origin cut.
            data_raw1 = eval_raw[
                eval_raw["asin"].isin(rolling_asins)
            ].copy()

            expected_h1 = pd.Timestamp(row["scot_fcd"])
            expected_h2 = expected_h1 + pd.Timedelta(days=7)
            expected_h3 = expected_h1 + pd.Timedelta(days=14)

            print("=" * 96)
            print("ROLLING COHORT + TARGET-WEEK CONSTRUCTION")
            print("=" * 96)
            print("Origin snapshot:", pd.Timestamp(row["data_cut"]).date())
            print("Evaluation snapshot:", pd.Timestamp(row["eval_cut"]).date())
            print("SCOT FCD:", expected_h1.date())
            print(
                "Origin max observed week:",
                origin_raw["order_week"].max(),
            )
            print(
                "Evaluation max observed week:",
                eval_raw["order_week"].max(),
            )
            print(
                "Expected H1/H2/H3:",
                [expected_h1, expected_h2, expected_h3],
            )
            print("Origin sampled ASINs:", len(sampled_set))
            print("SCOT ASINs:", len(scot_set))
            print("Evaluation ASINs:", len(eval_set))
            print("Final rolling ASIN intersection:", len(rolling_asins))
            print(
                f"Evaluation rows used={len(data_raw1):,} | "
                f"ASINs={data_raw1['asin'].nunique():,}"
            )

            if (
                resume_existing
                and forecast_path.exists()
                and aligned_path.exists()
            ):
                print("Existing outputs found; loading instead of retraining.")
                forecast_df = pd.read_csv(
                    forecast_path,
                    parse_dates=["order_week"],
                )

                # Use the exact same evaluator as two-stage, even on resume.
                result_stub = {
                    "forecast_df": forecast_df,
                }
                real_scot = run_high_sparse_scot_alignment_wape(
                    result=result_stub,
                    scot_df=scot_df,
                    data_raw1=data_raw1,
                    asin_stats=None,
                    remove_oos_dp=remove_oos_dp,
                    source="lp",
                )
                aligned_df = real_scot[
                    "forecast_df_scot_real"
                ].copy()

                aligned_df["data_cut"] = data_cut
                aligned_df["scot_fcd"] = scot_fcd
                aligned_df["eval_cut"] = pd.Timestamp(row["eval_cut"])
                aligned_df["data_s3_key"] = row["data_key"]
                aligned_df["scot_s3_key"] = row["scot_key"]
                aligned_df.to_csv(aligned_path, index=False)

                exposure_diag = compute_exposure_hat_diagnostics(
                    forecast_df,
                    data_cut=data_cut,
                )
                print_exposure_hat_diagnostics(
                    exposure_diag,
                    title=(
                        "EXPOSURE HAT DIAGNOSTICS | "
                        f"CUT={data_cut.date()} | LOADED"
                    ),
                )

                exposure_diag["overall"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_overall_cut_{cut_token}.csv",
                    index=False,
                )
                exposure_diag["by_horizon"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_by_horizon_cut_{cut_token}.csv",
                    index=False,
                )
                exposure_diag["by_cut_horizon"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_by_cut_horizon_cut_{cut_token}.csv",
                    index=False,
                )

                real_scot["h_wape_p50"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p50_cut_{cut_token}.csv",
                    index=False,
                )
                real_scot["h_wape_p70"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p70_cut_{cut_token}.csv",
                    index=False,
                )
                real_scot["h_wape_p90"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p90_cut_{cut_token}.csv",
                    index=False,
                )

                metrics = {}
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p50_wape"),
                        "p50",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p70_wape"),
                        "p70",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p90_wape"),
                        "p90",
                    )
                )
                metrics["p50_penalty_diff"] = real_scot.get(
                    "p50_penalty_diff"
                )
                metrics["p70_penalty_diff"] = real_scot.get(
                    "p70_penalty_diff"
                )
                metrics["p90_penalty_diff"] = real_scot.get(
                    "p90_penalty_diff"
                )

                result = None
                status = "loaded_existing"

            else:
                result = run_joint_exposure_demand_h3_end2end(
                    data_raw1=data_raw1,
                    scot_df=scot_df,
                    n_asins=len(rolling_asins),
                    seed=seed,
                    epochs=epochs,
                    history=history,
                    horizon=horizon,
                    d_model=d_model,
                    d_z=d_z,
                    batch_size=batch_size,
                    M_eval=M_eval,
                    beta_tail=beta_tail,
                    patience=patience,
                    lambda_under=lambda_under,
                    lambda_over=lambda_over,
                    lambda_exposure=lambda_exposure,
                    detach_exposure_for_demand=(
                        detach_exposure_for_demand
                    ),
                    remove_extreme=remove_extreme,
                    extreme_q=extreme_q,
                    output_csv=str(forecast_path),
                    remove_oos_dp=remove_oos_dp,
                )

                if "real_scot_outputs" not in result:
                    raise RuntimeError(
                        "SCOT evaluation did not produce real_scot_outputs."
                    )

                real_scot = result["real_scot_outputs"]
                forecast_df = result["forecast_df"].copy()

                joint_weeks = sorted(
                    pd.to_datetime(
                        forecast_df["order_week"], errors="coerce"
                    ).dropna().unique()
                )
                scot_h3_weeks = sorted(
                    pd.to_datetime(
                        scot_df["order_week"], errors="coerce"
                    ).dropna().unique()
                )[:3]
                expected_weeks = [
                    expected_h1.to_datetime64(),
                    expected_h2.to_datetime64(),
                    expected_h3.to_datetime64(),
                ]

                print("\n" + "=" * 96)
                print("ROLLING TARGET WEEK ASSERTION")
                print("=" * 96)
                print("Joint unique weeks:", joint_weeks)
                print("SCOT first 3 weeks:", scot_h3_weeks)
                print("Expected H1-H3:", expected_weeks)

                if {pd.Timestamp(w) for w in joint_weeks} != {pd.Timestamp(w) for w in expected_weeks}:
                    raise RuntimeError(
                        "Joint validation weeks do not equal the rolling "
                        "SCOT H1-H3 weeks. The evaluation snapshot pairing "
                        "or dataset split is incorrect."
                    )

                aligned_df = real_scot[
                    "forecast_df_scot_real"
                ].copy()

                forecast_df["data_cut"] = data_cut
                forecast_df["scot_fcd"] = scot_fcd
                forecast_df["eval_cut"] = pd.Timestamp(row["eval_cut"])
                forecast_df["data_s3_key"] = row["data_key"]
                forecast_df["scot_s3_key"] = row["scot_key"]

                aligned_df["data_cut"] = data_cut
                aligned_df["scot_fcd"] = scot_fcd
                aligned_df["data_s3_key"] = row["data_key"]
                aligned_df["scot_s3_key"] = row["scot_key"]

                forecast_df.to_csv(forecast_path, index=False)
                aligned_df.to_csv(aligned_path, index=False)

                # Exposure hat diagnostics for this rolling cut.
                exposure_diag = compute_exposure_hat_diagnostics(
                    forecast_df,
                    data_cut=data_cut,
                )
                print_exposure_hat_diagnostics(
                    exposure_diag,
                    title=(
                        "EXPOSURE HAT DIAGNOSTICS | "
                        f"CUT={data_cut.date()}"
                    ),
                )

                exposure_diag["overall"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_overall_cut_{cut_token}.csv",
                    index=False,
                )
                exposure_diag["by_horizon"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_by_horizon_cut_{cut_token}.csv",
                    index=False,
                )
                exposure_diag["by_cut_horizon"].to_csv(
                    per_cut_dir
                    / f"exposure_hat_by_cut_horizon_cut_{cut_token}.csv",
                    index=False,
                )

                real_scot["h_wape_p50"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p50_cut_{cut_token}.csv",
                    index=False,
                )
                real_scot["h_wape_p70"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p70_cut_{cut_token}.csv",
                    index=False,
                )
                real_scot["h_wape_p90"].to_csv(
                    per_cut_dir / f"demand_wape_by_horizon_p90_cut_{cut_token}.csv",
                    index=False,
                )

                # Keep the original two-step P50/P70 outputs in summary.
                metrics = {}
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p50_wape"),
                        "p50",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p70_wape"),
                        "p70",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p90_wape"),
                        "p90",
                    )
                )

                metrics["p50_penalty_diff"] = real_scot.get(
                    "p50_penalty_diff"
                )
                metrics["p70_penalty_diff"] = real_scot.get(
                    "p70_penalty_diff"
                )
                metrics["p90_penalty_diff"] = real_scot.get(
                    "p90_penalty_diff"
                )

                status = "trained"

            summary = {
                "data_cut": data_cut,
                "scot_fcd": scot_fcd,
                "status": status,
                "data_s3_key": row["data_key"],
                "scot_s3_key": row["scot_key"],
                "forecast_rows": len(forecast_df),
                "forecast_asins": forecast_df["asin"].nunique(),
                "aligned_rows": len(aligned_df),
                "aligned_asins": aligned_df["asin"].nunique(),
                "aligned_week_min": aligned_df["order_week"].min(),
                "aligned_week_max": aligned_df["order_week"].max(),
                "raw_asin_intersection": (
                    real_scot.get("alignment_debug", {}).get("raw_asin_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "canonical_asin_intersection": (
                    real_scot.get("alignment_debug", {}).get("canonical_asin_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "raw_week_intersection": (
                    real_scot.get("alignment_debug", {}).get("raw_week_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "sunday_week_intersection": (
                    real_scot.get("alignment_debug", {}).get("sunday_week_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                **{
                    f"exposure_{row['channel']}_{metric}": row[metric]
                    for _, row in exposure_diag["overall"].iterrows()
                    for metric in [
                        "wape",
                        "pred_true_ratio",
                        "corr",
                        "overbias",
                        "underbias",
                        "active_auc",
                    ]
                },
                **metrics,
                "error": "",
            }

            summary_rows.append(summary)
            all_forecasts.append(forecast_df)
            all_aligned.append(aligned_df)

            exposure_overall_cut = exposure_diag["overall"].copy()
            exposure_overall_cut["data_cut"] = data_cut
            all_exposure_overall.append(exposure_overall_cut)

            exposure_by_h_cut = exposure_diag["by_horizon"].copy()
            exposure_by_h_cut["data_cut"] = data_cut
            all_exposure_by_horizon.append(exposure_by_h_cut)

            if not exposure_diag["by_cut_horizon"].empty:
                all_exposure_by_cut_horizon.append(
                    exposure_diag["by_cut_horizon"].copy()
                )

            pd.DataFrame(summary_rows).to_csv(
                output_root / "joint_rolling_p50_p70_summary.csv",
                index=False,
            )

            print(
                f"Saved cut outputs | forecast={forecast_path} | "
                f"aligned={aligned_path}"
            )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"FAILED data_cut={data_cut.date()} | "
                f"{error_message}"
            )

            summary_rows.append({
                "data_cut": data_cut,
                "scot_fcd": scot_fcd,
                "status": "failed",
                "data_s3_key": row["data_key"],
                "scot_s3_key": row["scot_key"],
                "forecast_rows": 0,
                "forecast_asins": 0,
                "aligned_rows": 0,
                "aligned_asins": 0,
                "error": error_message,
            })

            pd.DataFrame(summary_rows).to_csv(
                output_root / "joint_rolling_p50_p70_summary.csv",
                index=False,
            )

            if not continue_on_error:
                raise

    summary_df = pd.DataFrame(summary_rows)

    if all_forecasts:
        full_forecast_df = pd.concat(
            all_forecasts,
            ignore_index=True,
        )
        full_forecast_df.to_csv(
            output_root / "joint_h3_forecast_full.csv",
            index=False,
        )
    else:
        full_forecast_df = pd.DataFrame()

    if all_aligned:
        full_aligned_df = pd.concat(
            all_aligned,
            ignore_index=True,
        )
        full_aligned_df.to_csv(
            output_root / "joint_h3_scot_aligned_full.csv",
            index=False,
        )
    else:
        full_aligned_df = pd.DataFrame()

    summary_df.to_csv(
        output_root / "joint_rolling_p50_p70_summary.csv",
        index=False,
    )

    if all_exposure_overall:
        exposure_overall_full = pd.concat(
            all_exposure_overall,
            ignore_index=True,
        )
        exposure_overall_full.to_csv(
            output_root / "exposure_hat_overall_by_cut.csv",
            index=False,
        )
    else:
        exposure_overall_full = pd.DataFrame()

    if all_exposure_by_horizon:
        exposure_by_horizon_full = pd.concat(
            all_exposure_by_horizon,
            ignore_index=True,
        )
        exposure_by_horizon_full.to_csv(
            output_root / "exposure_hat_by_cut_and_horizon.csv",
            index=False,
        )
    else:
        exposure_by_horizon_full = pd.DataFrame()

    if all_exposure_by_cut_horizon:
        exposure_by_cut_horizon_full = pd.concat(
            all_exposure_by_cut_horizon,
            ignore_index=True,
        )
    else:
        exposure_by_cut_horizon_full = pd.DataFrame()

    # Diagnostics across all rolling cuts combined.
    if not full_forecast_df.empty:
        combined_exposure_diag = compute_exposure_hat_diagnostics(
            full_forecast_df
        )
        print_exposure_hat_diagnostics(
            combined_exposure_diag,
            title="EXPOSURE HAT DIAGNOSTICS | ALL ROLLING CUTS",
        )
        combined_exposure_diag["overall"].to_csv(
            output_root / "exposure_hat_full_overall.csv",
            index=False,
        )
        combined_exposure_diag["by_horizon"].to_csv(
            output_root / "exposure_hat_full_by_horizon.csv",
            index=False,
        )
        combined_exposure_diag["by_cut_horizon"].to_csv(
            output_root / "exposure_hat_full_by_cut_horizon.csv",
            index=False,
        )
    else:
        combined_exposure_diag = {
            "overall": pd.DataFrame(),
            "by_horizon": pd.DataFrame(),
            "by_cut_horizon": pd.DataFrame(),
        }

    print("\n" + "=" * 88)
    print("ROLLING RUN COMPLETE")
    print("=" * 88)
    print("Successful cuts:", int((summary_df["status"] != "failed").sum()))
    print("Failed cuts:", int((summary_df["status"] == "failed").sum()))
    print(
        "Summary:",
        output_root / "joint_rolling_p50_p70_summary.csv",
    )
    print(
        "Full forecast:",
        output_root / "joint_h3_forecast_full.csv",
    )
    print(
        "Full aligned SCOT:",
        output_root / "joint_h3_scot_aligned_full.csv",
    )

    return {
        "summary": summary_df,
        "forecast_df": full_forecast_df,
        "aligned_scot_df": full_aligned_df,
        "exposure_hat_overall_by_cut": exposure_overall_full,
        "exposure_hat_by_cut_and_horizon": exposure_by_horizon_full,
        "exposure_hat_full_diagnostics": combined_exposure_diag,
        "snapshot_pairs": pairs,
        "output_root": str(output_root),
    }


# ============================================================================
# RECOMMENDED USAGE: FIRST TEST TWO ROLLING CUTS
# ============================================================================
#
# rolling_joint_h3 = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=5000,
#     seed=42,
#     max_snapshots=2,
#     select_latest=False,
#     epochs=3,
#     patience=2,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_true_rolling_h3_original_wape_exposure_diag",
#     resume_existing=True,
#     continue_on_error=False,
# )
#
# Full rolling run after the test succeeds:
#
# rolling_joint_h3 = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=5000,
#     seed=42,
#     max_snapshots=None,
#     select_latest=False,
#     epochs=60,
#     patience=6,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_true_rolling_h3_original_wape_exposure_diag",
#     resume_existing=True,
#     continue_on_error=True,
# )
#
# All-ASIN run (no sampling -- n_asins=None uses every ASIN in each cut).
# Uses its own output_root, separate from the n_asins=5000 runs above, so
# resume_existing=True can never load a cached per-cut result that was
# actually trained on the 5000-ASIN sample -- each output_root is its own
# independent cache namespace, keyed only by data_cut, not by n_asins:
#
# rolling_joint_h3_all_asins = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=None,
#     seed=42,
#     max_snapshots=None,
#     select_latest=False,
#     epochs=60,
#     patience=6,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_true_rolling_h3_all_asins",
#     resume_existing=True,
#     continue_on_error=True,
# )
