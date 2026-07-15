
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


CHANNELS = [
    ("total", "true_future_total_dph", "pred_total_dph_hat"),
    ("buy_box", "true_future_buy_box_dph", "pred_buy_box_dph_hat"),
    ("in_stock", "true_future_instock", "pred_instock_dph_hat"),
]


def _safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_auc(y_true, score):
    if roc_auc_score is None:
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


def _normalize_forecast_df(data_or_result):
    """
    Accept:
      1. forecast DataFrame directly
      2. model result dict containing result["forecast_df"]
      3. rolling result dict containing result["forecast_df"]
      4. CSV path
    """
    if isinstance(data_or_result, pd.DataFrame):
        df = data_or_result.copy()
    elif isinstance(data_or_result, dict):
        if "forecast_df" not in data_or_result:
            raise KeyError('Input dict must contain "forecast_df".')
        df = data_or_result["forecast_df"].copy()
    elif isinstance(data_or_result, (str, os.PathLike)):
        df = pd.read_csv(data_or_result)
    else:
        raise TypeError(
            "data_or_result must be a DataFrame, result dict, or CSV path."
        )

    df.columns = [str(c).strip() for c in df.columns]

    required = ["asin", "fcst_week_index"]
    for _, true_col, pred_col in CHANNELS:
        required.extend([true_col, pred_col])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["asin"] = df["asin"].astype(str)
    df["fcst_week_index"] = pd.to_numeric(
        df["fcst_week_index"], errors="coerce"
    )

    if "order_week" in df.columns:
        df["order_week"] = pd.to_datetime(
            df["order_week"], errors="coerce"
        )

    if "data_cut" in df.columns:
        df["data_cut"] = pd.to_datetime(
            df["data_cut"], errors="coerce"
        )

    for _, true_col, pred_col in CHANNELS:
        df[true_col] = pd.to_numeric(
            df[true_col], errors="coerce"
        )
        df[pred_col] = pd.to_numeric(
            df[pred_col], errors="coerce"
        )

    return df


def _metric_row(sub, channel, true_col, pred_col):
    valid = sub[[true_col, pred_col]].dropna()

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

    y_true = valid[true_col].to_numpy(dtype=float)
    y_pred = valid[pred_col].to_numpy(dtype=float)

    error = y_pred - y_true
    denom = np.abs(y_true).sum() + 1e-8

    return {
        "channel": channel,
        "n_rows": len(valid),
        "n_asins": sub.loc[valid.index, "asin"].nunique(),
        "true_mean": y_true.mean(),
        "pred_mean": y_pred.mean(),
        "pred_true_ratio": y_pred.sum() / (y_true.sum() + 1e-8),
        "wape": np.abs(error).sum() / denom,
        "mae": np.abs(error).mean(),
        "corr": _safe_corr(y_true, y_pred),
        "overbias": np.maximum(error, 0.0).sum() / denom,
        "underbias": np.maximum(-error, 0.0).sum() / denom,
        "true_zero_rate": np.mean(y_true <= 0),
        "pred_zero_rate": np.mean(y_pred <= 0),
        "active_auc": _safe_auc(y_true, y_pred),
    }


def exposure_hat_diagnostics_by_horizon(
    data_or_result,
    output_dir="exposure_hat_diagnostics",
    file_prefix="joint_exposure",
    print_results=True,
):
    """
    Diagnose exposure_hat performance by H1/H2/H3 for:
      - total DPH
      - buy-box DPH
      - in-stock DPH

    Outputs:
      1. overall metrics by channel
      2. metrics by horizon and channel
      3. metrics by rolling cut, horizon, and channel when data_cut exists

    Expected forecast columns:
      asin
      fcst_week_index
      true_future_total_dph
      true_future_buy_box_dph
      true_future_instock
      pred_total_dph_hat
      pred_buy_box_dph_hat
      pred_instock_dph_hat
    """
    df = _normalize_forecast_df(data_or_result)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    horizon_rows = []
    cut_horizon_rows = []

    # Overall.
    for channel, true_col, pred_col in CHANNELS:
        overall_rows.append(
            _metric_row(df, channel, true_col, pred_col)
        )

    overall_df = pd.DataFrame(overall_rows)

    # By H.
    horizons = sorted(
        int(h)
        for h in df["fcst_week_index"].dropna().unique()
    )

    for horizon in horizons:
        sub_h = df[df["fcst_week_index"] == horizon]

        for channel, true_col, pred_col in CHANNELS:
            row = _metric_row(
                sub_h, channel, true_col, pred_col
            )
            row["horizon"] = horizon
            horizon_rows.append(row)

    by_horizon_df = pd.DataFrame(horizon_rows)

    # By rolling cut + H, when available.
    if "data_cut" in df.columns:
        for data_cut, sub_cut in df.groupby("data_cut", dropna=False):
            for horizon in horizons:
                sub_h = sub_cut[
                    sub_cut["fcst_week_index"] == horizon
                ]

                for channel, true_col, pred_col in CHANNELS:
                    row = _metric_row(
                        sub_h, channel, true_col, pred_col
                    )
                    row["data_cut"] = data_cut
                    row["horizon"] = horizon
                    cut_horizon_rows.append(row)

    by_cut_horizon_df = pd.DataFrame(cut_horizon_rows)

    overall_path = output_dir / f"{file_prefix}_overall.csv"
    horizon_path = output_dir / f"{file_prefix}_by_horizon.csv"
    cut_horizon_path = (
        output_dir / f"{file_prefix}_by_cut_horizon.csv"
    )

    overall_df.to_csv(overall_path, index=False)
    by_horizon_df.to_csv(horizon_path, index=False)

    if not by_cut_horizon_df.empty:
        by_cut_horizon_df.to_csv(
            cut_horizon_path, index=False
        )

    if print_results:
        print("=" * 100)
        print("EXPOSURE HAT DIAGNOSTICS — OVERALL")
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

        print()
        print("=" * 100)
        print("EXPOSURE HAT DIAGNOSTICS — BY HORIZON")
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

        print()
        print("Saved:")
        print(" ", overall_path)
        print(" ", horizon_path)
        if not by_cut_horizon_df.empty:
            print(" ", cut_horizon_path)

    return {
        "overall": overall_df,
        "by_horizon": by_horizon_df,
        "by_cut_horizon": by_cut_horizon_df,
        "forecast_df": df,
        "output_dir": str(output_dir),
    }


# ============================================================================
# USAGE 1: NON-ROLLING JOINT RESULT
# ============================================================================
#
# exposure_diag = exposure_hat_diagnostics_by_horizon(
#     joint_result_h3,
#     output_dir="joint_h3_exposure_diagnostics",
#     file_prefix="joint_h3_nonrolling",
# )
#
# ============================================================================
# USAGE 2: ONE ROLLING CUT FORECAST CSV
# ============================================================================
#
# exposure_diag = exposure_hat_diagnostics_by_horizon(
#     "joint_rolling_h3_scot/per_cut/joint_h3_forecast_cut_2025-10-04.csv",
#     output_dir="joint_h3_exposure_diagnostics",
#     file_prefix="cut_2025-10-04",
# )
#
# ============================================================================
# USAGE 3: FULL ROLLING FORECAST CSV
# ============================================================================
#
# exposure_diag = exposure_hat_diagnostics_by_horizon(
#     "joint_rolling_h3_scot/joint_h3_forecast_full.csv",
#     output_dir="joint_h3_exposure_diagnostics",
#     file_prefix="joint_h3_rolling_full",
# )
#
# Useful views:
#
# print(exposure_diag["by_horizon"])
#
# h3_only = exposure_diag["by_horizon"].query("horizon == 3")
# print(h3_only)
#
# instock = exposure_diag["by_horizon"].query(
#     "channel == 'in_stock'"
# )
# print(instock)
