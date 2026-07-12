import numpy as np
import pandas as pd


def safe_wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return np.nan if denom <= 1e-8 else np.abs(y_true - y_pred).sum() / denom


def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def exposure_lag_analysis_all_asins(
    data_raw,
    max_lag=6,
    output_csv="exposure_lag_analysis_all_asins.csv",
):
    required_cols = [
        "asin",
        "order_week",
        "total_dph",
        "buy_box_dph",
        "in_stock_dph",
    ]

    missing = [c for c in required_cols if c not in data_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = data_raw[required_cols].copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"], errors="coerce")

    channels = ["total_dph", "buy_box_dph", "in_stock_dph"]
    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = (
        df.dropna(subset=["order_week"])
        .sort_values(["asin", "order_week"])
        .reset_index(drop=True)
    )

    print("=" * 80)
    print("EXPOSURE LAG ANALYSIS")
    print("=" * 80)
    print(f"ASINs: {df['asin'].nunique():,}")
    print(f"Rows : {len(df):,}")
    print(f"Lags : 1 to {max_lag}")
    print()

    rows = []

    for channel in channels:
        grouped = df.groupby("asin", sort=False)[channel]

        for lag in range(1, max_lag + 1):
            lag_values = grouped.shift(lag)
            valid = lag_values.notna()

            if not valid.any():
                continue

            y_true = df.loc[valid, channel].to_numpy(dtype=float)
            y_pred = lag_values.loc[valid].to_numpy(dtype=float)

            error = y_pred - y_true
            over_error = np.maximum(error, 0.0)
            under_error = np.maximum(-error, 0.0)
            true_sum = np.abs(y_true).sum()

            rows.append({
                "channel": channel,
                "lag": lag,
                "n_rows": len(y_true),
                "true_mean": y_true.mean(),
                "pred_mean": y_pred.mean(),
                "pred_true_ratio": y_pred.sum() / (y_true.sum() + 1e-8),
                "wape": safe_wape(y_true, y_pred),
                "mae": np.abs(error).mean(),
                "corr": safe_corr(y_true, y_pred),
                "overbias": over_error.sum() / (true_sum + 1e-8),
                "underbias": under_error.sum() / (true_sum + 1e-8),
                "zero_true_rate": np.mean(y_true <= 0),
                "zero_pred_rate": np.mean(y_pred <= 0),
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise RuntimeError("No valid lag evaluation rows were created.")

    result_df = result_df.sort_values(["channel", "lag"]).reset_index(drop=True)

    print("=" * 80)
    print("OVERALL LAG RESULTS")
    print("=" * 80)

    display_cols = [
        "channel", "lag", "wape", "corr", "pred_true_ratio",
        "overbias", "underbias", "true_mean", "pred_mean", "n_rows",
    ]
    print(result_df[display_cols].round(4).to_string(index=False))

    print()
    print("=" * 80)
    print("BEST LAG BY CHANNEL")
    print("=" * 80)

    best_rows = []
    for channel in channels:
        sub = result_df[result_df["channel"] == channel]
        if sub.empty:
            continue
        best = sub.loc[sub["wape"].idxmin()]
        best_rows.append(best)
        print(
            f"{channel}: best lag={int(best['lag'])} | "
            f"WAPE={best['wape']:.4f} | "
            f"Corr={best['corr']:.4f} | "
            f"Ratio={best['pred_true_ratio']:.4f}"
        )

    best_df = pd.DataFrame(best_rows)
    result_df.to_csv(output_csv, index=False)
    print()
    print(f"Saved lag analysis to: {output_csv}")

    return {
        "lag_results": result_df,
        "best_lag_by_channel": best_df,
    }


# Usage:
# lag_result = exposure_lag_analysis_all_asins(
#     data_raw=data_raw1,
#     max_lag=6,
#     output_csv="exposure_lag_analysis_all_asins.csv",
# )
# print(lag_result["lag_results"])
# print(lag_result["best_lag_by_channel"])
# print(lag_result["lag_results"].query("lag == 3"))
