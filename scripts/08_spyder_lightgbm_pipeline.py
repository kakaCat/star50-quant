#!/usr/bin/env python3
"""
Spyder runnable LightGBM fixed-window pipeline.

Put the two parquet files in the current Spyder working directory, or keep them
under data/raw in this project. Then run this file directly in Spyder.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def sanitize_import_paths():
    """Remove unreadable sys.path entries before importing third-party packages."""
    clean_paths = []
    removed_paths = []
    for path in sys.path:
        if path == "":
            clean_paths.append(path)
            continue
        try:
            path_obj = Path(path).expanduser()
            if not path_obj.exists():
                removed_paths.append(path)
                continue
            if path_obj.is_dir():
                # Some macOS-protected folders exist but fail only when importlib
                # tries to scan them. Test actual directory traversal up front.
                with os.scandir(path_obj):
                    pass
            elif not os.access(path_obj, os.R_OK):
                removed_paths.append(path)
                continue
            clean_paths.append(path)
        except OSError:
            removed_paths.append(path)
    sys.path[:] = clean_paths
    sys.path_importer_cache.clear()
    if removed_paths:
        print("Removed unreadable import paths:")
        for path in removed_paths:
            print(f"  {path}")


sanitize_import_paths()

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow


# Required path format
DATA_CACHE_PATH = "/Users/elize/Desktop/量化/star50_daily_hfq_data_6yrs.parquet"
INDEX_CACHE_PATH = "/Users/elize/Desktop/量化/star50_index_daily_6yrs.parquet"

# Optimized parameters from focused grid search
LGBM_PARAMS = {
    "n_estimators": 180,
    "learning_rate": 0.045,
    "num_leaves": 62,
    "min_child_samples": 120,
    "subsample": 0.708884,
    "colsample_bytree": 0.913066,
    "reg_alpha": 0.000601,
    "reg_lambda": 0.483284,
}

LABEL_COLUMN = "label_5d"
TRAIN_WINDOW_DAYS = 756
PREDICT_WINDOW_DAYS = 20
REBALANCE_DAYS = 5
TOP_QUANTILE = 0.15
TRANSACTION_COST = 0.0

OUTPUT_DIR = Path("outputs/spyder_lightgbm_pipeline")

STOCK_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "mom_20",
    "mom_60",
    "volatility_20",
    "volatility_60",
    "amplitude",
    "intraday_ret",
    "amount_chg_5",
    "vol_chg_5",
    "amount_ma20",
    "vol_ma20",
    "liquidity_proxy",
    "drawdown_20",
    "beta_to_index_60",
    "corr_to_index_60",
]

MARKET_FEATURE_COLUMNS = [
    "index_ret_1d",
    "index_volatility_20",
    "index_drawdown_20",
]

FEATURE_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
LABEL_COLUMNS = ["label_1d", "label_5d"]


def patch_pyarrow_extension_unregister():
    """Make pandas 3.0 + pyarrow 24 tolerate extension registry mismatches."""
    original_register = pyarrow.register_extension_type
    original_unregister = pyarrow.unregister_extension_type

    def safe_register(extension_type):
        try:
            return original_register(extension_type)
        except pyarrow.ArrowKeyError as exc:
            if "already defined" in str(exc):
                return None
            raise

    def safe_unregister(name):
        try:
            return original_unregister(name)
        except pyarrow.ArrowKeyError:
            if name == "arrow.py_extension_type":
                return None
            raise

    pyarrow.register_extension_type = safe_register
    pyarrow.unregister_extension_type = safe_unregister


patch_pyarrow_extension_unregister()


def resolve_input_path(file_name):
    """Find parquet in current directory first, then project data/raw."""
    candidates = [
        Path(file_name),
        Path("data/raw") / file_name,
        Path(__file__).resolve().parents[1] / "data" / "raw" / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {file_name}. Tried: {candidates}")


def load_raw_data():
    stock_path = resolve_input_path(DATA_CACHE_PATH)
    index_path = resolve_input_path(INDEX_CACHE_PATH)
    print(f"Stock data: {stock_path}")
    print(f"Index data: {index_path}")

    stock_df = pd.read_parquet(stock_path)
    index_df = pd.read_parquet(index_path)
    stock_df = stock_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    index_df = index_df.sort_values("trade_date").reset_index(drop=True)
    return stock_df, index_df


def add_index_features(index_df):
    df = index_df[["trade_date", "close"]].copy()
    df = df.rename(columns={"close": "index_close"})
    df["index_ret_1d"] = df["index_close"].pct_change()
    df["index_ret_5d"] = df["index_close"].pct_change(5)
    df["index_future_ret_1d"] = df["index_close"].shift(-1) / df["index_close"] - 1
    df["index_future_ret_5d"] = df["index_close"].shift(-5) / df["index_close"] - 1
    df["index_volatility_20"] = df["index_ret_1d"].rolling(20).std()
    df["index_drawdown_20"] = df["index_close"] / df["index_close"].rolling(20).max() - 1
    return df


def add_stock_features(stock_df):
    df = stock_df.copy()
    grouped = df.groupby("ts_code", group_keys=False)

    df["ret_1d"] = grouped["hfq_close"].pct_change()
    df["ret_5d"] = grouped["hfq_close"].pct_change(5)
    df["mom_20"] = grouped["hfq_close"].pct_change(20)
    df["mom_60"] = grouped["hfq_close"].pct_change(60)
    df["future_ret_1d"] = grouped["hfq_close"].shift(-1) / df["hfq_close"] - 1
    df["future_ret_5d"] = grouped["hfq_close"].shift(-5) / df["hfq_close"] - 1
    df["volatility_20"] = grouped["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    df["volatility_60"] = grouped["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)

    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"]
    df["intraday_ret"] = df["close"] / df["open"] - 1
    df["amount_chg_5"] = grouped["amount"].pct_change(5)
    df["vol_chg_5"] = grouped["vol"].pct_change(5)
    df["amount_ma20"] = grouped["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    df["vol_ma20"] = grouped["vol"].rolling(20).mean().reset_index(level=0, drop=True)
    df["liquidity_proxy"] = df["amount"] / df["amount_ma20"] - 1
    df["drawdown_20"] = df["hfq_close"] / grouped["hfq_close"].rolling(20).max().reset_index(level=0, drop=True) - 1
    return df


def add_relative_risk_features(df):
    result = df.copy()
    grouped = result.groupby("ts_code", group_keys=False)

    rolling_cov = grouped[["ret_1d", "index_ret_1d"]].rolling(60).cov().reset_index()
    cov_ret_index = rolling_cov[
        rolling_cov["level_2"].eq("ret_1d")
    ][["level_1", "index_ret_1d"]].set_index("level_1")["index_ret_1d"]
    index_var = grouped["index_ret_1d"].rolling(60).var().reset_index(level=0, drop=True)

    result["beta_to_index_60"] = cov_ret_index.reindex(result.index).to_numpy() / index_var.to_numpy()
    result["corr_to_index_60"] = (
        grouped[["ret_1d", "index_ret_1d"]]
        .rolling(60)
        .corr()
        .reset_index()
        .query("level_2 == 'ret_1d'")
        .set_index("level_1")["index_ret_1d"]
        .reindex(result.index)
        .to_numpy()
    )
    return result


def build_feature_dataset(stock_df, index_df):
    index_features = add_index_features(index_df)
    stock_features = add_stock_features(stock_df)
    df = stock_features.merge(index_features, on="trade_date", how="left", validate="many_to_one")
    df["label_1d"] = df["future_ret_1d"] - df["index_future_ret_1d"]
    df["label_5d"] = df["future_ret_5d"] - df["index_future_ret_5d"]
    df = add_relative_risk_features(df)
    keep_columns = ["ts_code", "trade_date", "hfq_close", "vol", "amount", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    return df[keep_columns].replace([np.inf, -np.inf], np.nan)


def mad_winsorize(series, n_mad=3.0):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if pd.isna(median) or pd.isna(mad) or mad == 0:
        return series
    scale = 1.4826 * mad
    return series.clip(lower=median - n_mad * scale, upper=median + n_mad * scale)


def zscore(series):
    mean = series.mean(skipna=True)
    std = series.std(skipna=True)
    if pd.isna(mean) or pd.isna(std):
        return series
    if std < 1e-12:
        return series * 0
    return (series - mean) / std


def preprocess_features(raw_df):
    df = raw_df.sort_values(["ts_code", "trade_date"]).copy()
    df[FEATURE_COLUMNS] = df.groupby("ts_code")[FEATURE_COLUMNS].ffill()
    df[FEATURE_COLUMNS] = df.groupby("trade_date")[FEATURE_COLUMNS].transform(
        lambda col: col.fillna(col.median(skipna=True))
    )
    df = df.dropna(subset=FEATURE_COLUMNS + LABEL_COLUMNS).copy()

    df = df.sort_values(["trade_date", "ts_code"]).copy()
    for column in STOCK_FEATURE_COLUMNS:
        df[column] = df.groupby("trade_date")[column].transform(mad_winsorize)
        df[column] = df.groupby("trade_date")[column].transform(zscore)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURE_COLUMNS + LABEL_COLUMNS).reset_index(drop=True)


def load_forward_returns(stock_df, index_df):
    stock_df = stock_df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = stock_df.groupby("ts_code", group_keys=False)
    stock_df["stock_ret_next_1d"] = grouped["hfq_close"].shift(-1) / stock_df["hfq_close"] - 1
    stock_df["realized_date"] = grouped["trade_date"].shift(-1)

    index_df = index_df.sort_values("trade_date").copy()
    index_df["index_ret_next_1d"] = index_df["close"].shift(-1) / index_df["close"] - 1

    returns = stock_df[["ts_code", "trade_date", "realized_date", "stock_ret_next_1d"]].merge(
        index_df[["trade_date", "index_ret_next_1d"]],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    return returns.dropna(subset=["realized_date", "stock_ret_next_1d", "index_ret_next_1d"])


def train_one_window(train_df):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "force_row_wise": True,
        "verbosity": -1,
        "seed": 42,
        "num_threads": -1,
        "learning_rate": LGBM_PARAMS["learning_rate"],
        "num_leaves": LGBM_PARAMS["num_leaves"],
        "min_child_samples": LGBM_PARAMS["min_child_samples"],
        "subsample": LGBM_PARAMS["subsample"],
        "colsample_bytree": LGBM_PARAMS["colsample_bytree"],
        "reg_alpha": LGBM_PARAMS["reg_alpha"],
        "reg_lambda": LGBM_PARAMS["reg_lambda"],
    }
    train_set = lgb.Dataset(
        train_df[FEATURE_COLUMNS],
        label=train_df[LABEL_COLUMN],
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=LGBM_PARAMS["n_estimators"],
    )
    return model


def rolling_fixed_window_predict(df):
    all_dates = np.array(sorted(df["trade_date"].unique()))
    prediction_parts = []
    window_id = 0
    start_idx = TRAIN_WINDOW_DAYS

    while start_idx < len(all_dates):
        train_dates = all_dates[start_idx - TRAIN_WINDOW_DAYS:start_idx]
        predict_dates = all_dates[start_idx:start_idx + PREDICT_WINDOW_DAYS]
        train_df = df[df["trade_date"].isin(train_dates)]
        predict_df = df[df["trade_date"].isin(predict_dates)].copy()
        if train_df.empty or predict_df.empty:
            break

        model = train_one_window(train_df)
        predict_df["pred_alpha"] = model.predict(predict_df[FEATURE_COLUMNS])
        predict_df["window_id"] = window_id
        prediction_parts.append(
            predict_df[["window_id", "ts_code", "trade_date", LABEL_COLUMN, "pred_alpha"]]
        )
        print(
            f"window={window_id:02d}, train={train_dates[0].date()}->{train_dates[-1].date()}, "
            f"predict={predict_dates[0].date()}->{predict_dates[-1].date()}, rows={len(predict_df)}"
        )
        window_id += 1
        start_idx += PREDICT_WINDOW_DAYS

    return pd.concat(prediction_parts, ignore_index=True)


def build_daily_strategy_returns(pred_df, forward_returns):
    pred_df = pred_df.sort_values(["trade_date", "pred_alpha"]).copy()
    dates = np.array(sorted(pred_df["trade_date"].unique()))
    daily_rows = []
    previous_holdings = set()

    for i in range(0, len(dates), REBALANCE_DAYS):
        rebalance_date = dates[i]
        holding_signal_dates = dates[i:i + REBALANCE_DAYS]
        cross_section = pred_df[pred_df["trade_date"].eq(rebalance_date)].copy()
        cutoff = cross_section["pred_alpha"].quantile(1 - TOP_QUANTILE)
        holdings = set(cross_section.loc[cross_section["pred_alpha"] >= cutoff, "ts_code"])
        if not holdings:
            continue

        turnover = 1.0
        if previous_holdings:
            overlap = len(holdings.intersection(previous_holdings)) / len(holdings)
            turnover = 1 - overlap
        cost = TRANSACTION_COST * turnover
        previous_holdings = holdings

        period_returns = forward_returns[
            forward_returns["trade_date"].isin(holding_signal_dates)
            & forward_returns["ts_code"].isin(holdings)
        ]
        for j, (signal_date, day_df) in enumerate(period_returns.groupby("trade_date", sort=True)):
            portfolio_ret = day_df["stock_ret_next_1d"].mean()
            if j == 0:
                portfolio_ret -= cost
            daily_rows.append(
                {
                    "signal_date": signal_date,
                    "realized_date": day_df["realized_date"].iloc[0],
                    "portfolio_ret": portfolio_ret,
                    "benchmark_ret": day_df["index_ret_next_1d"].iloc[0],
                    "holding_count": len(holdings),
                    "turnover": turnover if j == 0 else 0.0,
                    "transaction_cost": cost if j == 0 else 0.0,
                }
            )

    daily_df = pd.DataFrame(daily_rows).sort_values("realized_date").reset_index(drop=True)
    daily_df["excess_ret"] = daily_df["portfolio_ret"] - daily_df["benchmark_ret"]
    daily_df["nav"] = (1 + daily_df["portfolio_ret"]).cumprod()
    daily_df["benchmark_nav"] = (1 + daily_df["benchmark_ret"]).cumprod()
    daily_df["excess_nav"] = daily_df["nav"] / daily_df["benchmark_nav"]
    return daily_df


def max_drawdown(nav):
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    end_idx = int(drawdown.idxmin())
    start_idx = int(nav.loc[:end_idx].idxmax())
    return float(drawdown.loc[end_idx]), start_idx, end_idx


def prediction_metrics(pred_df):
    daily_ic = pred_df.groupby("trade_date")[["pred_alpha", LABEL_COLUMN]].apply(
        lambda x: x["pred_alpha"].corr(x[LABEL_COLUMN])
    )
    daily_rank_ic = pred_df.groupby("trade_date")[["pred_alpha", LABEL_COLUMN]].apply(
        lambda x: x["pred_alpha"].corr(x[LABEL_COLUMN], method="spearman")
    )

    def top_bottom(day_df):
        rank_pct = day_df["pred_alpha"].rank(pct=True, method="first")
        top = day_df.loc[rank_pct >= 0.8, LABEL_COLUMN].mean()
        bottom = day_df.loc[rank_pct <= 0.2, LABEL_COLUMN].mean()
        return top - bottom

    spread = pred_df.groupby("trade_date")[["pred_alpha", LABEL_COLUMN]].apply(top_bottom)
    return {
        "IC": daily_ic.mean(),
        "Rank IC": daily_rank_ic.mean(),
        "ICIR": daily_ic.mean() / daily_ic.std(ddof=1),
        "Rank ICIR": daily_rank_ic.mean() / daily_rank_ic.std(ddof=1),
        "Top-Bottom Spread": spread.mean(),
    }


def calculate_metrics(daily_df, pred_df):
    n_days = len(daily_df)
    annual_factor = 252 / n_days
    cumulative_nav = daily_df["nav"].iloc[-1]
    benchmark_nav = daily_df["benchmark_nav"].iloc[-1]
    annual_return = cumulative_nav ** annual_factor - 1
    annual_volatility = daily_df["portfolio_ret"].std(ddof=1) * np.sqrt(252)
    sharpe = annual_return / annual_volatility
    mdd, mdd_start_idx, mdd_end_idx = max_drawdown(daily_df["nav"])
    calmar = annual_return / abs(mdd)
    cumulative_excess_return = cumulative_nav / benchmark_nav - 1
    annual_excess_return = (1 + cumulative_excess_return) ** annual_factor - 1
    pred_metrics = prediction_metrics(pred_df)

    rows = [
        ("累计净值", cumulative_nav),
        ("累计收益", cumulative_nav - 1),
        ("年化收益", annual_return),
        ("夏普比率", sharpe),
        ("最大回撤", mdd),
        ("最大回撤开始", daily_df.loc[mdd_start_idx, "realized_date"]),
        ("最大回撤结束", daily_df.loc[mdd_end_idx, "realized_date"]),
        ("卡玛比率", calmar),
        ("年化收益/回撤比", calmar),
        ("基准累计净值", benchmark_nav),
        ("基准累计收益", benchmark_nav - 1),
        ("累计超额收益", cumulative_excess_return),
        ("年化超额收益", annual_excess_return),
        *pred_metrics.items(),
    ]
    return pd.DataFrame(rows, columns=["metric", "LightGBM"])


def plot_nav(daily_df):
    plt.figure(figsize=(10, 5))
    plt.plot(daily_df["realized_date"], daily_df["nav"], label="LightGBM")
    plt.plot(daily_df["realized_date"], daily_df["benchmark_nav"], label="STAR50 Benchmark")
    plt.plot(daily_df["realized_date"], daily_df["excess_nav"], label="Excess NAV")
    plt.title("LightGBM Fixed-Window Backtest NAV")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spyder_lightgbm_nav.png", dpi=160)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading raw parquet data...")
    stock_df, index_df = load_raw_data()

    print("2. Building features and labels...")
    raw_features = build_feature_dataset(stock_df, index_df)
    raw_features.to_parquet(OUTPUT_DIR / "spyder_raw_features.parquet", index=False)

    print("3. Preprocessing cross-section features...")
    features = preprocess_features(raw_features)
    features.to_parquet(OUTPUT_DIR / "spyder_preprocessed_features.parquet", index=False)

    print("4. Building forward returns...")
    forward_returns = load_forward_returns(stock_df, index_df)

    print("5. Fixed-window LightGBM training and prediction...")
    predictions = rolling_fixed_window_predict(features)
    predictions.to_parquet(OUTPUT_DIR / "spyder_lightgbm_predictions.parquet", index=False)

    print("6. Backtesting top-alpha portfolio...")
    daily_nav = build_daily_strategy_returns(predictions, forward_returns)
    daily_nav.to_csv(OUTPUT_DIR / "spyder_lightgbm_daily_nav.csv", index=False, encoding="utf-8-sig")

    print("7. Calculating metrics...")
    metrics = calculate_metrics(daily_nav, predictions)
    metrics.to_csv(OUTPUT_DIR / "spyder_lightgbm_metrics.csv", index=False, encoding="utf-8-sig")
    plot_nav(daily_nav)

    print("\nFinal metrics:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
