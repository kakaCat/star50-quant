#!/usr/bin/env python3
"""
阶段一：基础量价因子与超额收益标签构建

使用 data/raw 中的个股后复权日行情和科创50指数日行情，生成阶段一
训练数据。该脚本只负责原始因子和标签构建；MAD 去极值、Z-Score
标准化等截面预处理建议放在下一步脚本中完成。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STOCK_FILE = RAW_DIR / "star50_daily_hfq_data_6yrs.parquet"
INDEX_FILE = RAW_DIR / "star50_index_daily_6yrs.parquet"
FEATURE_FILE = PROCESSED_DIR / "star50_features_stage1.parquet"
SUMMARY_FILE = OUTPUT_DIR / "stage1_feature_summary.csv"

FEATURE_COLUMNS = [
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
    "index_ret_1d",
    "index_volatility_20",
    "index_drawdown_20",
]

LABEL_COLUMNS = ["label_1d", "label_5d"]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取原始 Parquet 数据。"""
    stock_df = pd.read_parquet(STOCK_FILE)
    index_df = pd.read_parquet(INDEX_FILE)

    stock_df = stock_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    index_df = index_df.sort_values("trade_date").reset_index(drop=True)
    return stock_df, index_df


def add_index_features(index_df: pd.DataFrame) -> pd.DataFrame:
    """构造指数收益和市场状态特征。"""
    df = index_df[["trade_date", "close"]].copy()
    df = df.rename(columns={"close": "index_close"})
    df["index_ret_1d"] = df["index_close"].pct_change()
    df["index_ret_5d"] = df["index_close"].pct_change(5)
    df["index_future_ret_1d"] = df["index_close"].shift(-1) / df["index_close"] - 1
    df["index_future_ret_5d"] = df["index_close"].shift(-5) / df["index_close"] - 1
    df["index_volatility_20"] = df["index_ret_1d"].rolling(20).std()

    index_roll_max_20 = df["index_close"].rolling(20).max()
    df["index_drawdown_20"] = df["index_close"] / index_roll_max_20 - 1
    return df


def add_stock_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """构造个股基础量价特征。"""
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

    roll_max_20 = grouped["hfq_close"].rolling(20).max().reset_index(level=0, drop=True)
    df["drawdown_20"] = df["hfq_close"] / roll_max_20 - 1
    return df


def add_relative_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造相对科创50指数的滚动 beta 和相关性。"""
    result = df.copy()
    grouped = result.groupby("ts_code", group_keys=False)

    rolling_cov = (
        grouped[["ret_1d", "index_ret_1d"]]
        .rolling(60)
        .cov()
        .reset_index()
    )
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


def build_feature_dataset(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """合并个股、指数特征并生成超额收益标签。"""
    index_features = add_index_features(index_df)
    stock_features = add_stock_features(stock_df)

    df = stock_features.merge(index_features, on="trade_date", how="left", validate="many_to_one")
    df["label_1d"] = df["future_ret_1d"] - df["index_future_ret_1d"]
    df["label_5d"] = df["future_ret_5d"] - df["index_future_ret_5d"]
    df = add_relative_risk_features(df)

    output_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "hfq_close",
        "vol",
        "amount",
        *FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    ]
    df = df[output_columns].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def save_feature_summary(df: pd.DataFrame) -> None:
    """保存特征和标签的质量摘要。"""
    columns = FEATURE_COLUMNS + LABEL_COLUMNS
    summary = pd.DataFrame(
        {
            "column": columns,
            "dtype": [str(df[col].dtype) for col in columns],
            "non_null": [int(df[col].notna().sum()) for col in columns],
            "nulls": [int(df[col].isna().sum()) for col in columns],
            "mean": [df[col].mean(skipna=True) for col in columns],
            "std": [df[col].std(skipna=True) for col in columns],
            "min": [df[col].min(skipna=True) for col in columns],
            "p25": [df[col].quantile(0.25) for col in columns],
            "median": [df[col].median(skipna=True) for col in columns],
            "p75": [df[col].quantile(0.75) for col in columns],
            "max": [df[col].max(skipna=True) for col in columns],
        }
    )
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")


def print_validation_report(df: pd.DataFrame) -> None:
    """打印构建结果验收信息。"""
    duplicated_count = int(df.duplicated(["ts_code", "trade_date"]).sum())
    full_feature_rows = int(df[FEATURE_COLUMNS + LABEL_COLUMNS].notna().all(axis=1).sum())

    print("\n" + "=" * 100)
    print("阶段一特征数据构建完成")
    print(f"输出形状：{df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"证券数量：{df['ts_code'].nunique()}")
    print(f"日期范围：{df['trade_date'].min()} -> {df['trade_date'].max()}")
    print(f"重复键 ts_code + trade_date：{duplicated_count}")
    print(f"核心因子和标签完整行数：{full_feature_rows}")
    print(f"特征文件：{FEATURE_FILE}")
    print(f"摘要文件：{SUMMARY_FILE}")

    print("\n前5行：")
    preview_columns = [
        "ts_code",
        "trade_date",
        "ret_1d",
        "mom_20",
        "volatility_20",
        "beta_to_index_60",
        "corr_to_index_60",
        "label_1d",
        "label_5d",
    ]
    print(df[preview_columns].head(5).to_string(index=False))


def main() -> None:
    os.chdir(PROJECT_ROOT)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df, index_df = load_raw_data()
    feature_df = build_feature_dataset(stock_df, index_df)

    feature_df.to_parquet(FEATURE_FILE, index=False)
    save_feature_summary(feature_df)
    print_validation_report(feature_df)


if __name__ == "__main__":
    main()
