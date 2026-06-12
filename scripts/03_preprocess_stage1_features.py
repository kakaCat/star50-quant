#!/usr/bin/env python3
"""
阶段一：截面预处理

对基础量价因子执行：
1. 同股票前向填充
2. 同交易日截面中位数填充
3. 同交易日截面 MAD 去极值
4. 同交易日截面 Z-Score 标准化

标签列不参与填充、去极值或标准化。最终删除仍缺少核心因子或标签的行。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_FILE = PROCESSED_DIR / "star50_features_stage1.parquet"
OUTPUT_FILE = PROCESSED_DIR / "star50_features_stage1_preprocessed.parquet"
SUMMARY_FILE = OUTPUT_DIR / "stage1_preprocess_summary.csv"

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
KEY_COLUMNS = ["ts_code", "trade_date"]


def mad_winsorize(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """对单个交易日截面做 MAD 去极值。"""
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)

    if pd.isna(median) or pd.isna(mad) or mad == 0:
        return series

    scale = 1.4826 * mad
    lower = median - n_mad * scale
    upper = median + n_mad * scale
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    """对单个交易日截面做 Z-Score 标准化。"""
    mean = series.mean(skipna=True)
    std = series.std(skipna=True)

    if pd.isna(mean) or pd.isna(std):
        return series
    if std < 1e-12:
        return series * 0
    return (series - mean) / std


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """先按股票前向填充，再按交易日截面中位数填充。"""
    result = df.sort_values(["ts_code", "trade_date"]).copy()
    result[FEATURE_COLUMNS] = result.groupby("ts_code")[FEATURE_COLUMNS].ffill()
    result[FEATURE_COLUMNS] = result.groupby("trade_date")[FEATURE_COLUMNS].transform(
        lambda col: col.fillna(col.median(skipna=True))
    )
    return result


def preprocess_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """按交易日截面执行 MAD 去极值和 Z-Score 标准化。"""
    result = df.sort_values(["trade_date", "ts_code"]).copy()

    for column in STOCK_FEATURE_COLUMNS:
        result[column] = result.groupby("trade_date")[column].transform(mad_winsorize)
        result[column] = result.groupby("trade_date")[column].transform(zscore)

    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return result


def build_summary(before: pd.DataFrame, after_fill: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """生成预处理摘要。"""
    rows = []
    for column in FEATURE_COLUMNS + LABEL_COLUMNS:
        rows.append(
            {
                "column": column,
                "raw_nulls": int(before[column].isna().sum()),
                "after_fill_nulls": int(after_fill[column].isna().sum()),
                "final_nulls": int(final_df[column].isna().sum()),
                "final_mean": final_df[column].mean(skipna=True),
                "final_std": final_df[column].std(skipna=True),
                "final_min": final_df[column].min(skipna=True),
                "final_median": final_df[column].median(skipna=True),
                "final_max": final_df[column].max(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def validate_cross_section(final_df: pd.DataFrame) -> pd.DataFrame:
    """检查截面标准化后每日均值和标准差是否接近期望值。"""
    date_stats = final_df.groupby("trade_date")[STOCK_FEATURE_COLUMNS].agg(["mean", "std"])
    max_abs_mean = date_stats.xs("mean", axis=1, level=1).abs().max().max()
    std_values = date_stats.xs("std", axis=1, level=1)
    variable_std_values = std_values.where(std_values > 1e-12)
    max_std_gap = (variable_std_values - 1).abs().max().max()

    return pd.DataFrame(
        [
            {"check": "max_abs_daily_feature_mean", "value": max_abs_mean},
            {"check": "max_abs_daily_feature_std_minus_1", "value": max_std_gap},
        ]
    )


def print_validation_report(raw_df: pd.DataFrame, filled_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    """打印预处理结果。"""
    duplicate_count = int(final_df.duplicated(KEY_COLUMNS).sum())
    inf_count = int(np.isinf(final_df.select_dtypes(include="number")).sum().sum())
    dropped_rows = len(raw_df) - len(final_df)
    cross_section_checks = validate_cross_section(final_df)

    print("\n" + "=" * 100)
    print("阶段一截面预处理完成")
    print(f"原始行数：{len(raw_df)}")
    print(f"填充后仍缺核心因子/标签的行数：{int(filled_df[FEATURE_COLUMNS + LABEL_COLUMNS].isna().any(axis=1).sum())}")
    print(f"最终行数：{len(final_df)}")
    print(f"删除行数：{dropped_rows}")
    print(f"重复键 ts_code + trade_date：{duplicate_count}")
    print(f"无限值 inf：{inf_count}")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"摘要文件：{SUMMARY_FILE}")

    print("\n截面标准化检查：")
    print(cross_section_checks.to_string(index=False))

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
    print(final_df[preview_columns].head(5).to_string(index=False))


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_parquet(INPUT_FILE)
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)

    filled_df = fill_missing_features(raw_df)
    trainable_df = filled_df.dropna(subset=FEATURE_COLUMNS + LABEL_COLUMNS)
    preprocessed_df = preprocess_cross_section(trainable_df)
    final_df = preprocessed_df.reset_index(drop=True)

    final_df.to_parquet(OUTPUT_FILE, index=False)

    summary = build_summary(raw_df, filled_df, final_df)
    checks = validate_cross_section(final_df)
    pd.concat([summary, checks], ignore_index=True).to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_validation_report(raw_df, filled_df, final_df)


if __name__ == "__main__":
    main()
