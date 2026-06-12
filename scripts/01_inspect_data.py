#!/usr/bin/env python3
"""
阶段一：原始数据检查脚本

读取 data/raw 中的科创50个股与指数 Parquet 文件，输出基础结构、
前5行、缺失值、重复值、日期范围，并保存质量检查摘要。
"""

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_FILE = OUTPUT_DIR / "data_quality_summary.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STOCK_FILE = RAW_DIR / "star50_daily_hfq_data_6yrs.parquet"
INDEX_FILE = RAW_DIR / "star50_index_daily_6yrs.parquet"


def build_column_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """生成字段级质量摘要。"""
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null": [int(df[col].notna().sum()) for col in df.columns],
            "nulls": [int(df[col].isna().sum()) for col in df.columns],
            "nunique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def print_dataset_report(
    df: pd.DataFrame,
    dataset_name: str,
    duplicate_keys: list[str],
) -> pd.DataFrame:
    """打印单个数据集检查报告，并返回字段摘要。"""
    print("\n" + "=" * 100)
    print(f"数据集：{dataset_name}")
    print(f"形状：{df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"内存占用：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    if "trade_date" in df.columns:
        print(f"日期范围：{df['trade_date'].min()} -> {df['trade_date'].max()}")

    if "ts_code" in df.columns:
        print(f"证券数量：{df['ts_code'].nunique()}")

    duplicated_count = int(df.duplicated(duplicate_keys).sum())
    print(f"按 {duplicate_keys} 检查重复行：{duplicated_count}")

    print("\n字段摘要：")
    summary = build_column_summary(df, dataset_name)
    print(summary.to_string(index=False))

    print("\n前5行：")
    print(df.head(5).to_string(index=False))

    return summary


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df = pd.read_parquet(STOCK_FILE)
    index_df = pd.read_parquet(INDEX_FILE)

    stock_df = stock_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    index_df = index_df.sort_values("trade_date").reset_index(drop=True)

    summaries = [
        print_dataset_report(
            stock_df,
            dataset_name="star50_daily_hfq_data",
            duplicate_keys=["ts_code", "trade_date"],
        ),
        print_dataset_report(
            index_df,
            dataset_name="star50_index_daily",
            duplicate_keys=["trade_date"],
        ),
    ]

    quality_summary = pd.concat(summaries, ignore_index=True)
    quality_summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    common_dates = set(stock_df["trade_date"]).intersection(set(index_df["trade_date"]))
    print("\n" + "=" * 100)
    print("跨表对齐检查：")
    print(f"股票交易日期数量：{stock_df['trade_date'].nunique()}")
    print(f"指数交易日期数量：{index_df['trade_date'].nunique()}")
    print(f"共同交易日期数量：{len(common_dates)}")
    print(f"质量摘要已保存：{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
