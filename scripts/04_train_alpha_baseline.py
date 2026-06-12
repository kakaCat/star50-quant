#!/usr/bin/env python3
"""
阶段二：Alpha 预测 Baseline 模型

读取阶段一预处理后的特征数据，使用 LightGBM 预测未来超额收益标签。
采用按交易日排序的时间切分，避免未来数据进入训练集。
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "star50_features_stage1_preprocessed.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "alpha_baseline"
MODEL_DIR = PROJECT_ROOT / "models" / "alpha"

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


def load_dataset(label_column: str) -> pd.DataFrame:
    """读取并校验训练数据。"""
    df = pd.read_parquet(INPUT_FILE)
    required_columns = ["ts_code", "trade_date", label_column, *FEATURE_COLUMNS]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要字段: {missing_columns}")

    df = df[required_columns].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[label_column, *FEATURE_COLUMNS])
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return df


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按交易日切分 train/validation/test，避免未来函数。"""
    dates = np.array(sorted(df["trade_date"].unique()))
    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_df = df[df["trade_date"].isin(train_dates)].copy()
    val_df = df[df["trade_date"].isin(val_dates)].copy()
    test_df = df[df["trade_date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, label_column: str) -> lgb.LGBMRegressor:
    """训练 LightGBM 回归模型。"""
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[label_column],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[label_column])],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def information_coefficient(df: pd.DataFrame, label_column: str) -> tuple[float, float]:
    """计算日均 IC 和 Rank IC。"""
    daily_ic = df.groupby("trade_date")[["pred_alpha", label_column]].apply(
        lambda x: x["pred_alpha"].corr(x[label_column])
    )
    daily_rank_ic = df.groupby("trade_date")[["pred_alpha", label_column]].apply(
        lambda x: x["pred_alpha"].corr(x[label_column], method="spearman")
    )
    return daily_ic.mean(), daily_rank_ic.mean()


def top_bottom_spread(df: pd.DataFrame, label_column: str) -> float:
    """计算每日预测前20%减后20%的平均真实超额收益差。"""
    def calc_one_day(day_df: pd.DataFrame) -> float:
        rank_pct = day_df["pred_alpha"].rank(pct=True, method="first")
        top = day_df.loc[rank_pct >= 0.8, label_column].mean()
        bottom = day_df.loc[rank_pct <= 0.2, label_column].mean()
        return top - bottom

    return df.groupby("trade_date")[["pred_alpha", label_column]].apply(calc_one_day).mean()


def evaluate_predictions(df: pd.DataFrame, label_column: str, split_name: str) -> dict:
    """计算回归和横截面排序指标。"""
    rmse = mean_squared_error(df[label_column], df["pred_alpha"]) ** 0.5
    mae = mean_absolute_error(df[label_column], df["pred_alpha"])
    ic, rank_ic = information_coefficient(df, label_column)
    spread = top_bottom_spread(df, label_column)

    return {
        "split": split_name,
        "rows": len(df),
        "start_date": df["trade_date"].min(),
        "end_date": df["trade_date"].max(),
        "rmse": rmse,
        "mae": mae,
        "ic": ic,
        "rank_ic": rank_ic,
        "top_bottom_spread": spread,
    }


def save_outputs(
    model: lgb.LGBMRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
) -> None:
    """保存模型、预测、指标和特征重要性。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for split_df in [train_df, val_df, test_df]:
        split_df["pred_alpha"] = model.predict(split_df[FEATURE_COLUMNS])

    pred_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="validation"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    pred_df[["split", "ts_code", "trade_date", label_column, "pred_alpha"]].to_parquet(
        OUTPUT_DIR / f"alpha_predictions_{label_column}.parquet",
        index=False,
    )

    metrics = pd.DataFrame(
        [
            evaluate_predictions(train_df, label_column, "train"),
            evaluate_predictions(val_df, label_column, "validation"),
            evaluate_predictions(test_df, label_column, "test"),
        ]
    )
    metrics.to_csv(OUTPUT_DIR / f"alpha_metrics_{label_column}.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / f"alpha_feature_importance_{label_column}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model.booster_.save_model(str(MODEL_DIR / f"lightgbm_alpha_{label_column}.txt"))

    print("\n" + "=" * 100)
    print("阶段二 Alpha Baseline 训练完成")
    print(metrics.to_string(index=False))
    print("\n特征重要性前10：")
    print(importance.head(10).to_string(index=False))
    print(f"\n预测文件：{OUTPUT_DIR / f'alpha_predictions_{label_column}.parquet'}")
    print(f"指标文件：{OUTPUT_DIR / f'alpha_metrics_{label_column}.csv'}")
    print(f"重要性文件：{OUTPUT_DIR / f'alpha_feature_importance_{label_column}.csv'}")
    print(f"模型文件：{MODEL_DIR / f'lightgbm_alpha_{label_column}.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练阶段二 Alpha Baseline 模型")
    parser.add_argument(
        "--label",
        choices=["label_1d", "label_5d"],
        default="label_5d",
        help="预测目标，默认使用未来5日超额收益",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    df = load_dataset(args.label)
    train_df, val_df, test_df = split_by_date(df)

    print("=" * 100)
    print("阶段二 Alpha Baseline")
    print(f"输入文件：{INPUT_FILE}")
    print(f"标签：{args.label}")
    print(f"特征数：{len(FEATURE_COLUMNS)}")
    print(f"样本数：{len(df)}")
    print(f"Train：{len(train_df)} | {train_df['trade_date'].min()} -> {train_df['trade_date'].max()}")
    print(f"Validation：{len(val_df)} | {val_df['trade_date'].min()} -> {val_df['trade_date'].max()}")
    print(f"Test：{len(test_df)} | {test_df['trade_date'].min()} -> {test_df['trade_date'].max()}")

    model = train_model(train_df, val_df, args.label)
    save_outputs(model, train_df, val_df, test_df, args.label)


if __name__ == "__main__":
    main()
