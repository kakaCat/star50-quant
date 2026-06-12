#!/usr/bin/env python3
"""
阶段二：LightGBM 固定窗口训练与净值指标输出

流程：
1. 使用阶段一预处理后的因子数据；
2. 采用固定长度训练窗口滚动训练 LightGBM；
3. 对未来预测窗口生成 Alpha 预测；
4. 每隔固定交易日调仓，买入预测值最高的一组股票；
5. 输出累计净值、年化收益、夏普比率、最大回撤、超额收益等指标。
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "star50_features_stage1_preprocessed.parquet"
STOCK_FILE = PROJECT_ROOT / "data" / "raw" / "star50_daily_hfq_data_6yrs.parquet"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "star50_index_daily_6yrs.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "alpha_fixed_window"

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


def load_features(label_column: str) -> pd.DataFrame:
    """读取训练特征。"""
    df = pd.read_parquet(FEATURE_FILE)
    columns = ["ts_code", "trade_date", label_column, *FEATURE_COLUMNS]
    df = df[columns].dropna(subset=[label_column, *FEATURE_COLUMNS]).copy()
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return df


def load_forward_returns() -> pd.DataFrame:
    """读取原始行情，计算下一交易日个股收益和指数收益。"""
    stock_df = pd.read_parquet(STOCK_FILE)
    index_df = pd.read_parquet(INDEX_FILE)

    stock_df = stock_df.sort_values(["ts_code", "trade_date"]).copy()
    stock_grouped = stock_df.groupby("ts_code", group_keys=False)
    stock_df["stock_ret_next_1d"] = stock_grouped["hfq_close"].shift(-1) / stock_df["hfq_close"] - 1
    stock_df["realized_date"] = stock_grouped["trade_date"].shift(-1)

    index_df = index_df.sort_values("trade_date").copy()
    index_df["index_ret_next_1d"] = index_df["close"].shift(-1) / index_df["close"] - 1

    returns = stock_df[["ts_code", "trade_date", "realized_date", "stock_ret_next_1d"]].merge(
        index_df[["trade_date", "index_ret_next_1d"]],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    return returns.dropna(subset=["realized_date", "stock_ret_next_1d", "index_ret_next_1d"])


def train_one_window(
    train_df: pd.DataFrame,
    label_column: str,
    lgbm_params: dict | None = None,
) -> lgb.LGBMRegressor:
    """训练单个固定窗口 LightGBM 模型。"""
    model_params = {
        "n_estimators": 120,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    if lgbm_params:
        model_params.update(lgbm_params)

    model = lgb.LGBMRegressor(
        objective="regression",
        force_row_wise=True,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
        **model_params,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df[label_column])
    return model


def make_lgbm_params(args: argparse.Namespace) -> dict:
    """从命令行参数生成 LightGBM 参数。"""
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
    }


def rolling_fixed_window_predict(
    df: pd.DataFrame,
    label_column: str,
    train_window_days: int,
    predict_window_days: int,
    lgbm_params: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """固定训练窗口滚动预测。"""
    all_dates = np.array(sorted(df["trade_date"].unique()))
    prediction_parts = []
    window_id = 0

    start_idx = train_window_days
    while start_idx < len(all_dates):
        train_dates = all_dates[start_idx - train_window_days:start_idx]
        predict_dates = all_dates[start_idx:start_idx + predict_window_days]

        train_df = df[df["trade_date"].isin(train_dates)]
        predict_df = df[df["trade_date"].isin(predict_dates)].copy()
        if train_df.empty or predict_df.empty:
            break

        model = train_one_window(train_df, label_column, lgbm_params=lgbm_params)
        predict_df["pred_alpha"] = model.predict(predict_df[FEATURE_COLUMNS])
        predict_df["window_id"] = window_id
        predict_df["train_start"] = train_dates[0]
        predict_df["train_end"] = train_dates[-1]
        prediction_parts.append(
            predict_df[[
                "window_id",
                "train_start",
                "train_end",
                "ts_code",
                "trade_date",
                label_column,
                "pred_alpha",
            ]]
        )

        if verbose:
            print(
                f"窗口 {window_id:02d}: train {train_dates[0].date()} -> {train_dates[-1].date()}, "
                f"predict {predict_dates[0].date()} -> {predict_dates[-1].date()}, rows={len(predict_df)}"
            )
        window_id += 1
        start_idx += predict_window_days

    return pd.concat(prediction_parts, ignore_index=True)


def build_daily_strategy_returns(
    pred_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    rebalance_days: int,
    top_quantile: float,
    transaction_cost: float,
) -> pd.DataFrame:
    """根据预测值构建 Top 分组等权组合日收益。"""
    pred_df = pred_df.sort_values(["trade_date", "pred_alpha"]).copy()
    dates = np.array(sorted(pred_df["trade_date"].unique()))
    daily_rows = []
    previous_holdings: set[str] = set()

    for i in range(0, len(dates), rebalance_days):
        rebalance_date = dates[i]
        holding_signal_dates = dates[i:i + rebalance_days]

        cross_section = pred_df[pred_df["trade_date"].eq(rebalance_date)].copy()
        cutoff = cross_section["pred_alpha"].quantile(1 - top_quantile)
        holdings = set(cross_section.loc[cross_section["pred_alpha"] >= cutoff, "ts_code"])
        if not holdings:
            continue

        turnover = 1.0
        if previous_holdings:
            overlap = len(holdings.intersection(previous_holdings)) / len(holdings)
            turnover = 1 - overlap
        cost = transaction_cost * turnover
        previous_holdings = holdings

        period_returns = forward_returns[
            forward_returns["trade_date"].isin(holding_signal_dates)
            & forward_returns["ts_code"].isin(holdings)
        ]
        grouped = period_returns.groupby("trade_date", sort=True)
        for j, (signal_date, day_df) in enumerate(grouped):
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


def max_drawdown(nav: pd.Series) -> tuple[float, int, int]:
    """计算最大回撤及开始、结束位置。"""
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    end_idx = int(drawdown.idxmin())
    start_idx = int(nav.loc[:end_idx].idxmax())
    return float(drawdown.loc[end_idx]), start_idx, end_idx


def calculate_prediction_metrics(pred_df: pd.DataFrame, label_column: str) -> dict:
    """计算固定窗口预测结果的 IC 类指标。"""
    daily_ic = pred_df.groupby("trade_date")[["pred_alpha", label_column]].apply(
        lambda x: x["pred_alpha"].corr(x[label_column])
    )
    daily_rank_ic = pred_df.groupby("trade_date")[["pred_alpha", label_column]].apply(
        lambda x: x["pred_alpha"].corr(x[label_column], method="spearman")
    )

    def calc_top_bottom(day_df: pd.DataFrame) -> float:
        rank_pct = day_df["pred_alpha"].rank(pct=True, method="first")
        top = day_df.loc[rank_pct >= 0.8, label_column].mean()
        bottom = day_df.loc[rank_pct <= 0.2, label_column].mean()
        return top - bottom

    top_bottom = pred_df.groupby("trade_date")[["pred_alpha", label_column]].apply(calc_top_bottom)
    return {
        "ic": daily_ic.mean(),
        "rank_ic": daily_rank_ic.mean(),
        "ic_ir": daily_ic.mean() / daily_ic.std(ddof=1) if daily_ic.std(ddof=1) != 0 else np.nan,
        "rank_ic_ir": daily_rank_ic.mean() / daily_rank_ic.std(ddof=1)
        if daily_rank_ic.std(ddof=1) != 0
        else np.nan,
        "top_bottom_spread": top_bottom.mean(),
    }


def calculate_metrics(
    daily_df: pd.DataFrame,
    pred_df: pd.DataFrame | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """输出与研报表格风格一致的净值指标。"""
    n_days = len(daily_df)
    annual_factor = 252 / n_days

    cumulative_nav = daily_df["nav"].iloc[-1]
    benchmark_nav = daily_df["benchmark_nav"].iloc[-1]
    cumulative_return = cumulative_nav - 1
    benchmark_return = benchmark_nav - 1
    annual_return = cumulative_nav ** annual_factor - 1
    annual_volatility = daily_df["portfolio_ret"].std(ddof=1) * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    mdd, mdd_start_idx, mdd_end_idx = max_drawdown(daily_df["nav"])
    annual_return_drawdown = annual_return / abs(mdd) if mdd != 0 else np.nan

    cumulative_excess_return = cumulative_nav / benchmark_nav - 1
    annual_excess_return = (1 + cumulative_excess_return) ** annual_factor - 1

    rows = [
        {"metric": "累计净值", "LightGBM": cumulative_nav},
        {"metric": "累计收益", "LightGBM": cumulative_return},
        {"metric": "年化收益", "LightGBM": annual_return},
        {"metric": "夏普比率", "LightGBM": sharpe},
        {"metric": "最大回撤", "LightGBM": mdd},
        {"metric": "最大回撤开始", "LightGBM": daily_df.loc[mdd_start_idx, "realized_date"]},
        {"metric": "最大回撤结束", "LightGBM": daily_df.loc[mdd_end_idx, "realized_date"]},
        {"metric": "卡玛比率", "LightGBM": annual_return_drawdown},
        {"metric": "年化收益/回撤比", "LightGBM": annual_return_drawdown},
        {"metric": "基准累计净值", "LightGBM": benchmark_nav},
        {"metric": "基准累计收益", "LightGBM": benchmark_return},
        {"metric": "累计超额收益", "LightGBM": cumulative_excess_return},
        {"metric": "年化超额收益", "LightGBM": annual_excess_return},
    ]

    if pred_df is not None and label_column is not None:
        pred_metrics = calculate_prediction_metrics(pred_df, label_column)
        rows.extend(
            [
                {"metric": "IC", "LightGBM": pred_metrics["ic"]},
                {"metric": "Rank IC", "LightGBM": pred_metrics["rank_ic"]},
                {"metric": "ICIR", "LightGBM": pred_metrics["ic_ir"]},
                {"metric": "Rank ICIR", "LightGBM": pred_metrics["rank_ic_ir"]},
                {"metric": "Top-Bottom Spread", "LightGBM": pred_metrics["top_bottom_spread"]},
            ]
        )

    return pd.DataFrame(rows)


def plot_nav(daily_df: pd.DataFrame, output_file: Path) -> None:
    """保存策略与基准净值图。"""
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
    plt.savefig(output_file, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 固定窗口训练与净值指标输出")
    parser.add_argument("--label", choices=["label_1d", "label_5d"], default="label_5d")
    parser.add_argument("--train-window-days", type=int, default=756, help="固定训练窗口交易日数")
    parser.add_argument("--predict-window-days", type=int, default=20, help="每次模型预测交易日数")
    parser.add_argument("--rebalance-days", type=int, default=5, help="调仓间隔交易日数")
    parser.add_argument("--top-quantile", type=float, default=0.2, help="买入预测值最高的比例")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="单次换手成本，默认不扣费")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features(args.label)
    forward_returns = load_forward_returns()

    print("=" * 100)
    print("LightGBM 固定窗口训练")
    print(f"标签：{args.label}")
    print(f"训练窗口：{args.train_window_days} 个交易日")
    print(f"预测窗口：{args.predict_window_days} 个交易日")
    print(f"调仓间隔：{args.rebalance_days} 个交易日")
    print(f"Top 比例：{args.top_quantile:.0%}")
    print(f"交易成本：{args.transaction_cost:.4%}")
    lgbm_params = make_lgbm_params(args)
    print(f"LightGBM 参数：{lgbm_params}")

    predictions = rolling_fixed_window_predict(
        features,
        label_column=args.label,
        train_window_days=args.train_window_days,
        predict_window_days=args.predict_window_days,
        lgbm_params=lgbm_params,
    )
    daily_returns = build_daily_strategy_returns(
        predictions,
        forward_returns,
        rebalance_days=args.rebalance_days,
        top_quantile=args.top_quantile,
        transaction_cost=args.transaction_cost,
    )
    metrics = calculate_metrics(daily_returns, predictions, args.label)

    pred_file = OUTPUT_DIR / f"fixed_window_predictions_{args.label}.parquet"
    daily_file = OUTPUT_DIR / f"fixed_window_daily_nav_{args.label}.csv"
    metrics_file = OUTPUT_DIR / f"fixed_window_metrics_{args.label}.csv"
    nav_plot_file = OUTPUT_DIR / f"fixed_window_nav_{args.label}.png"

    predictions.to_parquet(pred_file, index=False)
    daily_returns.to_csv(daily_file, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    plot_nav(daily_returns, nav_plot_file)

    print("\n" + "=" * 100)
    print("固定窗口 LightGBM 净值指标")
    print(metrics.to_string(index=False))
    print(f"\n预测文件：{pred_file}")
    print(f"每日净值：{daily_file}")
    print(f"指标文件：{metrics_file}")
    print(f"净值图：{nav_plot_file}")


if __name__ == "__main__":
    main()
