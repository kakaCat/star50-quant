#!/usr/bin/env python3
"""
阶段二：LightGBM 固定窗口网格搜索

在小范围参数网格上搜索固定窗口 LightGBM 策略，输出全部试验结果、
按卡玛目标排序的最佳结果，以及低回撤候选结果。
"""

import argparse
import importlib.util
import itertools
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_WINDOW_SCRIPT = PROJECT_ROOT / "scripts" / "05_lightgbm_fixed_window_backtest.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "alpha_fixed_window_grid"


def load_fixed_window_module():
    """加载固定窗口回测脚本中的函数。"""
    spec = importlib.util.spec_from_file_location("fixed_window_backtest", FIXED_WINDOW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    """从指标表中取数值。"""
    value = metrics.loc[metrics["metric"].eq(name), "LightGBM"].iloc[0]
    return float(value)


def build_grid(mode: str) -> list[dict]:
    """生成参数网格。"""
    if mode == "quick":
        grid = {
            "n_estimators": [80, 120, 160],
            "learning_rate": [0.03, 0.06],
            "num_leaves": [8, 31],
            "min_child_samples": [80, 160],
            "subsample": [0.7],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.001, 0.1],
            "reg_lambda": [0.1, 1.0],
            "rebalance_days": [5],
            "top_quantile": [0.15, 0.20, 0.30],
        }
    else:
        grid = {
            "n_estimators": [80, 120, 160, 200],
            "learning_rate": [0.02, 0.03, 0.05, 0.07],
            "num_leaves": [8, 16, 31, 63],
            "min_child_samples": [60, 100, 160],
            "subsample": [0.65, 0.8],
            "colsample_bytree": [0.7, 0.9],
            "reg_alpha": [0.001, 0.1],
            "reg_lambda": [0.1, 1.0],
            "rebalance_days": [5, 10],
            "top_quantile": [0.10, 0.15, 0.20, 0.30],
        }

    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def split_params(params: dict) -> tuple[dict, dict]:
    """拆分模型参数和组合参数。"""
    lgbm_keys = [
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]
    portfolio_keys = ["rebalance_days", "top_quantile"]
    return (
        {key: params[key] for key in lgbm_keys},
        {key: params[key] for key in portfolio_keys},
    )


def evaluate_grid_item(
    fw,
    features: pd.DataFrame,
    forward_returns: pd.DataFrame,
    label: str,
    train_window_days: int,
    predict_window_days: int,
    transaction_cost: float,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行一个网格参数组合。"""
    lgbm_params, portfolio_params = split_params(params)
    predictions = fw.rolling_fixed_window_predict(
        features,
        label_column=label,
        train_window_days=train_window_days,
        predict_window_days=predict_window_days,
        lgbm_params=lgbm_params,
        verbose=False,
    )
    daily_returns = fw.build_daily_strategy_returns(
        predictions,
        forward_returns,
        rebalance_days=portfolio_params["rebalance_days"],
        top_quantile=portfolio_params["top_quantile"],
        transaction_cost=transaction_cost,
    )
    metrics = fw.calculate_metrics(daily_returns, predictions, label)
    return metrics, predictions, daily_returns


def objective_from_metrics(metrics: pd.DataFrame) -> float:
    """卡玛优先、回撤惩罚的排序目标。"""
    calmar = metric_value(metrics, "卡玛比率")
    annual_return = metric_value(metrics, "年化收益")
    max_drawdown = metric_value(metrics, "最大回撤")
    rank_ic = metric_value(metrics, "Rank IC")
    drawdown_penalty = max(0.0, abs(max_drawdown) - 0.40) * 6.0
    drawdown_reward = max(0.0, 0.40 - abs(max_drawdown)) * 1.5
    return calmar + 0.15 * annual_return + 0.3 * rank_ic + drawdown_reward - drawdown_penalty


def save_payload(
    fw,
    label: str,
    prefix: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    params: dict,
) -> None:
    """保存某个候选结果。"""
    metrics.to_csv(OUTPUT_DIR / f"{prefix}_metrics_{label}.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(OUTPUT_DIR / f"{prefix}_predictions_{label}.parquet", index=False)
    daily_returns.to_csv(OUTPUT_DIR / f"{prefix}_daily_nav_{label}.csv", index=False, encoding="utf-8-sig")
    fw.plot_nav(daily_returns, OUTPUT_DIR / f"{prefix}_nav_{label}.png")
    pd.DataFrame([params]).to_csv(OUTPUT_DIR / f"{prefix}_params_{label}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 固定窗口网格搜索")
    parser.add_argument("--label", choices=["label_1d", "label_5d"], default="label_5d")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--max-runs", type=int, default=24, help="最多运行多少组；0 表示跑完整网格")
    parser.add_argument("--train-window-days", type=int, default=756)
    parser.add_argument("--predict-window-days", type=int, default=20)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fw = load_fixed_window_module()
    features = fw.load_features(args.label)
    forward_returns = fw.load_forward_returns()

    grid_items = build_grid(args.mode)
    if args.max_runs > 0:
        grid_items = grid_items[:args.max_runs]

    print("=" * 100)
    print("LightGBM 固定窗口网格搜索")
    print(f"模式：{args.mode}")
    print(f"运行组数：{len(grid_items)}")
    print(f"标签：{args.label}")

    records = []
    best_payload = None
    low_drawdown_payload = None

    for i, params in enumerate(grid_items):
        metrics, predictions, daily_returns = evaluate_grid_item(
            fw=fw,
            features=features,
            forward_returns=forward_returns,
            label=args.label,
            train_window_days=args.train_window_days,
            predict_window_days=args.predict_window_days,
            transaction_cost=args.transaction_cost,
            params=params,
        )
        score = objective_from_metrics(metrics)
        record = {
            "run": i,
            "score": score,
            "累计净值": metric_value(metrics, "累计净值"),
            "年化收益": metric_value(metrics, "年化收益"),
            "最大回撤": metric_value(metrics, "最大回撤"),
            "卡玛比率": metric_value(metrics, "卡玛比率"),
            "夏普比率": metric_value(metrics, "夏普比率"),
            "IC": metric_value(metrics, "IC"),
            "Rank IC": metric_value(metrics, "Rank IC"),
            "Top-Bottom Spread": metric_value(metrics, "Top-Bottom Spread"),
            **params,
        }
        records.append(record)

        if best_payload is None or score > best_payload["score"]:
            best_payload = {
                "score": score,
                "metrics": metrics,
                "predictions": predictions,
                "daily_returns": daily_returns,
                "params": params,
            }

        if low_drawdown_payload is None or abs(record["最大回撤"]) < abs(low_drawdown_payload["record"]["最大回撤"]):
            low_drawdown_payload = {
                "record": record,
                "metrics": metrics,
                "predictions": predictions,
                "daily_returns": daily_returns,
                "params": params,
            }

        print(
            f"run={i:03d} score={score:.4f} calmar={record['卡玛比率']:.4f} "
            f"mdd={record['最大回撤']:.2%} ann={record['年化收益']:.2%} "
            f"rank_ic={record['Rank IC']:.4f}"
        )

    results = pd.DataFrame(records).sort_values("score", ascending=False)
    results_file = OUTPUT_DIR / f"grid_results_{args.label}.csv"
    results.to_csv(results_file, index=False, encoding="utf-8-sig")

    save_payload(
        fw,
        args.label,
        "best",
        best_payload["metrics"],
        best_payload["predictions"],
        best_payload["daily_returns"],
        best_payload["params"],
    )
    save_payload(
        fw,
        args.label,
        "low_drawdown",
        low_drawdown_payload["metrics"],
        low_drawdown_payload["predictions"],
        low_drawdown_payload["daily_returns"],
        low_drawdown_payload["params"],
    )

    print("\n" + "=" * 100)
    print("网格搜索完成")
    print("Top 10：")
    print(results.head(10).to_string(index=False))
    print(f"\n全部结果：{results_file}")
    print(f"最佳指标：{OUTPUT_DIR / f'best_metrics_{args.label}.csv'}")
    print(f"低回撤指标：{OUTPUT_DIR / f'low_drawdown_metrics_{args.label}.csv'}")


if __name__ == "__main__":
    main()
