#!/usr/bin/env python3
"""
Generate LightGBM IC diagnostic report in Spyder.

Run from /Users/elize/Desktop/量化 after spyder_lightgbm_pipeline.py has produced:
outputs/spyder_lightgbm_pipeline/spyder_lightgbm_predictions.parquet
outputs/spyder_lightgbm_pipeline/spyder_preprocessed_features.parquet
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def sanitize_import_paths():
    clean_paths = []
    for path in sys.path:
        if path == "":
            clean_paths.append(path)
            continue
        try:
            path_obj = Path(path).expanduser()
            if path_obj.exists():
                if path_obj.is_dir():
                    with os.scandir(path_obj):
                        pass
                clean_paths.append(path)
        except OSError:
            continue
    sys.path[:] = clean_paths
    sys.path_importer_cache.clear()


sanitize_import_paths()

import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import pyarrow


def patch_pyarrow_extension_registry():
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


patch_pyarrow_extension_registry()

BASE_DIR = Path("/Users/elize/Desktop/量化")
PIPELINE_DIR = BASE_DIR / "outputs" / "spyder_lightgbm_pipeline"
REPORT_DIR = BASE_DIR / "outputs" / "ic_report"
PREDICTION_PATH = PIPELINE_DIR / "spyder_lightgbm_predictions.parquet"
FEATURE_PATH = PIPELINE_DIR / "spyder_preprocessed_features.parquet"
REPORT_PATH = BASE_DIR / "lightgbm_ic_report_generated.png"
REPORT_PDF_PATH = BASE_DIR / "lightgbm_ic_report_generated.pdf"
SPLIT_DATE = pd.Timestamp("2025-01-01")

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

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "force_row_wise": True,
    "seed": 42,
    "num_threads": -1,
    "learning_rate": 0.045,
    "num_leaves": 62,
    "min_child_samples": 120,
    "subsample": 0.708884,
    "colsample_bytree": 0.913066,
    "reg_alpha": 0.000601,
    "reg_lambda": 0.483284,
}


def setup_chinese_font():
    font_candidates = ["Songti SC", "SimSong", "PingFang SC", "Heiti SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def build_daily_ic(pred):
    rows = []
    for date, day_df in pred.groupby("trade_date"):
        if len(day_df) < 5:
            continue
        rows.append(
            {
                "trade_date": date,
                "IC": day_df["pred_alpha"].corr(day_df["label_5d"]),
                "Rank_IC": day_df["pred_alpha"].corr(day_df["label_5d"], method="spearman"),
                "n_stocks": len(day_df),
            }
        )
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    daily["month"] = daily["trade_date"].dt.to_period("M").dt.to_timestamp()
    return daily


def build_summary(daily):
    def one_row(name, series):
        series = series.dropna()
        std = series.std(ddof=1)
        return {
            "区间": name,
            "IC均值": series.mean(),
            "IC标准差": std,
            "ICIR": series.mean() / std if std != 0 else np.nan,
            "正IC占比": (series > 0).mean(),
            "样本数": len(series),
        }

    in_sample = daily["trade_date"] < SPLIT_DATE
    out_sample = daily["trade_date"] >= SPLIT_DATE
    return pd.DataFrame(
        [
            one_row("样本内（2025前）", daily.loc[in_sample, "Rank_IC"]),
            one_row("样本外（2025后）", daily.loc[out_sample, "Rank_IC"]),
            one_row("全样本", daily["Rank_IC"]),
        ]
    )


def build_feature_importance():
    features = pd.read_parquet(FEATURE_PATH)
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    train_features = features[features["trade_date"] < SPLIT_DATE].dropna(
        subset=FEATURE_COLUMNS + ["label_5d"]
    )
    train_set = lgb.Dataset(
        train_features[FEATURE_COLUMNS],
        label=train_features["label_5d"],
        feature_name=FEATURE_COLUMNS,
        free_raw_data=False,
    )
    model = lgb.train(LGBM_PARAMS, train_set, num_boost_round=180)
    return pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_split", ascending=False)


def draw_report(daily, monthly, summary, importance, show_plot=True):
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.35, 1.0],
        height_ratios=[1.1, 1.1, 1.1],
        wspace=0.22,
        hspace=0.50,
    )
    fig.suptitle(
        "LightGBM 固定窗口模型 IC 诊断报告\n"
        "训练窗口: 756交易日滚动 | 预测期: 2023-05-18 → 2025-12-24 | 样本外切分: 2025-01-01",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    monthly_in = monthly[monthly["month"] < SPLIT_DATE]
    monthly_out = monthly[monthly["month"] >= SPLIT_DATE]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(monthly_in["month"], monthly_in["Rank_IC"], width=20, color="#9ecae1", alpha=0.8, label="样本内月度Rank IC")
    ax1.bar(monthly_out["month"], monthly_out["Rank_IC"], width=20, color="#fb6a4a", alpha=0.55, label="样本外月度Rank IC")
    ax1.plot(monthly["month"], monthly["Rank_IC"].rolling(3, min_periods=1).mean(), color="black", linestyle="--", linewidth=2, label="3期滚动均值")
    ax1.axvline(SPLIT_DATE, color="gray", linestyle="--", linewidth=2, label="切分点 2025-01-01")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(0.03, color="green", linestyle=":", linewidth=1)
    ax1.axhline(-0.03, color="green", linestyle=":", linewidth=1)
    ax1.set_title("模型 Rank IC 时序（月度均值）", fontweight="bold")
    ax1.set_ylabel("Rank IC")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("IC 统计汇总", fontweight="bold", pad=10)
    table_df = summary[["区间", "IC均值", "IC标准差", "ICIR", "正IC占比"]].copy()
    for col in ["IC均值", "IC标准差", "ICIR"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")
    table_df["正IC占比"] = table_df["正IC占比"].map(lambda x: f"{x:.2%}")
    table = ax2.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_facecolor("#c7d3e5")
            cell.set_text_props(fontweight="bold")
        elif row == 2:
            cell.set_facecolor("#f8d9d7")

    ax3 = fig.add_subplot(gs[1, 0])
    daily["cum_rank_ic"] = daily["Rank_IC"].fillna(0).cumsum()
    in_daily = daily[daily["trade_date"] < SPLIT_DATE]
    out_daily = daily[daily["trade_date"] >= SPLIT_DATE].copy()
    out_daily["cum_rank_ic_oos"] = out_daily["Rank_IC"].fillna(0).cumsum()
    ax3.plot(in_daily["trade_date"], in_daily["cum_rank_ic"], color="#377eb8", linewidth=2.2, label="样本内累计Rank IC")
    ax3.plot(out_daily["trade_date"], out_daily["cum_rank_ic_oos"], color="#fb4d3d", linewidth=2.2, label="样本外累计Rank IC")
    ax3.axvline(SPLIT_DATE, color="gray", linestyle="--", linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Rank IC 累积和（斜率持续向上 = 排序能力稳定）", fontweight="bold")
    ax3.set_ylabel("Rank IC 累积和")
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    in_mask = daily["trade_date"] < SPLIT_DATE
    out_mask = daily["trade_date"] >= SPLIT_DATE
    ax4.hist(daily.loc[in_mask, "Rank_IC"].dropna(), bins=22, alpha=0.65, color="#74a9cf", label="样本内")
    ax4.hist(daily.loc[out_mask, "Rank_IC"].dropna(), bins=18, alpha=0.55, color="#fb6a4a", label="样本外")
    in_mean = daily.loc[in_mask, "Rank_IC"].mean()
    out_mean = daily.loc[out_mask, "Rank_IC"].mean()
    ax4.axvline(0, color="black", linewidth=1)
    ax4.axvline(in_mean, color="#377eb8", linestyle="--", linewidth=2, label=f"内均值 {in_mean:.4f}")
    ax4.axvline(out_mean, color="#fb4d3d", linestyle="--", linewidth=2, label=f"外均值 {out_mean:.4f}")
    ax4.set_title("Rank IC 分布对比", fontweight="bold")
    ax4.set_xlabel("Rank IC")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.25)

    ax5 = fig.add_subplot(gs[2, 0])
    oos = daily[daily["trade_date"] >= SPLIT_DATE].copy()
    roll_mean = oos["Rank_IC"].rolling(21, min_periods=5).mean()
    roll_std = oos["Rank_IC"].rolling(21, min_periods=5).std()
    ax5.plot(oos["trade_date"], roll_mean, color="#fb4d3d", linewidth=2.2, label="21日滚动Rank IC均值")
    ax5.fill_between(oos["trade_date"], roll_mean - roll_std, roll_mean + roll_std, color="#fb6a4a", alpha=0.18, label="±1σ区间")
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.axhline(0.03, color="green", linestyle=":", linewidth=1, label="±0.03基准线")
    ax5.axhline(-0.03, color="green", linestyle=":", linewidth=1)
    ax5.set_title("样本外 Rank IC 滚动均值 ± 1σ 区间", fontweight="bold")
    ax5.set_ylabel("Rank IC")
    ax5.legend(loc="lower left")
    ax5.grid(alpha=0.25)

    ax6 = fig.add_subplot(gs[2, 1])
    top = importance.head(15).sort_values("importance_split", ascending=True)
    market_features = {
        "index_ret_1d",
        "index_volatility_20",
        "index_drawdown_20",
        "beta_to_index_60",
        "corr_to_index_60",
    }
    colors = ["#377eb8" if feature in market_features else "#ff8c1a" for feature in top["feature"]]
    ax6.barh(top["feature"], top["importance_split"], color=colors, alpha=0.9)
    ax6.set_title("因子重要性 Top15（蓝色=市场/风险，橙色=个股量价）", fontweight="bold")
    ax6.set_xlabel("Importance（分裂次数）")
    ax6.grid(axis="x", alpha=0.25)

    fig.savefig(REPORT_PATH, dpi=180, bbox_inches="tight")
    fig.savefig(REPORT_PDF_PATH, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("1. Reading LightGBM predictions...")
    pred = pd.read_parquet(PREDICTION_PATH).dropna(subset=["pred_alpha", "label_5d"])
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])

    print("2. Calculating daily/monthly IC...")
    daily = build_daily_ic(pred)
    monthly = daily.groupby("month", as_index=False).agg({"IC": "mean", "Rank_IC": "mean", "n_stocks": "mean"})
    summary = build_summary(daily)

    print("3. Training full-sample LightGBM for feature importance...")
    importance = build_feature_importance()

    print("4. Saving chart data...")
    daily.to_csv(REPORT_DIR / "lightgbm_daily_ic.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(REPORT_DIR / "lightgbm_monthly_ic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT_DIR / "lightgbm_ic_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(REPORT_DIR / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("5. Drawing IC report...")
    setup_chinese_font()
    draw_report(daily, monthly, summary, importance, show_plot=True)

    print("\nIC summary:")
    print(summary.to_string(index=False))
    print(f"\nImage saved to: {REPORT_PATH}")
    print(f"PDF saved to: {REPORT_PDF_PATH}")
    print(f"Data saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
