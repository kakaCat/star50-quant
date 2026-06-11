import pandas as pd 
import numpy as np 
import time
import warnings 
import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset 

warnings.filterwarnings('ignore')

FORWARD_DAYS = 5

DATA_CACHE_PATH = 'star50_daily_hfq_data_6yrs.parquet'
INDEX_CACHE_PATH = 'star50_index_daily_6yrs.parquet'

# PyTorch 超参数
EPOCHS = 25
BATCH_SIZE = 1024
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================
# 3. 特征工程与 Label 构造 (修复特征标准化逻辑)
# ==========================================
def build_features_and_labels(stock_df, index_df):
    print("正在构建环境特征与高维量价特征...")

    # ---------------- A. 构建大盘环境特征 ----------------
    index_df = index_df.sort_values('trade_date').reset_index(drop=True)
    index_df['idx_ret_1d'] = index_df['close'].pct_change(1)
    
    index_df['idx_bias_20d'] = index_df['close'] / index_df['close'].rolling(20).mean() - 1
    index_df['idx_bias_60d'] = index_df['close'] / index_df['close'].rolling(60).mean() - 1
    index_df['idx_vol_20d'] = index_df['idx_ret_1d'].rolling(20).std()
    index_df['idx_vol_mom'] = index_df['vol'] / index_df['vol'].rolling(60).mean() - 1
    index_df['idx_future_ret'] = index_df['close'].shift(-FORWARD_DAYS) / index_df['open'].shift(-1) - 1
    
    idx_cols = ['trade_date', 'idx_ret_1d', 'idx_bias_20d', 'idx_bias_60d', 'idx_vol_20d', 'idx_vol_mom', 'idx_future_ret']
    
    # ---------------- B. 构建个股量价特征 ----------------
    stock_df = stock_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    grouped = stock_df.groupby('ts_code')

    stock_df['ret_1d'] = grouped['hfq_close'].pct_change(1)
    stock_df['hl_spread'] = (stock_df['hfq_high'] - stock_df['hfq_low']) / stock_df['hfq_close']
    stock_df['gap_open'] = stock_df['hfq_open'] / grouped['hfq_close'].shift(1) - 1

    stock_features = ['hl_spread', 'gap_open']
    windows = [3, 5, 10, 20]
    for window in windows:
        col_mom = f'mom_{window}d'
        stock_df[col_mom] = grouped['hfq_close'].pct_change(window)
        stock_features.append(col_mom)

        col_vol = f'vol_{window}d'
        stock_df[col_vol] = grouped['ret_1d'].transform(lambda x: x.rolling(window).std())
        stock_features.append(col_vol)

        col_skew = f'skew_{window}d'
        stock_df[col_skew] = grouped['ret_1d'].transform(lambda x: x.rolling(window).skew())
        stock_features.append(col_skew)

        col_vol_ma = f'vol_ma_{window}d'
        stock_df[col_vol_ma] = grouped['vol'].transform(lambda x: x.rolling(window).mean())
        col_vol_mom = f'vol_mom_{window}d'
        stock_df[col_vol_mom] = stock_df['vol'] / stock_df[col_vol_ma] - 1 
        stock_features.append(col_vol_mom)

        col_pv_corr = f'pv_corr_{window}d'
        stock_df[col_pv_corr] = grouped.apply(lambda x: x['ret_1d'].rolling(window).corr(x['vol'])).reset_index(level=0, drop=True)
        stock_features.append(col_pv_corr)

    stock_df['bias_20d'] = stock_df['hfq_close'] / grouped['hfq_close'].transform(lambda x: x.rolling(20).mean()) - 1
    stock_features.append('bias_20d')
    
    # 截面散度是在个股层面上计算的宏观/环境特征
    stock_df['cs_dispersion'] = stock_df.groupby('trade_date')['ret_1d'].transform('std')
    regime_features = ['idx_bias_20d', 'idx_bias_60d', 'idx_vol_20d', 'idx_vol_mom', 'cs_dispersion']

    print("正在对【个股特征】执行横截面正态化 (Cross-Sectional Z-Score)...")
    # 修正点 1：只对个股特征做横截面正态化，大盘特征不能做截面正态化
    for f in stock_features:
        stock_df[f] = stock_df.groupby('trade_date')[f].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        stock_df[f] = stock_df[f].fillna(0) 

    # ---------------- C. 剥离真实 Beta 与重构 Alpha Label ----------------
    print("正在合并环境特征，计算滚动 60 日 Beta 并剥离真实残差 (True Residual)...")
    df = pd.merge(stock_df, index_df[idx_cols], on='trade_date', how='inner')
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 修正点 2：在合并大盘数据后，对环境特征（宏观时序特征）执行全局/时序正态化
    print("正在对【环境特征】执行时序正态化 (Time-Series Z-Score)...")
    for f in regime_features:
        df[f] = (df[f] - df[f].mean()) / (df[f].std() + 1e-8)
        df[f] = df[f].fillna(0)

    # 计算 Beta 与残差
    df['cov_60d'] = df.groupby('ts_code', group_keys=False).apply(lambda g: g['ret_1d'].rolling(60).cov(g['idx_ret_1d']))
    df['idx_var_60d'] = df.groupby('ts_code')['idx_ret_1d'].transform(lambda x: x.rolling(60).var())
    df['beta_60d'] = (df['cov_60d'] / (df['idx_var_60d'] + 1e-8)).fillna(1.0).clip(-2, 3)

    df['stock_future_ret'] = df.groupby('ts_code')['hfq_close'].shift(-FORWARD_DAYS) / df.groupby('ts_code')['hfq_open'].shift(-1) - 1
    df['residual_ret'] = df['stock_future_ret'] - df['beta_60d'] * df['idx_future_ret']

    df = df.dropna(subset=['residual_ret'])

    # Label 分桶映射
    def qcut_labels(x):
        if len(x) < 5:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return pd.qcut(x, q=5, labels=False, duplicates='drop')
    
    df['label'] = df.groupby('trade_date')['residual_ret'].apply(qcut_labels).reset_index(level=0, drop=True)
    df = df.dropna(subset=['label'])
    
    # 转换为 [0, 1] 区间内的连续值，利于神经网络收敛
    df['label'] = df['label'].astype(float) / 4.0 

    return df, stock_features, regime_features

# ==========================================
# 4. PyTorch Neural MoE 网络定义
# ==========================================
class NeuralMoE(nn.Module):
    def __init__(self, stock_dim, regime_dim):
        super(NeuralMoE, self).__init__()

        # Expert 1: 牛市/偏强环境专家
        self.expert_bull = nn.Sequential(
            nn.Linear(stock_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # Expert 2: 熊市/偏弱环境专家
        self.expert_bear = nn.Sequential(
            nn.Linear(stock_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # Gating Network: 基于宏观环境特征输出各专家权重
        self.gate = nn.Sequential(
            nn.Linear(regime_dim, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1) # 保证 w_bull + w_bear = 1
        )

    def forward(self, x_stock, x_regime):
        # 1. 获取专家打分
        score_bull = self.expert_bull(x_stock)
        score_bear = self.expert_bear(x_stock)

        # 2. 获取门控权重
        weights = self.gate(x_regime)
        w_bull = weights[:, 0].unsqueeze(1)
        w_bear = weights[:, 1].unsqueeze(1)

        # 3. 动态融合打分
        final_score = w_bull * score_bull + w_bear * score_bear 
        return final_score
    
# ==========================================
# 5. DL MoE 滚动训练与评估
# ==========================================
def train_dl_moe_walk_forward(df, stock_features, regime_features):
    print(f"启动 Phase 1: PyTorch 端到端混合专家模型 (MoE) 滚动训练... 运行设备: {DEVICE}")
    df = df.sort_values(['trade_date', 'ts_code'])
    df['year_month'] = df['trade_date'].dt.to_period('M')
    months = np.sort(df['year_month'].unique())

    train_window = 24
    test_window = 1
    out_of_sample_preds = []

    stock_dim = len(stock_features)
    regime_dim = len(regime_features)

    for i in range(train_window, len(months), test_window):
        train_months = months[i - train_window: i]
        test_months = months[i : i + test_window]
        
        train_df = df[df['year_month'].isin(train_months)]
        test_df = df[df['year_month'].isin(test_months)]

        if len(test_df) == 0: continue
        print(f"训练窗口: {train_months[0]} -> {train_months[-1]} | 预测: {test_months[0]}")

        # 构建 Tensor
        X_stock_tr = torch.tensor(train_df[stock_features].values, dtype=torch.float32).to(DEVICE)
        X_regime_tr = torch.tensor(train_df[regime_features].values, dtype=torch.float32).to(DEVICE)
        Y_tr = torch.tensor(train_df['label'].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        X_stock_te = torch.tensor(test_df[stock_features].values, dtype=torch.float32).to(DEVICE)
        X_regime_te = torch.tensor(test_df[regime_features].values, dtype=torch.float32).to(DEVICE)

        train_dataset = TensorDataset(X_stock_tr, X_regime_tr, Y_tr)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 实例化网络与优化器
        model = NeuralMoE(stock_dim, regime_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.MSELoss() # 对于分桶 Label，MSE 回归足以进行序关系学习

        # 训练过程
        model.train()
        for epoch in range(EPOCHS):
            for batch_x_s, batch_x_r, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x_s, batch_x_r)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        # 推理评估过程
        model.eval()
        with torch.no_grad():
            preds_te = model(X_stock_te, X_regime_te)
            pred_scores = preds_te.cpu().numpy().flatten()

        test_df_copy = test_df.copy()
        test_df_copy['pred_score'] = pred_scores
        out_of_sample_preds.append(test_df_copy)

    final_test_df = pd.concat(out_of_sample_preds, ignore_index=True)
    return final_test_df

# ==========================================
# 6. 极简回测验证
# ==========================================
def simple_eval(test_df):
    print("\n========== 6年全周期 Out-of-Sample 排序效果评估 (Deep Learning MoE) ==========")
    test_df['pred_score_reverse'] = -test_df['pred_score']

    ic_list = []
    ic_rev_list = []

    for date, group in test_df.groupby('trade_date'):
        if len(group) > 2:
            ic = group['pred_score'].corr(group['residual_ret'], method='spearman')
            ic_list.append(ic)

            ic_rev = group['pred_score_reverse'].corr(group['residual_ret'], method='spearman')
            ic_rev_list.append(ic_rev)

    rank_ic = np.nanmean(ic_list)
    ic_ir = rank_ic / np.nanstd(ic_list) if np.nanstd(ic_list) != 0 else 0

    rank_ic_rev = np.nanmean(ic_rev_list)
    ic_ir_rev = rank_ic_rev / np.nanstd(ic_rev_list) if np.nanstd(ic_rev_list) != 0 else 0 

    print(f"【真实Alpha正向模型】测试集 残差Rank IC: {rank_ic:.4f}, ICIR: {ic_ir:.4f}")
    print(f"【真实Alpha反向策略】测试集 残差Rank IC: {rank_ic_rev:.4f}, ICIR: {ic_ir_rev:.4f}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':

    index_df = pd.read_parquet(INDEX_CACHE_PATH)

    raw_df = pd.read_parquet(DATA_CACHE_PATH)
    
    processed_df, stock_feats, regime_feats = build_features_and_labels(raw_df, index_df)

    predictions_df = train_dl_moe_walk_forward(processed_df, stock_feats, regime_feats)

    simple_eval(predictions_df)