"""
特征工程模块
============

将30个原始技术因子扩展到160+增强特征。

策略:
1. 因子交叉特征 (+50)
2. 时序衍生特征 (+30)
3. 非线性变换 (+20)
4. 截面统计特征 (+15)
5. Alpha因子 (+15)
"""

import pandas as pd
import numpy as np
import yaml
from typing import List, Dict
from pathlib import Path

from src.features.alpha_factors import AlphaFactorCalculator


class FeatureEngineer:
    """
    特征工程器

    从30个原始因子扩展到160个特征。
    """

    def __init__(self, config_path: str = 'configs/feature_config.yaml'):
        """
        初始化

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.feature_names = []
        self.alpha_calculator = AlphaFactorCalculator()

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        主入口：完整特征转换

        Args:
            df: 输入DataFrame（包含30个原始因子）

        Returns:
            增强后的DataFrame（160+特征）
        """
        print("="*60)
        print("特征工程：30因子 → 160+特征")
        print("="*60)

        df_enhanced = df.copy()
        original_feature_count = len([c for c in df.columns if c not in ['ts_code', 'factor_date']])

        # 1. 因子交叉 (+50)
        print("\n[1/5] 生成因子交叉特征...")
        df_enhanced = self.add_cross_features(df_enhanced)

        # 2. 时序衍生 (+30)
        print("[2/5] 生成时序衍生特征...")
        df_enhanced = self.add_temporal_features(df_enhanced)

        # 3. 非线性变换 (+20)
        print("[3/5] 生成非线性变换特征...")
        df_enhanced = self.add_nonlinear_features(df_enhanced)

        # 4. 截面统计 (+15)
        print("[4/5] 生成截面统计特征...")
        df_enhanced = self.add_cross_sectional_features(df_enhanced)

        # 5. Alpha因子 (+15)
        print("[5/5] 生成Alpha因子...")
        df_enhanced = self.add_alpha_factors(df_enhanced)

        # 收集特征名称
        self.feature_names = [c for c in df_enhanced.columns if c not in ['ts_code', 'factor_date']]

        final_feature_count = len(self.feature_names)
        print(f"\n特征扩展完成: {original_feature_count} → {final_feature_count} (+{final_feature_count - original_feature_count})")

        return df_enhanced

    def add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加因子交叉特征 (+50)

        包括:
        - 量价交叉 (20)
        - MACD×量能 (15)
        - 趋势×动量 (15)
        """
        result = df.copy()
        config = self.config['feature_engineering']['cross_features']

        # 1. 动量×量能交叉
        for factor1, factor2 in config['momentum_volume']:
            if factor1 in result.columns and factor2 in result.columns:
                col_name = f'{factor1}_x_{factor2}'
                result[col_name] = result[factor1] * result[factor2]

        # 2. MACD×量能交叉
        for factor1, factor2 in config['macd_volume']:
            if factor1 in result.columns and factor2 in result.columns:
                col_name = f'{factor1}_x_{factor2}'
                result[col_name] = result[factor1] * result[factor2]

        # 3. 动量比率
        for factor1, factor2, op in config['momentum_ratio']:
            if factor1 in result.columns and factor2 in result.columns:
                if op == 'div':
                    col_name = f'{factor1}_div_{factor2}'
                    result[col_name] = result[factor1] / (result[factor2] + 1e-6)

        # 4. 趋势×动量交叉
        for factor1, factor2, op in config['trend_momentum']:
            if factor1 in result.columns and factor2 in result.columns:
                if op == 'slope_mult':
                    # MA斜率 × 动量
                    ma_slope = result.groupby('ts_code')[factor1].pct_change(5)
                    col_name = f'{factor1}_slope_x_{factor2}'
                    result[col_name] = ma_slope * result[factor2]

        # 5. 波动率调整
        for factor, volatility, op in config['volatility_adjusted']:
            if factor in result.columns and volatility in result.columns:
                if op == 'div':
                    col_name = f'{factor}_div_{volatility}'
                    result[col_name] = result[factor] / (result[volatility] + 1e-6)

        return result

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时序衍生特征 (+30)

        包括:
        - 因子动量 (15)
        - 因子波动 (15)
        """
        result = df.copy()
        config = self.config['feature_engineering']

        # 使用top_factors
        top_factors = config['top_factors'][:15]
        momentum_periods = config['temporal_features']['momentum_periods']
        volatility_periods = config['temporal_features']['volatility_periods']

        for factor in top_factors:
            if factor not in result.columns:
                continue

            # 因子动量（加1天滞后避免未来信息）
            for period in momentum_periods:
                col_name = f'{factor}_momentum_{period}d'
                result[col_name] = result.groupby('ts_code')[factor].pct_change(period).shift(1)

            # 因子波动（只用20天，加1天滞后）
            for period in volatility_periods:
                col_name = f'{factor}_volatility_{period}d'
                result[col_name] = result.groupby('ts_code')[factor].transform(
                    lambda x: x.rolling(period, min_periods=1).std()
                ).shift(1)

        return result

    def add_nonlinear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加非线性变换特征 (+20)

        包括:
        - Log变换 (10)
        - Rank归一化 (10)
        """
        result = df.copy()
        config = self.config['feature_engineering']['nonlinear_transforms']

        # Log变换
        for factor in config['log_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_log'
            result[col_name] = np.sign(result[factor]) * np.log1p(np.abs(result[factor]))

        # Rank归一化
        for factor in config['rank_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_rank'
            result[col_name] = result.groupby('factor_date')[factor].rank(pct=True)

        return result

    def add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加截面统计特征 (+15)

        包括:
        - 分位数 (7)
        - Z-score (8)
        """
        result = df.copy()
        config = self.config['feature_engineering']['cross_sectional']

        # 分位数特征
        for factor in config['quantile_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_quantile'
            result[col_name] = result.groupby('factor_date')[factor].rank(pct=True)

        # Z-score特征
        top_factors = self.config['feature_engineering']['top_factors'][:10]
        for factor in top_factors:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_zscore'
            grouped = result.groupby('factor_date')[factor]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            result[col_name] = (result[factor] - mean) / (std + 1e-6)

        return result

    def add_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加Alpha因子 (+10-15)

        使用AlphaFactorCalculator计算经典Alpha因子
        """
        # 需要确保数据有必要的列
        required_cols = ['ts_code', 'close', 'open', 'high', 'low', 'volume']

        # 如果有trade_date但没有factor_date，临时映射
        if 'trade_date' not in df.columns and 'factor_date' in df.columns:
            df_with_trade_date = df.copy()
            df_with_trade_date['trade_date'] = df_with_trade_date['factor_date']
        else:
            df_with_trade_date = df.copy()

        # 计算Alpha因子
        config = self.config['feature_engineering']['alpha_factors']
        alpha_list = config['enabled']

        result = self.alpha_calculator.calculate_batch(df_with_trade_date, alpha_list)

        # 删除临时列
        if 'trade_date' in result.columns and 'trade_date' not in df.columns:
            result = result.drop('trade_date', axis=1)

        return result

    def get_feature_names(self) -> List[str]:
        """
        返回所有特征名称

        Returns:
            特征名称列表
        """
        return self.feature_names

    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        验证特征质量

        检查:
        - NaN比例
        - Inf值
        - 特征相关性

        Returns:
            验证结果字典
        """
        validation_config = self.config.get('validation', {})

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        results = {
            'total_features': len(numeric_cols),
            'nan_ratio': {},
            'inf_count': {},
            'high_correlation_pairs': []
        }

        # 检查NaN
        if validation_config.get('check_nan', True):
            for col in numeric_cols:
                nan_ratio = df[col].isna().mean()
                if nan_ratio > validation_config.get('max_missing_ratio', 0.1):
                    results['nan_ratio'][col] = nan_ratio

        # 检查Inf
        if validation_config.get('check_inf', True):
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    results['inf_count'][col] = inf_count

        # 检查高相关性
        corr_threshold = validation_config.get('correlation_threshold', 0.95)
        corr_matrix = df[numeric_cols].corr().abs()

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    results['high_correlation_pairs'].append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return results
