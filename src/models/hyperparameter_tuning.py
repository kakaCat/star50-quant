"""
超参数优化框架
==============

支持三种调参方法：
1. 随机搜索 (Random Search)
2. 网格搜索 (Grid Search)
3. 贝叶斯优化 (Bayesian Optimization)

目标函数综合评估：
- IC (Information Coefficient): 预测能力
- IR (Information Ratio): 风险调整后的超额收益
- 年化收益: 策略盈利能力
- 最大回撤: 风险控制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid, ParameterSampler
from scipy.stats import uniform, randint
import optuna
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TuningResult:
    """调参结果数据类"""
    best_params: Dict
    best_score: float
    all_trials: pd.DataFrame
    search_method: str
    timestamp: str

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_method': self.search_method,
            'timestamp': self.timestamp,
            'n_trials': len(self.all_trials)
        }

    def save(self, filepath: str):
        """保存结果到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # 保存详细试验结果
        trials_path = filepath.replace('.json', '_trials.csv')
        self.all_trials.to_csv(trials_path, index=False)
        print(f"Results saved to {filepath}")
        print(f"Trials saved to {trials_path}")


class ObjectiveFunction:
    """
    目标函数类

    综合评估模型的IC、IR、年化收益和最大回撤
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        n_splits: int = 5,
        ic_weight: float = 0.4,
        ir_weight: float = 0.3,
        return_weight: float = 0.2,
        drawdown_weight: float = 0.1
    ):
        """
        初始化目标函数

        Args:
            features: 特征数据
            labels: 标签数据
            n_splits: 交叉验证折数
            ic_weight: IC权重
            ir_weight: IR权重
            return_weight: 年化收益权重
            drawdown_weight: 回撤权重（负向指标）
        """
        self.features = features
        self.labels = labels
        self.n_splits = n_splits

        # 权重归一化
        total_weight = ic_weight + ir_weight + return_weight + drawdown_weight
        self.ic_weight = ic_weight / total_weight
        self.ir_weight = ir_weight / total_weight
        self.return_weight = return_weight / total_weight
        self.drawdown_weight = drawdown_weight / total_weight

        # 准备数据
        feature_cols = [col for col in features.columns
                       if col not in ['ts_code', 'factor_date']]
        self.feature_names = feature_cols
        self.X = features[feature_cols].values
        self.y = labels['forward_return'].values

    def evaluate_params(self, params: Dict) -> Dict[str, float]:
        """
        评估参数组合

        Args:
            params: XGBoost参数字典

        Returns:
            评估指标字典 {ic, ir, annual_return, max_drawdown, composite_score}
        """
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        ic_scores = []
        predictions_all = []
        actuals_all = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 训练模型
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get('num_boost_round', 100),
                evals=[(dval, 'valid')],
                early_stopping_rounds=10,
                verbose_eval=0
            )

            # 预测
            y_pred = model.predict(dval)

            # 计算IC
            ic = np.corrcoef(y_pred, y_val)[0, 1]
            if not np.isnan(ic):
                ic_scores.append(ic)

            predictions_all.extend(y_pred)
            actuals_all.extend(y_val)

        # 计算综合指标
        predictions_all = np.array(predictions_all)
        actuals_all = np.array(actuals_all)

        # IC (Information Coefficient) - 整体预测能力
        ic_mean = np.mean(ic_scores) if ic_scores else 0.0
        ic_std = np.std(ic_scores) if ic_scores else 1.0

        # 模拟策略收益（简化版：按预测排序，做多top 20%）
        returns_sim = self._simulate_strategy(predictions_all, actuals_all)

        # 计算策略的超额收益（假设无风险利率为0，基准收益为0）
        excess_returns = returns_sim  # 简化：假设超额收益=策略收益

        # IR (Information Ratio) - 正确定义：超额收益的均值/标准差
        # 注意：这里是年化IR
        if len(excess_returns) > 1:
            excess_mean = np.mean(excess_returns)
            excess_std = np.std(excess_returns, ddof=1)  # 样本标准差
            # 假设returns_sim是窗口收益（每50个样本一个窗口）
            # 年化：假设一年约252个交易日，每个窗口约代表几天
            periods_per_year = 252 / 50  # 粗略估计
            ir = (excess_mean * periods_per_year) / (excess_std * np.sqrt(periods_per_year)) if excess_std > 0 else 0.0
        else:
            ir = 0.0

        # 年化收益 - 正确计算（考虑复利）
        if len(returns_sim) > 0:
            cumulative_return = np.prod(1 + returns_sim) - 1
            n_periods = len(returns_sim)
            # 假设每个窗口代表约50个样本（约50天）
            n_days = n_periods * 50
            if n_days > 0:
                annual_return = (1 + cumulative_return) ** (252 / n_days) - 1
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0

        # 最大回撤
        cumulative = (1 + returns_sim).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 综合评分（0-1范围）
        # IC: 目标>0.04, 归一化到0-1
        ic_score = min(max(ic_mean / 0.08, 0), 1)  # 0.08为优秀水平

        # IR: 目标>=1.5, 归一化到0-1
        ir_score = min(max(ir / 3.0, 0), 1)  # 3.0为优秀水平

        # 年化收益: 目标>35%, 归一化到0-1
        return_score = min(max(annual_return / 0.7, 0), 1)  # 70%为优秀水平

        # 最大回撤: 目标<=-20%, 越小越好，归一化到0-1
        drawdown_score = min(max(1 + max_drawdown / 0.4, 0), 1)  # -40%为可接受底线

        # 加权综合评分
        composite_score = (
            self.ic_weight * ic_score +
            self.ir_weight * ir_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score
        )

        return {
            'ic': ic_mean,
            'ir': ir,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'composite_score': composite_score
        }

    def _simulate_strategy(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """
        简化策略模拟：按预测排序，做多top 20%

        注意：这是一个简化的模拟，将所有样本视为一个时间序列来计算回撤

        Args:
            predictions: 预测值
            actuals: 实际收益率

        Returns:
            策略日收益率序列
        """
        # 将预测值和实际收益按时间顺序分组
        # 假设数据是按时间顺序排列的（通过交叉验证的索引保持了顺序）

        # 每个时间点选择top 20%的股票
        # 由于这里是混合了多个fold的数据，我们采用滚动窗口方式模拟
        window_size = 50  # 每50个样本作为一个时间窗口
        returns_series = []

        for i in range(0, len(predictions), window_size):
            end_idx = min(i + window_size, len(predictions))
            window_pred = predictions[i:end_idx]
            window_actual = actuals[i:end_idx]

            if len(window_pred) == 0:
                continue

            # 选择top 20%
            n_top = max(int(len(window_pred) * 0.2), 1)
            top_indices = np.argsort(window_pred)[-n_top:]

            # 计算这个窗口的收益
            window_return = np.mean(window_actual[top_indices])
            returns_series.append(window_return)

        return np.array(returns_series)


class HyperparameterTuner:
    """
    超参数调优器

    支持三种搜索方法：随机搜索、网格搜索、贝叶斯优化
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        param_space: Dict
    ):
        """
        初始化调优器

        Args:
            objective_fn: 目标函数实例
            param_space: 参数搜索空间
        """
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.trials = []

    def random_search(
        self,
        n_iter: int = 50,
        random_state: int = 42
    ) -> TuningResult:
        """
        随机搜索

        从参数空间中随机采样，评估性能

        Args:
            n_iter: 迭代次数
            random_state: 随机种子

        Returns:
            TuningResult对象
        """
        print("="*60)
        print("随机搜索 (Random Search)")
        print("="*60)
        print(f"迭代次数: {n_iter}")
        print(f"参数空间: {self.param_space}")

        np.random.seed(random_state)

        # 转换参数空间为scipy分布
        param_distributions = {}
        for key, value in self.param_space.items():
            if isinstance(value, list):
                param_distributions[key] = value
            elif isinstance(value, tuple) and len(value) == 2:
                # (min, max) -> uniform distribution
                param_distributions[key] = uniform(value[0], value[1] - value[0])

        # 随机采样
        sampler = ParameterSampler(
            param_distributions,
            n_iter=n_iter,
            random_state=random_state
        )

        best_score = -np.inf
        best_params = None
        trials = []

        for i, params in enumerate(sampler, 1):
            print(f"\n[{i}/{n_iter}] Testing params: {params}")

            # 转换参数类型
            params = self._convert_params(params)

            # 评估
            scores = self.objective_fn.evaluate_params(params)

            print(f"  IC={scores['ic']:.4f}, IR={scores['ir']:.4f}, "
                  f"Return={scores['annual_return']:.2%}, DD={scores['max_drawdown']:.2%}")
            print(f"  Composite Score: {scores['composite_score']:.4f}")

            # 记录
            trial = {**params, **scores}
            trials.append(trial)

            # 更新最佳
            if scores['composite_score'] > best_score:
                best_score = scores['composite_score']
                best_params = params.copy()
                print(f"  ✓ New best score!")

        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=pd.DataFrame(trials),
            search_method='random_search',
            timestamp=datetime.now().isoformat()
        )

        self._print_summary(result)
        return result

    def grid_search(self) -> TuningResult:
        """
        网格搜索

        遍历参数空间的所有组合

        Returns:
            TuningResult对象
        """
        print("="*60)
        print("网格搜索 (Grid Search)")
        print("="*60)

        # 转换参数空间为列表
        param_grid = {}
        for key, value in self.param_space.items():
            if isinstance(value, list):
                param_grid[key] = value
            elif isinstance(value, tuple) and len(value) == 3:
                # (min, max, step) -> list
                param_grid[key] = list(np.arange(value[0], value[1], value[2]))
            else:
                param_grid[key] = [value]

        # 生成所有组合
        grid = list(ParameterGrid(param_grid))
        n_combinations = len(grid)

        print(f"总组合数: {n_combinations}")
        print(f"参数空间: {param_grid}")

        if n_combinations > 100:
            print(f"警告: 组合数过多({n_combinations})，建议使用随机搜索或贝叶斯优化")

        best_score = -np.inf
        best_params = None
        trials = []

        for i, params in enumerate(grid, 1):
            print(f"\n[{i}/{n_combinations}] Testing params: {params}")

            # 转换参数类型
            params = self._convert_params(params)

            # 评估
            scores = self.objective_fn.evaluate_params(params)

            print(f"  IC={scores['ic']:.4f}, IR={scores['ir']:.4f}, "
                  f"Return={scores['annual_return']:.2%}, DD={scores['max_drawdown']:.2%}")
            print(f"  Composite Score: {scores['composite_score']:.4f}")

            # 记录
            trial = {**params, **scores}
            trials.append(trial)

            # 更新最佳
            if scores['composite_score'] > best_score:
                best_score = scores['composite_score']
                best_params = params.copy()
                print(f"  ✓ New best score!")

        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=pd.DataFrame(trials),
            search_method='grid_search',
            timestamp=datetime.now().isoformat()
        )

        self._print_summary(result)
        return result

    def bayesian_optimization(
        self,
        n_trials: int = 50,
        n_startup_trials: int = 10
    ) -> TuningResult:
        """
        贝叶斯优化

        使用Optuna框架进行基于TPE（Tree-structured Parzen Estimator）的优化

        Args:
            n_trials: 试验次数
            n_startup_trials: 随机初始化试验数

        Returns:
            TuningResult对象
        """
        print("="*60)
        print("贝叶斯优化 (Bayesian Optimization with TPE)")
        print("="*60)
        print(f"试验次数: {n_trials}")
        print(f"初始随机试验: {n_startup_trials}")

        def objective(trial):
            """Optuna目标函数"""
            params = {}

            for key, value in self.param_space.items():
                if isinstance(value, list):
                    if all(isinstance(v, int) for v in value):
                        params[key] = trial.suggest_int(key, min(value), max(value))
                    elif all(isinstance(v, float) for v in value):
                        params[key] = trial.suggest_float(key, min(value), max(value))
                    else:
                        params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple) and len(value) == 2:
                    if isinstance(value[0], int):
                        params[key] = trial.suggest_int(key, value[0], value[1])
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])

            # 转换参数类型
            params = self._convert_params(params)

            # 评估
            scores = self.objective_fn.evaluate_params(params)

            # 记录额外指标
            trial.set_user_attr('ic', scores['ic'])
            trial.set_user_attr('ir', scores['ir'])
            trial.set_user_attr('annual_return', scores['annual_return'])
            trial.set_user_attr('max_drawdown', scores['max_drawdown'])

            return scores['composite_score']

        # 创建study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                seed=42
            )
        )

        # 优化
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # 收集结果
        trials = []
        for trial in study.trials:
            trial_data = {
                **trial.params,
                'ic': trial.user_attrs.get('ic', 0),
                'ir': trial.user_attrs.get('ir', 0),
                'annual_return': trial.user_attrs.get('annual_return', 0),
                'max_drawdown': trial.user_attrs.get('max_drawdown', 0),
                'composite_score': trial.value
            }
            trials.append(trial_data)

        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=pd.DataFrame(trials),
            search_method='bayesian_optimization',
            timestamp=datetime.now().isoformat()
        )

        self._print_summary(result)

        # 打印重要性分析
        print("\n参数重要性分析:")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param}: {importance:.4f}")
        except:
            print("  (无法计算参数重要性)")

        return result

    def _convert_params(self, params: Dict) -> Dict:
        """转换参数类型（整数、浮点数）"""
        converted = {}
        for key, value in params.items():
            if key in ['max_depth', 'num_boost_round', 'min_child_weight']:
                converted[key] = int(value)
            else:
                converted[key] = value

        # 添加固定参数
        converted.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'seed': 42,
            'verbosity': 0
        })

        return converted

    def _print_summary(self, result: TuningResult):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("调参完成!")
        print("="*60)
        print(f"\n最佳参数:")
        for key, value in result.best_params.items():
            print(f"  {key}: {value}")

        print(f"\n最佳综合评分: {result.best_score:.4f}")

        # 获取最佳试验的详细指标
        best_trial = result.all_trials.loc[result.all_trials['composite_score'].idxmax()]
        print(f"\n详细指标:")
        print(f"  IC: {best_trial['ic']:.4f} (目标 > 0.04)")
        print(f"  IR: {best_trial['ir']:.4f} (目标 >= 1.5)")
        print(f"  年化收益: {best_trial['annual_return']:.2%} (目标 > 35%)")
        print(f"  最大回撤: {best_trial['max_drawdown']:.2%} (目标 <= -20%)")

        # 目标达成情况
        print(f"\n目标达成情况:")
        ic_ok = "✓" if best_trial['ic'] > 0.04 else "✗"
        ir_ok = "✓" if best_trial['ir'] >= 1.5 else "✗"
        ret_ok = "✓" if best_trial['annual_return'] > 0.35 else "✗"
        dd_ok = "✓" if best_trial['max_drawdown'] >= -0.20 else "✗"

        print(f"  {ic_ok} IC > 0.04")
        print(f"  {ir_ok} IR >= 1.5")
        print(f"  {ret_ok} 年化收益 > 35%")
        print(f"  {dd_ok} 最大回撤 <= 20%")
