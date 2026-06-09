"""多数据源管理器 - 提供自动故障转移和容错能力

支持的数据源：
1. akshare (优先级最高) - 免费、功能全面
2. baostock (备用) - 稳定的历史数据

特性：
- 自动故障转移：当主数据源失败时自动切换到备用源
- 熔断器机制：防止持续调用失败的数据源
- 简单缓存：减少重复请求
"""

import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """熔断器 - 防止持续调用失败的数据源"""

    def __init__(self, failure_threshold: int = 3, timeout_seconds: int = 60):
        """
        Args:
            failure_threshold: 连续失败多少次后触发熔断
            timeout_seconds: 熔断后等待多久再尝试
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self):
        """记录成功，重置计数器"""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"熔断器打开，连续失败 {self.failure_count} 次")

    def can_attempt(self) -> bool:
        """判断是否可以尝试请求"""
        if not self.is_open:
            return True

        # 检查是否已经过了超时时间
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout_seconds:
                logger.info(f"熔断器恢复尝试，已等待 {elapsed:.0f} 秒")
                self.is_open = False
                self.failure_count = 0
                return True

        return False


class MultiSourceCollector:
    """多数据源采集器 - 自动故障转移"""

    def __init__(self):
        self.data_source = 'akshare'  # 默认使用 akshare
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            'akshare': CircuitBreaker(failure_threshold=3, timeout_seconds=60),
            'baostock': CircuitBreaker(failure_threshold=3, timeout_seconds=60),
        }
        self.stats = {
            'akshare': {'success': 0, 'failure': 0},
            'baostock': {'success': 0, 'failure': 0},
        }

        # 简单内存缓存（股票代码列表）
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 300  # 5分钟缓存

    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < self._cache_ttl:
                return self._cache[key]
            else:
                # 缓存过期，删除
                del self._cache[key]
                del self._cache_time[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        self._cache[key] = value
        self._cache_time[key] = time.time()

    def get_star50_components(self) -> List[str]:
        """获取科创50成分股代码

        自动故障转移：akshare -> baostock

        Returns:
            成分股代码列表
        """
        cache_key = 'star50_components'
        cached = self._get_cache(cache_key)
        if cached:
            logger.info(f"使用缓存的科创50成分股列表: {len(cached)} 只")
            return cached

        # 尝试 akshare
        if self.circuit_breakers['akshare'].can_attempt():
            try:
                components = self._get_star50_from_akshare()
                if components:
                    self.circuit_breakers['akshare'].record_success()
                    self.stats['akshare']['success'] += 1
                    self._set_cache(cache_key, components)
                    logger.info(f"从 akshare 获取科创50成分股: {len(components)} 只")
                    return components
            except Exception as e:
                logger.error(f"akshare 获取科创50成分股失败: {e}")
                self.circuit_breakers['akshare'].record_failure()
                self.stats['akshare']['failure'] += 1

        # 尝试 baostock（如果实现了的话）
        if self.circuit_breakers['baostock'].can_attempt():
            try:
                components = self._get_star50_from_baostock()
                if components:
                    self.circuit_breakers['baostock'].record_success()
                    self.stats['baostock']['success'] += 1
                    self._set_cache(cache_key, components)
                    logger.info(f"从 baostock 获取科创50成分股: {len(components)} 只")
                    return components
            except Exception as e:
                logger.error(f"baostock 获取科创50成分股失败: {e}")
                self.circuit_breakers['baostock'].record_failure()
                self.stats['baostock']['failure'] += 1

        raise Exception("所有数据源均失败")

    def _get_star50_from_akshare(self) -> List[str]:
        """从 akshare 获取科创50成分股"""
        import akshare as ak

        df = ak.index_stock_cons_csindex(symbol="000688")
        if df is not None and not df.empty:
            # 添加交易所后缀 .SH (科创板都是上海)
            codes = [f"{code}.SH" for code in df['成分券代码'].tolist()]
            return codes
        return []

    def _get_star50_from_baostock(self) -> List[str]:
        """从 baostock 获取科创50成分股（备用方案）"""
        # TODO: 实现 baostock 版本
        # baostock 可能没有指数成分股接口，可以使用硬编码的成分股列表作为备用
        logger.warning("baostock 暂不支持科创50成分股查询")
        return []

    def collect_daily_data(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """采集股票日线数据

        自动故障转移，每只股票独立重试

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame: 包含所有采集成功的股票数据
        """
        all_data = []
        total = len(ts_codes)

        for idx, ts_code in enumerate(ts_codes, 1):
            try:
                # 尝试 akshare
                if self.circuit_breakers['akshare'].can_attempt():
                    try:
                        df = self._collect_from_akshare(ts_code, start_date, end_date)
                        if df is not None and not df.empty:
                            self.circuit_breakers['akshare'].record_success()
                            self.stats['akshare']['success'] += 1
                            all_data.append(df)
                            logger.info(f"[{idx}/{total}] {ts_code}: akshare 采集 {len(df)} 条记录")
                            continue
                    except Exception as e:
                        logger.warning(f"[{idx}/{total}] {ts_code}: akshare 失败 - {e}")
                        self.circuit_breakers['akshare'].record_failure()
                        self.stats['akshare']['failure'] += 1

                # 尝试 baostock
                if self.circuit_breakers['baostock'].can_attempt():
                    try:
                        df = self._collect_from_baostock(ts_code, start_date, end_date)
                        if df is not None and not df.empty:
                            self.circuit_breakers['baostock'].record_success()
                            self.stats['baostock']['success'] += 1
                            all_data.append(df)
                            logger.info(f"[{idx}/{total}] {ts_code}: baostock 采集 {len(df)} 条记录")
                            continue
                    except Exception as e:
                        logger.warning(f"[{idx}/{total}] {ts_code}: baostock 失败 - {e}")
                        self.circuit_breakers['baostock'].record_failure()
                        self.stats['baostock']['failure'] += 1

                # 所有数据源都失败
                logger.error(f"[{idx}/{total}] {ts_code}: 所有数据源均失败")

            except Exception as e:
                logger.error(f"[{idx}/{total}] {ts_code}: 采集异常 - {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"采集完成，共 {len(result)} 条记录")
            return result
        else:
            logger.warning("未采集到任何数据")
            return pd.DataFrame()

    def _collect_from_akshare(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """从 akshare 采集单只股票数据"""
        import akshare as ak

        # 转换代码格式：688009.SH -> 688009
        symbol = ts_code.split('.')[0]

        # 转换日期格式：YYYY-MM-DD -> YYYYMMDD
        start_date_str = start_date.replace('-', '')
        end_date_str = end_date.replace('-', '')

        # 调用 akshare
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start_date_str,
            end_date=end_date_str,
            adjust='qfq'  # 前复权
        )

        if df is None or df.empty:
            return None

        # 转换为标准格式
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['日期']).dt.date
        df['open'] = df['开盘'].astype(float)
        df['high'] = df['最高'].astype(float)
        df['low'] = df['最低'].astype(float)
        df['close'] = df['收盘'].astype(float)
        df['volume'] = df['成交量'].astype(float)
        df['amount'] = df['成交额'].astype(float)

        # 只保留需要的列
        result = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        return result

    def _collect_from_baostock(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """从 baostock 采集单只股票数据"""
        try:
            import baostock as bs
        except ImportError:
            logger.warning("baostock 未安装，请运行: pip install baostock")
            return None

        # 转换代码格式：688009.SH -> sh.688009
        exchange = ts_code.split('.')[1].lower()  # SH -> sh
        symbol = ts_code.split('.')[0]
        bs_code = f"{exchange}.{symbol}"

        # 登录
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return None

        try:
            # 查询历史数据
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="1"  # 前复权
            )

            if rs.error_code != '0':
                logger.error(f"baostock 查询失败: {rs.error_msg}")
                return None

            # 转换为 DataFrame
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 转换为标准格式
            df['ts_code'] = ts_code
            df['trade_date'] = pd.to_datetime(df['date']).dt.date
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df['amount'].astype(float)

            # 只保留需要的列
            result = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            return result

        finally:
            bs.logout()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'sources': self.stats,
            'circuit_breakers': {
                name: {
                    'is_open': cb.is_open,
                    'failure_count': cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }

