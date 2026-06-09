"""
因子计算异常类
==============

定义因子计算过程中的各种异常。
"""


class FactorCalculationError(Exception):
    """因子计算基础异常"""
    pass


class DataValidationError(FactorCalculationError):
    """数据验证错误"""
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(f"Data validation error{f' in field {field}' if field else ''}: {message}")


class InsufficientDataError(FactorCalculationError):
    """数据不足错误"""
    def __init__(self, required: int, actual: int, message: str = None):
        self.required = required
        self.actual = actual
        msg = message or f"Insufficient data: required {required} points, got {actual}"
        super().__init__(msg)


class InvalidParameterError(FactorCalculationError):
    """无效参数错误"""
    def __init__(self, parameter: str, value, reason: str = None):
        self.parameter = parameter
        self.value = value
        msg = f"Invalid parameter '{parameter}' with value {value}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
