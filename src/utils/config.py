# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 替换环境变量
    config = _replace_env_vars(config)

    return config


def _replace_env_vars(config: Any) -> Any:
    """递归替换配置中的环境变量"""
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config


def load_env():
    """加载.env文件"""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
