# scripts/setup_database.py
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_env, setup_logger
from src.data.database import DatabaseManager


def main():
    """初始化数据库"""
    # 加载环境变量
    load_env()

    # 设置日志
    logger = setup_logger(log_level="INFO")

    logger.info("=== 开始初始化数据库 ===")

    # 加载数据库配置
    db_config = load_config("configs/db_config.yaml")

    # 创建数据库管理器
    db_manager = DatabaseManager(db_config)

    try:
        # 连接数据库
        db_manager.connect()
        logger.info("数据库连接成功")

        # 执行schema.sql
        schema_file = Path(__file__).parent.parent / "src" / "data" / "database" / "schema.sql"
        logger.info(f"执行SQL文件: {schema_file}")
        db_manager.execute_sql_file(str(schema_file))

        logger.info("=== 数据库初始化完成 ===")

    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        sys.exit(1)
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()
