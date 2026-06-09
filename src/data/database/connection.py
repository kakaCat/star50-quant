# src/data/database/connection.py
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
from loguru import logger


Base = declarative_base()


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, config: dict):
        """
        初始化数据库管理器

        Args:
            config: 数据库配置字典
        """
        self.config = config['postgres']
        self.engine = None
        self.Session = None
        self.metadata = MetaData()

    def connect(self):
        """创建数据库连接"""
        if self.engine is not None:
            logger.warning("数据库已连接，跳过重复连接")
            return

        # 构建连接URL
        db_url = (
            f"postgresql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

        # 创建引擎
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=self.config.get('pool_size', 10),
            max_overflow=self.config.get('max_overflow', 20),
            pool_timeout=self.config.get('pool_timeout', 30),
            pool_recycle=self.config.get('pool_recycle', 3600),
            echo=self.config.get('echo', False)
        )

        # 创建Session工厂
        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"数据库连接成功: {self.config['host']}:{self.config['port']}/{self.config['database']}")

    def disconnect(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None
            logger.info("数据库连接已关闭")

    @contextmanager
    def get_session(self) -> Generator:
        """
        获取数据库会话（上下文管理器）

        Yields:
            SQLAlchemy Session对象
        """
        if self.Session is None:
            raise RuntimeError("数据库未连接，请先调用connect()")

        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()

    def execute_sql_file(self, sql_file: str):
        """
        执行SQL文件

        Args:
            sql_file: SQL文件路径
        """
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        with self.engine.begin() as conn:
            # 分割SQL语句并执行
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            for statement in statements:
                conn.execute(text(statement))

        logger.info(f"SQL文件执行成功: {sql_file}")
