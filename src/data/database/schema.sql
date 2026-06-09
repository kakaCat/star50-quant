-- src/data/database/schema.sql
-- 科创50指数增强策略数据库表结构

-- 股票日频数据表
CREATE TABLE IF NOT EXISTS stock_daily (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,          -- 股票代码
    trade_date DATE NOT NULL,               -- 交易日期
    open DECIMAL(10, 2),                    -- 开盘价
    high DECIMAL(10, 2),                    -- 最高价
    low DECIMAL(10, 2),                     -- 最低价
    close DECIMAL(10, 2),                   -- 收盘价
    volume BIGINT,                          -- 成交量（手）
    amount DECIMAL(20, 2),                  -- 成交额（元）
    turnover_rate DECIMAL(10, 4),          -- 换手率（%）
    amplitude DECIMAL(10, 4),              -- 振幅（%）
    pct_change DECIMAL(10, 4),             -- 涨跌幅（%）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_stock_daily_code ON stock_daily(ts_code);
CREATE INDEX idx_stock_daily_date ON stock_daily(trade_date);
CREATE INDEX idx_stock_daily_code_date ON stock_daily(ts_code, trade_date);

-- 股票分钟频数据表（可选）
CREATE TABLE IF NOT EXISTS stock_minute (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_datetime TIMESTAMP NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    amount DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_datetime)
);

CREATE INDEX idx_stock_minute_code ON stock_minute(ts_code);
CREATE INDEX idx_stock_minute_datetime ON stock_minute(trade_datetime);

-- 基本面数据表
CREATE TABLE IF NOT EXISTS fundamentals (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    total_mv DECIMAL(20, 2),               -- 总市值
    circ_mv DECIMAL(20, 2),                -- 流通市值
    pb DECIMAL(10, 4),                     -- 市净率
    pe DECIMAL(10, 4),                     -- 市盈率
    ps DECIMAL(10, 4),                     -- 市销率
    dv_ratio DECIMAL(10, 4),               -- 股息率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_fundamentals_code ON fundamentals(ts_code);
CREATE INDEX idx_fundamentals_date ON fundamentals(trade_date);

-- 因子数据表
CREATE TABLE IF NOT EXISTS factors (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    factor_name VARCHAR(50) NOT NULL,
    factor_value DECIMAL(20, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date, factor_name)
);

CREATE INDEX idx_factors_code ON factors(ts_code);
CREATE INDEX idx_factors_date ON factors(trade_date);
CREATE INDEX idx_factors_name ON factors(factor_name);
CREATE INDEX idx_factors_code_date ON factors(ts_code, trade_date);

-- 指数成分股历史表
CREATE TABLE IF NOT EXISTS index_components (
    id SERIAL PRIMARY KEY,
    index_code VARCHAR(20) NOT NULL,       -- 指数代码
    ts_code VARCHAR(20) NOT NULL,          -- 成分股代码
    in_date DATE NOT NULL,                 -- 纳入日期
    out_date DATE,                         -- 剔除日期（NULL表示仍在指数中）
    weight DECIMAL(10, 6),                 -- 权重
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_index_components_index ON index_components(index_code);
CREATE INDEX idx_index_components_stock ON index_components(ts_code);
CREATE INDEX idx_index_components_date ON index_components(in_date, out_date);

-- 指数日频数据表
CREATE TABLE IF NOT EXISTS index_daily (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,          -- 指数代码
    trade_date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    amount DECIMAL(20, 2),
    pct_change DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_index_daily_code ON index_daily(ts_code);
CREATE INDEX idx_index_daily_date ON index_daily(trade_date);
