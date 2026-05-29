"""项目配置管理"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 允许 .env 中存在未声明字段(向后兼容)
    )

    # ── 数据库 ──
    database_url: str = "postgresql://quant_user:quant_pass@localhost:5432/quant_db"
    redis_url: str = "redis://localhost:6379/0"

    # ── API Token ──
    tushare_token: str = ""

    # ── 路径 ──
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data_storage"
    cache_dir: Path = project_root / "data" / "cache"
    parquet_dir: Path = project_root / "data" / "parquet"  # L2 增量缓存
    log_dir: Path = project_root / "logs"

    # ── 交易参数 ──
    initial_capital: float = 1_000_000
    commission_rate: float = 0.00025
    stamp_duty: float = 0.001
    min_commission: float = 5.0

    # ── 回测 ──
    benchmark_index: str = "000300.SH"
    risk_free_rate: float = 0.03

    # ────────────────────────────────────────────────────
    # 限流配置 (QPS = 每秒最大请求数)
    # ────────────────────────────────────────────────────
    # 腾讯接口 - qt.gtimg.cn 批量行情, 单次可带 80 只
    rate_limit_tencent_qt: float = 5.0
    # 腾讯接口 - web.ifzq.gtimg.cn K线 (单只单请求)
    rate_limit_tencent_kline: float = 3.0
    # AKShare 走的是东财/新浪/同花顺等多源 - 比较敏感
    rate_limit_akshare: float = 2.0
    # Tushare Pro 免费 5000 积分 ~ 200次/分钟
    rate_limit_tushare: float = 0.5

    # ── 并发控制 ──
    max_workers_kline: int = 4        # 拉日 K 并发线程数
    max_workers_quote: int = 2        # 批量行情并发数

    # ── 多源降级 ──
    data_source_order: str = "tencent,tushare,akshare"  # 日K/板块等通用数据源降级顺序；本机 AKShare 偶发断连，Tushare token 已配置
    incremental_use_fallback: bool = True               # 增量扫描是否启用多源降级

    # ── 每日调度 ──
    daily_scan_enabled: bool = True
    daily_scan_hour: int = 16
    daily_scan_minute: int = 0
    daily_scan_lookback_days: int = 250
    daily_scan_alert_file: str = "logs/daily_scan_alerts.log"
    daily_scan_failure_threshold: int = 20              # 失败只数超过该值时写 ERROR 告警

    # ── 市场时间 ──
    market_close_hour: int = 15
    market_close_minute: int = 30                       # 15:30 视为收盘后缓冲，此时间后数据应完整

    # ── 每日全池因子预计算 (数据扫描后运行) ──
    daily_factor_enabled: bool = True
    daily_factor_hour: int = 16
    daily_factor_minute: int = 30
    daily_factor_max_workers: int = 4
    daily_factor_keep_days: int = 7

    # ── 重试与退避 ──
    retry_max_attempts: int = 4
    retry_base_delay: float = 1.0     # 基础退避秒数
    retry_max_delay: float = 30.0     # 单次最大退避

    # ── 熔断 ──
    circuit_failure_threshold: int = 10   # 连续失败 N 次熔断
    circuit_recovery_seconds: int = 300   # 熔断后 5 分钟尝试恢复
    circuit_half_open_max: int = 3        # 半开状态最多放行 N 个请求

    # ────────────────────────────────────────────────────
    # 缓存配置
    # ────────────────────────────────────────────────────
    cache_l1_size: int = 1024              # 内存 LRU 最大条目数
    cache_l1_ttl_seconds: int = 600        # L1 内存缓存 TTL
    cache_l2_quote_ttl_seconds: int = 1800 # L2 行情快照 TTL (盘中可短一些)
    cache_l2_kline_use_increment: bool = True  # K线启用增量更新

    # ────────────────────────────────────────────────────
    # 股票池过滤
    # ────────────────────────────────────────────────────
    universe_source: str = "all_a"     # all_a | hs300 | zz500 | zz1000
    max_stocks: int = 6000              # 全 A 约 5400 只

    exclude_st: bool = True             # 排除 ST / *ST / SST / S
    exclude_delisting: bool = True      # 排除退市整理
    exclude_new_stock_days: int = 60    # 排除上市不足 N 天
    exclude_suspended: bool = True      # 排除停牌
    exclude_bj: bool = False            # 是否排除北交所

    min_float_mv_yi: float = 0.0        # 最小流通市值(亿元) 0=不限
    max_float_mv_yi: Optional[float] = None  # 最大流通市值(亿元)
    min_avg_turnover: float = 0.0       # 最小日均换手率%

    # ────────────────────────────────────────────────────
    # 数据校验
    # ────────────────────────────────────────────────────
    validate_price_jump_ratio: float = 0.22  # 单日涨跌幅上限(超过则疑似异常)
    validate_strict: bool = False            # 严格模式: 校验失败直接抛错



settings = Settings()
# 确保关键目录存在
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.parquet_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
