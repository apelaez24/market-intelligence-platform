"""Configuration via environment variables / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings

TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
MAINNET_API_URL = "https://api.hyperliquid.xyz"


class Settings(BaseSettings):
    HL_PRIVATE_KEY: str = ""
    HL_API_URL: str = MAINNET_API_URL
    HL_NO_STOP_NO_TRADE: bool = False
    HL_DEFAULT_SL_PCT: float = 2.0
    HL_DEFAULT_SLIPPAGE: float = 0.05
    HL_MONITOR_INTERVAL_SECONDS: int = 5
    HL_TESTNET: bool = False
    HL_DRY_RUN: bool = False
    HL_TELEGRAM_BOT_TOKEN: str = ""
    HL_TELEGRAM_CHAT_ID: str = ""
    HL_SCAN_INTERVAL_SECONDS: int = 3600
    HL_SCAN_COOLDOWN_SECONDS: int = 3600

    # --- Phase 1A: Normal lane tuning ---
    HL_SCAN_MIN_SCORE: float = 20.0
    HL_SCAN_MIN_PCT_24H: float = 8.0
    HL_SCAN_MIN_VOLUME_USD_24H: float = 1_000_000.0
    HL_SCAN_MIN_OI_USD: float = 0.0

    # Liquidity percentile: keep top N% by OI/volume (0.70 = top 70%)
    HL_SCAN_LIQUIDITY_PERCENTILE_KEEP: float = 0.70
    # "oi" | "volume" | "auto" (auto = prefer OI, fallback volume)
    HL_SCAN_LIQUIDITY_METRIC: str = "auto"

    # Momentum acceleration curve
    HL_SCAN_PUMP_CAP_PCT: float = 25.0
    HL_SCAN_PUMP_EXP: float = 1.6

    # --- Phase 1B: Extreme pump lane ---
    HL_EXTREME_ENABLED: bool = True
    HL_EXTREME_PCT_24H: float = 20.0
    HL_EXTREME_SCORE_THRESHOLD: float = 45.0
    HL_EXTREME_MAX_ITEMS: int = 3
    HL_EXTREME_COOLDOWN_SECONDS: int = 21600  # 6 hours

    # --- Cross-alert de-duplication (DB) ---
    HL_PG_HOST: str = "127.0.0.1"
    HL_PG_PORT: int = 5432
    HL_PG_DBNAME: str = "market_intel"
    HL_PG_USER: str = "postgres"
    HL_PG_PASSWORD: str = ""
    HL_ALERT_XDEDUPE_ENABLED: bool = False  # safe default: off

    # --- Pump-Fade scoring (addendum on EXTREME PUMP alerts) ---
    HL_PUMP_FADE_ENABLED: bool = False
    HL_PUMP_FADE_TIMEOUT_S: int = 10
    HL_PUMP_FADE_CACHE_TTL: int = 900        # 15 minutes
    HL_PUMP_FADE_MIN_SCORE: int = 60         # minimum to show addendum
    HL_PG_COINBASE_DBNAME: str = "coinbase_data"

    # --- Two-chat routing (FEED vs TRADES) ---
    HL_TELEGRAM_CHAT_ID_FEED: str = ""       # defaults to HL_TELEGRAM_CHAT_ID at runtime
    HL_TELEGRAM_CHAT_ID_TRADES: str = ""     # required for trade alerts; skip if empty
    HL_PUMP_FADE_TRADE_MIN_SCORE: int = 75   # min fade score for TRADES chat
    HL_PUMP_FADE_TRADE_COOLDOWN_MIN: int = 120  # per-symbol cooldown (minutes)

    # --- Phase 2: Signal-Quality Alerts ---
    # Dynamic percentile filter: keep top N% of candidates
    HL_SCAN_PUMP_PERCENTILE_KEEP: float = 0.30       # keep top 30% by pump metric
    HL_SCAN_SCORE_PERCENTILE_KEEP: float = 0.40      # keep top 40% by composite score
    HL_SCAN_PERCENTILE_MODE: str = "pump"             # "pump" | "score" | "both"

    # Momentum acceleration weights (must sum to 100)
    HL_SCAN_W_PUMP: float = 40.0
    HL_SCAN_W_FUNDING: float = 30.0
    HL_SCAN_W_OI: float = 20.0
    HL_SCAN_W_ACCEL: float = 10.0

    # Acceleration curve parameters
    HL_SCAN_ACCEL_A0: float = 0.0    # baseline (subtracted before normalization)
    HL_SCAN_ACCEL_A1: float = 2.0    # scale: 2% = meaningful acceleration

    # Candle cache TTL for 1h/4h return computation (seconds)
    HL_SCAN_CANDLE_CACHE_TTL: int = 300  # 5 min (within a scan cycle)

    # --- Phase 3: Alert Outcome Tracking ---
    HL_ALERT_TRACKING_ENABLED: bool = True

    # --- Phase 4: Conviction Score Fusion ---
    HL_CONVICTION_ENABLED: bool = False       # safe default: off
    HL_CONVICTION_MIN: float = 55.0           # min conviction to send alert
    HL_CONVICTION_W_BASE: float = 40.0
    HL_CONVICTION_W_HISTORY: float = 20.0
    HL_CONVICTION_W_REGIME: float = 20.0
    HL_CONVICTION_W_LIQUIDITY: float = 10.0
    HL_CONVICTION_W_GEO: float = 10.0
    HL_CONVICTION_MIN_SAMPLE: int = 10        # min alerts for symbol-specific stats
    HL_CONVICTION_CACHE_TTL: int = 600        # 10 min cache for DB lookups
    HL_CONVICTION_SHADOW: bool = True          # shadow mode: compute+log but don't block
    HL_CONVICTION_HISTORY_PRIOR: float = 50.0  # prior score when no history data

    # --- Phase 5: Market Regime Engine ---
    HL_REGIME_ENABLED: bool = False
    HL_REGIME_CACHE_TTL: int = 600
    HL_REGIME_SLOPE_UP: float = 0.002
    HL_REGIME_SLOPE_DOWN: float = -0.002
    HL_REGIME_ATR_EXPAND: float = 1.25
    HL_REGIME_ATR_CONTRACT: float = 0.85
    HL_REGIME_BREADTH_PUMP: float = 0.18
    HL_REGIME_BREADTH_DUMP: float = 0.18
    HL_REGIME_STORE_ENABLED: bool = True
    HL_REGIME_CONVICTION_ADJUST: bool = False
    HL_REGIME_CONVICTION_UP_PENALTY: float = 7.0
    HL_REGIME_CONVICTION_DOWN_BOOST: float = 7.0

    # --- Phase 5.3: Signal Clustering ---
    HL_CLUSTER_ENABLED: bool = False
    HL_CLUSTER_MIN_SIZE: int = 3
    HL_CLUSTER_CONVICTION_BOOST: float = 5.0
    HL_CLUSTER_MAX_PER_MESSAGE: int = 8

    # --- Phase 6: Adaptive Signal Weights ---
    HL_ADAPTIVE_WEIGHTS_ENABLED: bool = False
    HL_ADAPTIVE_WEIGHTS_EMA_ALPHA: float = 0.2
    HL_ADAPTIVE_WEIGHTS_LOOKBACK_DAYS: int = 7
    HL_ADAPTIVE_WEIGHTS_MIN_SAMPLE: int = 15
    HL_ADAPTIVE_WEIGHTS_MIN_GLOBAL: int = 30
    HL_ADAPTIVE_WEIGHTS_MIN_PER_REGIME: int = 20
    HL_ADAPTIVE_WEIGHTS_MAX_DAILY_DELTA: float = 0.08

    # --- Phase 7: Token Personality Memory ---
    HL_TOKEN_MEMORY_ENABLED: bool = True
    HL_TOKEN_MEMORY_LOOKBACK_DAYS: int = 30
    HL_TOKEN_MEMORY_MIN_SAMPLE: int = 5
    HL_TOKEN_MEMORY_MIN_SAMPLE_REGIME: int = 3
    HL_TOKEN_MEMORY_CONFIDENCE_MIN: float = 40.0
    HL_TOKEN_MEMORY_MAX_BOOST: float = 6.0
    HL_TOKEN_MEMORY_MAX_PENALTY: float = 6.0
    HL_TOKEN_MEMORY_CONVICTION_ENABLED: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def pg_dsn(self) -> str:
        """Build a libpq-style DSN from individual settings."""
        return (f"host={self.HL_PG_HOST} port={self.HL_PG_PORT} "
                f"dbname={self.HL_PG_DBNAME} user={self.HL_PG_USER} "
                f"password={self.HL_PG_PASSWORD}")

    @property
    def pg_dsn_coinbase(self) -> str:
        """Build a libpq-style DSN for coinbase_data DB."""
        return (f"host={self.HL_PG_HOST} port={self.HL_PG_PORT} "
                f"dbname={self.HL_PG_COINBASE_DBNAME} user={self.HL_PG_USER} "
                f"password={self.HL_PG_PASSWORD}")

    @property
    def feed_chat_id(self) -> str:
        """Return FEED chat ID, falling back to HL_TELEGRAM_CHAT_ID."""
        return self.HL_TELEGRAM_CHAT_ID_FEED or self.HL_TELEGRAM_CHAT_ID

    @property
    def effective_api_url(self) -> str:
        """Return testnet URL when HL_TESTNET is True, otherwise HL_API_URL."""
        if self.HL_TESTNET:
            return TESTNET_API_URL
        return self.HL_API_URL


settings = Settings()
