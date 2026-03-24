# AGP Platform

A production market intelligence platform for cryptocurrency perpetual futures, deployed on a
Raspberry Pi 5 homelab. Built across seven iterative engineering phases, the platform combines
real-time market scanning, a multi-component signal fusion engine, outcome-tracked alert history,
and an AI-powered assistant — all running as managed systemd services against a PostgreSQL backend.

---

## Systems

### 1. hltrader — Market Scanner & Signal Engine

A continuously-running scanner for perpetual futures markets. Identifies overextended assets using
a composite scoring engine, filters candidates through multiple intelligence layers, and delivers
batched Telegram alerts.

**Scoring & Detection**
- Composite scoring engine with configurable per-signal weights (pump momentum, funding rate,
  open interest, momentum acceleration)
- Dynamic percentile filtering: score threshold self-calibrates to the current distribution
  of candidates each cycle
- Dual alert lanes (Normal / Extreme) with independent per-coin cooldown tracking and
  per-lane batch caps
- Momentum acceleration derived from multi-timeframe candle data with per-cycle caching

**Conviction Score Fusion (Phase 4)**
- 5-component weighted fusion score (0–100): base signal strength, historical win rate for
  the symbol, market regime alignment, liquidity, and geo-event correlation
- Conviction score gates alert delivery; shadow mode logs decisions without blocking
- Human-readable "why allowed / why blocked" reasons surface the top contributing components
- TTL-cached DB lookups for historical win rates, BTC regime state, and geo-event table

**Market Regime Engine (Phase 5)**
- 6-code regime classification: `{UPTREND,DOWNTREND,CHOP} × {EXPANSION,CONTRACTION}`
- BTC trend derived from multi-timeframe returns; volume state uses hysteresis to prevent
  rapid regime flapping
- Regime snapshots persisted to PostgreSQL; conviction score adjusts per current regime

**Adaptive Signal Weights (Phase 6)**
- Daily Pearson correlation learning between signal components and 24h alert outcomes
- EMA smoothing prevents whipsaw from single-day noise
- Delta clamping limits maximum weight shift per day
- REGIME → GLOBAL → STATIC fallback chain: uses regime-specific weights when enough samples
  exist, falls back to cross-regime aggregate, then falls back to static defaults
- Weights stored normalized; re-scaled to scorer at runtime

**Token Personality Memory (Phase 7)**
- Per-symbol behavioral profiles built from rolling outcome history: trend_follower,
  mean_reverter, volatile, choppy
- Best-regime and worst-regime fingerprinting; regime match adds a bounded conviction
  adjustment (capped positive and negative)
- Confidence-gated: profiles below minimum sample count are ignored
- Daily compute timer; CLI inspect commands

**Sector Clustering (Phase 5.3)**
- Canonical sector map (DeFi, L1, AI, Gaming, RWA, etc.) groups concurrent alerts
- Distinguishes coordinated sector moves from isolated pumps
- Cluster type recorded on alert outcome rows for post-hoc analysis

**Pump-Fade Engine**
- RSI-14 (Wilder-smoothed) on 5m and 1h candles; rollover detection (peak + consecutive decline)
- Bollinger Band squeeze detection as a risk flag
- Two independent routing paths: feed addendum for moderate confidence, standalone trade alert
  for high confidence + RSI rollover; separate cooldown tracking per path

---

### 2. Alert Outcome Tracking & Evaluation (Phase 3 / 6.5)

Every alert is recorded with entry price and scores at send time, then evaluated against
candle data at multiple horizons.

- Outcome columns: 1h return, 4h return, 24h return, MFE (max favorable excursion), MAE
  (max adverse excursion) over 24h
- Primary candle source: PostgreSQL OHLCV tables for major symbols
- Fallback candle source: exchange candleSnapshot API for all other symbols — per-run bounded
  cache with rate limiting; primary connection failure is non-fatal
- `get_weekly_stats()` excludes NULL returns from denominators (no denominator inflation)
- `reset_null_evals()` + `backfill` CLI commands for historical repair
- Evaluation runs every 15 minutes via systemd timer

---

### 3. Nacho — Financial Intelligence AI Assistant

A Telegram bot powered by a local LLM, providing natural-language market intelligence over
Interactive Brokers and Hyperliquid data.

**14-Step Async `/ask` Pipeline**
- Intent classification: routes queries to RECAP / DECISION / EXPLAIN / RESEARCH / SYSTEM /
  EXECUTION handlers before LLM dispatch
- 3-layer anti-hallucination: coverage probe (does the data exist?) → prompt gate (tells LLM
  what it does not have) → regex output validator (blocks forbidden claims)
- Context cards injected per query type: market state, intraday prices, macro summary, open
  positions, portfolio snapshot — each card emits freshness metadata (age in minutes, stale flag)

**Memory & Learning**
- Conversation memory via PostgreSQL full-text search + ONNX vector embeddings (all-MiniLM-L6-v2)
- Singleton ONNX embedding wrapper (loaded once, reused across requests)
- Reaction-based learning: emoji feedback on bot responses writes good/bad training examples to DB

**Market State Card**
- Unified JSON + Markdown state file written every 10 minutes by a dedicated systemd timer
- Contains: current regime code, recent alert summary, geo-event digest, adaptive weights block
  (source, as-of date, sample size, values), system health
- Read by the AI assistant as a structured context card on every `/ask`

**Additional Services**
- Geo-event detector: scores news articles for conflict/escalation using a 3-stage deterministic
  gate; high-severity events feed into the conviction system's geo component
- Congress bill ingestion + policy catalyst scoring
- FOMC rate decision parser (HIKE / CUT / HOLD classification)
- HL trading with symbol allowlist and position size limits
- Daily morning digest, alertstats breakdown (conviction tiers, win rates, personality leaderboard)

---

### 4. H2 Neutral Reversion Research Module

Statistical study on large intraday movers segmented by funding-rate regime. Identified a
mean-reversion edge in the neutral-funding bucket, validated with bootstrap confidence intervals
and non-parametric tests (Mann-Whitney U, Welch t-test).

**Research pipeline**: extraction script (5m candles with 1h fallback) → per-bucket bootstrap CI
→ deep-dive (OI-weighted signal, Go/No-Go gate) → live alpha dashboard → weekly Markdown report.

**Production system**: alpha filters applied in real time. Every qualifying signal logged
to PostgreSQL with entry price, funding rate, BTC regime state, bid/ask spread, and relative volume.
Market state (funding percentile distribution + BTC return) updated every 15 minutes by a dedicated
service and consumed by the detector with a stale-state guard.

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| **Hardware** | Raspberry Pi 5, 8 GB RAM, ~917 GB SSD, Ubuntu 24.04 ARM64 |
| **Database** | PostgreSQL 16 — two databases, 24+ migrations, managed with raw SQL scripts |
| **Services** | 7 systemd services + 5 timers; remote deploy via SSH + Python patch scripts |
| **Local LLM** | Qwen2.5-3B for offline inference (no external API dependency for base queries) |
| **Containerization** | Docker — isolated agent with a dedicated workspace volume |
| **Config** | pydantic-settings; all tuning via `.env`; systemd ExecStart has no CLI overrides |
| **CLI** | Typer-based; subcommands registered via `app.add_typer()` |
| **Deploy pattern** | Python patch scripts pushed via `scp`; zero-downtime via service restart |
| **Fail-open design** | DB failures logged but never block alert delivery |

## Tech Stack

Python · PostgreSQL 16 · Raspberry Pi 5 (Ubuntu 24.04 ARM64) · Systemd · Telegram Bot API ·
Docker · ONNX Runtime · pydantic-settings · Typer · pytest · psycopg2

## Tests

**509 tests, all passing** — pytest with `unittest.mock`; DB layer tested via mock cursor context
managers. Test suite covers scoring engine, adaptive weights, conviction fusion, outcome evaluation,
candle fallback logic, sector clustering, token personality memory, and regime classification.
