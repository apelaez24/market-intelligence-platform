-- Phase 7: Token Personality Memory
-- Stores compact per-symbol behavioral aggregates from alert_outcomes.
-- Recomputed daily. No raw history stored.

CREATE TABLE IF NOT EXISTS token_personality_memory (
    symbol                TEXT PRIMARY KEY,
    sample_size           INTEGER NOT NULL,
    sample_size_clustered INTEGER NOT NULL DEFAULT 0,
    sample_size_unclustered INTEGER NOT NULL DEFAULT 0,

    win_1h                REAL,
    win_4h                REAL,
    win_24h               REAL,

    avg_ret_1h            REAL,
    avg_ret_4h            REAL,
    avg_ret_24h           REAL,

    avg_mfe_24h           REAL,
    avg_mae_24h           REAL,

    trend_follow_score    REAL,
    mean_reversion_score  REAL,
    reversal_speed_score  REAL,
    cluster_sensitivity   REAL,

    best_regime           TEXT,
    worst_regime          TEXT,
    confidence_score      REAL,

    personality_label     TEXT,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tpm_updated ON token_personality_memory (updated_at);
CREATE INDEX IF NOT EXISTS idx_tpm_confidence ON token_personality_memory (confidence_score);
