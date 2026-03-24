-- Phase 6: Adaptive Signal Weights
-- Stores learned signal weights per day per regime.

CREATE TABLE IF NOT EXISTS adaptive_signal_weights (
    date         DATE        NOT NULL,
    regime_code  TEXT        NOT NULL DEFAULT '_GLOBAL',
    pump_weight  REAL        NOT NULL,
    oi_weight    REAL        NOT NULL,
    funding_weight REAL      NOT NULL,
    accel_weight REAL        NOT NULL,
    sample_size  INTEGER     NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (date, regime_code)
);
