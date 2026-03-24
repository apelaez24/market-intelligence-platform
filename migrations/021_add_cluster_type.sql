-- Phase 5.3: Signal Clustering — add cluster_type to alert_outcomes
-- Nullable TEXT: NULL for individual alerts, sector name for clustered ("meme", "ai", etc.)

ALTER TABLE alert_outcomes
  ADD COLUMN IF NOT EXISTS cluster_type TEXT;

CREATE INDEX IF NOT EXISTS idx_alert_outcomes_cluster
  ON alert_outcomes (cluster_type)
  WHERE cluster_type IS NOT NULL;
