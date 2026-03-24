"""Phase 5.3: Signal Clustering — detect sector rotations.

Groups candidates by sector and consolidates alerts when 3+ symbols
in the same sector trigger in a single scan cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Shared Sector Data (canonical source) ────────────────────

SECTOR_MAP: dict[str, str] = {
    # AI / ML
    "AIXBT": "ai", "TAO": "ai", "FET": "ai", "RENDER": "ai", "AKT": "ai",
    "NEAR": "ai", "AR": "ai", "WLD": "ai", "OLAS": "ai", "PRIME": "ai",
    # Meme
    "MEME": "meme", "DOGE": "meme", "SHIB": "meme", "PEPE": "meme",
    "WIF": "meme", "BONK": "meme", "FLOKI": "meme", "NEIRO": "meme",
    "POPCAT": "meme", "MEW": "meme", "BRETT": "meme", "MOG": "meme",
    # DeFi
    "UNI": "defi", "AAVE": "defi", "MKR": "defi", "CRV": "defi",
    "PENDLE": "defi", "DYDX": "defi", "SNX": "defi", "COMP": "defi",
    "SUSHI": "defi", "1INCH": "defi", "JUP": "defi",
    # L1 / Infrastructure
    "BTC": "l1", "ETH": "l1", "SOL": "l1", "AVAX": "l1", "ADA": "l1",
    "DOT": "l1", "ATOM": "l1", "SUI": "l1", "APT": "l1", "SEI": "l1",
    "TIA": "l1", "INJ": "l1", "HYPE": "l1",
    # L2
    "OP": "l2", "ARB": "l2", "STRK": "l2", "MATIC": "l2",
    "BLAST": "l2", "MANTA": "l2", "ZK": "l2",
    # Gaming
    "IMX": "gaming", "GALA": "gaming", "AXS": "gaming",
    "RONIN": "gaming", "PIXEL": "gaming", "XAI": "gaming",
}

SECTOR_LABELS: dict[str, str] = {
    "ai": "AI tokens",
    "meme": "Meme coins",
    "defi": "DeFi protocols",
    "l1": "L1 chains",
    "l2": "L2 scaling",
    "gaming": "Gaming tokens",
}


# ── SectorCluster ────────────────────────────────────────────

@dataclass
class SectorCluster:
    """A group of candidates in the same sector."""
    sector: str              # "meme", "ai", etc.
    label: str               # "Meme coins", "AI tokens"
    candidates: list         # list[ShortCandidate]
    avg_composite: float
    avg_pct_24h: float


# ── Core Logic ───────────────────────────────────────────────

def detect_clusters(
    candidates: list,
    *,
    min_cluster_size: int = 3,
    conviction_boost: float = 8.0,
) -> tuple[list[SectorCluster], list]:
    """Detect sector clusters and return (clusters, unclustered_candidates).

    1. Group candidates by SECTOR_MAP
    2. Sectors with count >= min_cluster_size form a SectorCluster
    3. "other" sector never clusters
    4. Apply conviction boost to cluster members (capped at 100)
    5. Fail-open: on any error, return ([], all_candidates)
    """
    try:
        # Group by sector
        by_sector: dict[str, list] = {}
        for c in candidates:
            sector = SECTOR_MAP.get(c.coin, "other")
            by_sector.setdefault(sector, []).append(c)

        clusters: list[SectorCluster] = []
        unclustered: list = []

        for sector, members in by_sector.items():
            # "other" never clusters
            if sector == "other" or len(members) < min_cluster_size:
                unclustered.extend(members)
                continue

            # Sort by composite descending within the cluster
            members.sort(key=lambda c: c.composite, reverse=True)

            avg_composite = sum(c.composite for c in members) / len(members)
            avg_pct = sum(c.pct_24h for c in members) / len(members)

            cluster = SectorCluster(
                sector=sector,
                label=SECTOR_LABELS.get(sector, sector),
                candidates=members,
                avg_composite=avg_composite,
                avg_pct_24h=avg_pct,
            )
            clusters.append(cluster)

            # Apply conviction boost to cluster members
            for c in members:
                if c.conviction_score is not None:
                    c.conviction_score = min(100.0, c.conviction_score + conviction_boost)

        return clusters, unclustered

    except Exception as exc:
        log.warning("detect_clusters failed (fail-open): %s", exc)
        return [], list(candidates)
