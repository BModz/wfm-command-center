"""Central configuration for the NileConnect WFM model.

NileConnect is a fictional Cairo telecom contact centre. Every number here is a
modelling assumption chosen to look like a mid-size Egyptian telco operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INTERVAL_MINUTES = 30
INTERVAL_SECONDS = INTERVAL_MINUTES * 60
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES

HISTORY_START = pd.Timestamp("2024-10-01")
HISTORY_END = pd.Timestamp("2026-09-30")
FORECAST_WEEKS = 26
# First Sunday after the history ends: WFM weeks run Sunday to Saturday in Egypt.
FORECAST_START = pd.Timestamp("2026-10-04")
FORECAST_END = FORECAST_START + pd.Timedelta(weeks=FORECAST_WEEKS) - pd.Timedelta(days=1)

SL_TARGET = 0.80
SL_THRESHOLD_S = 20
MAX_OCCUPANCY = 0.88
DEFAULT_SEED = 42

# The RTA desk replays this day on AR_TECH; an afternoon network outage is injected.
RTA_QUEUE = "AR_TECH"
RTA_REPLAY_DATE = pd.Timestamp("2026-09-22")
RTA_OUTAGE_START = "14:00"
RTA_OUTAGE_HOURS = 4.5
RTA_OUTAGE_PEAK = 3.0


@dataclass(frozen=True)
class QueueSpec:
    """Static description of one contact queue.

    Attributes:
        code: Short queue identifier used throughout the data.
        name: Business-facing queue name.
        daily_calls: Average offered calls per day at the start of the history.
        aht_s: Average handle time (talk + hold + wrap) in seconds at the start of history.
        open_hour: First opening hour (inclusive).
        close_hour: Closing hour (exclusive); 24 means open around the clock.
        yoy_growth: Underlying year-on-year volume growth.
        patience_s: Mean caller patience used by the Erlang A abandonment model.
    """

    code: str
    name: str
    daily_calls: float
    aht_s: float
    open_hour: int
    close_hour: int
    yoy_growth: float
    patience_s: float

    @property
    def is_24x7(self) -> bool:
        return self.open_hour == 0 and self.close_hour == 24

    @property
    def hours_label(self) -> str:
        return "24/7" if self.is_24x7 else f"{self.open_hour:02d}:00-{self.close_hour:02d}:00"


QUEUES: dict[str, QueueSpec] = {
    q.code: q
    for q in (
        QueueSpec("AR_BILLING", "Billing & Payments (Arabic)", 5200, 270, 0, 24, 0.06, 300),
        QueueSpec("AR_TECH", "Technical Support (Arabic)", 4300, 410, 0, 24, 0.09, 360),
        QueueSpec("EN_SUPPORT", "English Support", 900, 350, 0, 24, 0.14, 240),
        QueueSpec("SALES", "Sales & Upgrades", 1600, 480, 9, 23, 0.07, 180),
    )
}
QUEUE_CODES: list[str] = list(QUEUES)


def open_mask(queue: str) -> np.ndarray:
    """Boolean mask over the 48 daily intervals marking when a queue is open."""
    spec = QUEUES[queue]
    hours = np.arange(INTERVALS_PER_DAY) * INTERVAL_MINUTES / 60
    return (hours >= spec.open_hour) & (hours < spec.close_hour)
