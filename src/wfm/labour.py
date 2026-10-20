"""Egyptian labour-law parameters used as planning constraints.

Based on public summaries of Egypt's Labour Law No. 14 of 2025 (in force from
September 2025, replacing Law No. 12 of 2003). Executive regulations were still
being finalised at the time of writing and secondary sources disagree on some
figures, so these are modelling assumptions, not legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabourRules:
    """Working-time rules applied by the scheduler, capacity plan and RTA desk."""

    max_daily_hours: float = 8.0
    max_weekly_hours: float = 48.0
    min_daily_break_hours: float = 1.0
    max_continuous_work_hours: float = 5.0
    min_weekly_rest_hours: float = 24.0
    max_daily_overtime_hours: float = 2.0
    overtime_premium_day: float = 0.35
    overtime_premium_night: float = 0.70
    annual_leave_days: int = 21
    contract_weekly_hours: float = 40.0  # NileConnect contract: 5 x 8 paid hours
    night_start_hour: int = 22  # site policy for the night overtime premium window
    night_end_hour: int = 6

    @property
    def shift_length_hours(self) -> float:
        """Shift length on site: paid hours plus the unpaid break."""
        return self.max_daily_hours + self.min_daily_break_hours


EGYPT_LABOUR = LabourRules()
