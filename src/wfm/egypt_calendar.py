"""Egyptian working calendar: weekends, public holidays, Ramadan and Eid.

Islamic dates follow the lunar calendar and are confirmed by moon sighting, so the
dates below are approximate (they can move by a day). Sham El Nessim follows the
Coptic Orthodox Easter Monday. The Egyptian working week runs Sunday to Thursday,
with Friday as the main weekend day and Saturday as a reduced day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate Ramadan periods (first and last fasting day).
RAMADAN_PERIODS: dict[int, tuple[str, str]] = {
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-19"),
    2027: ("2027-02-08", "2027-03-09"),
}
# Eid al-Fitr first day; Egypt typically closes for three days.
EID_AL_FITR: dict[int, str] = {2025: "2025-03-30", 2026: "2026-03-20", 2027: "2027-03-10"}
# Eid al-Adha first day; Arafat Day (the day before) plus three Eid days are off.
EID_AL_ADHA: dict[int, str] = {2025: "2025-06-06", 2026: "2026-05-27", 2027: "2027-05-16"}
ISLAMIC_NEW_YEAR: dict[int, str] = {2025: "2025-06-26", 2026: "2026-06-16"}
PROPHET_BIRTHDAY: dict[int, str] = {2025: "2025-09-04", 2026: "2026-08-25"}
SHAM_EL_NESSIM: dict[int, str] = {2025: "2025-04-21", 2026: "2026-04-13", 2027: "2027-05-03"}

FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 7): "Coptic Christmas",
    (1, 25): "Revolution Day (25 January)",
    (4, 25): "Sinai Liberation Day",
    (5, 1): "Labour Day",
    (6, 30): "June 30 Revolution",
    (7, 23): "Revolution Day (23 July)",
    (10, 6): "Armed Forces Day",
}

DAY_TYPES = ("weekday", "friday", "saturday")


def _holiday_table(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """List every public holiday date between start and end with its name and kind."""
    rows: list[tuple[pd.Timestamp, str, str]] = []
    for year in range(start.year, end.year + 1):
        for (month, day), name in FIXED_HOLIDAYS.items():
            rows.append((pd.Timestamp(year, month, day), name, "national"))
        if year in EID_AL_FITR:
            first = pd.Timestamp(EID_AL_FITR[year])
            rows += [(first + pd.Timedelta(days=i), "Eid al-Fitr", "eid") for i in range(3)]
        if year in EID_AL_ADHA:
            first = pd.Timestamp(EID_AL_ADHA[year])
            rows.append((first - pd.Timedelta(days=1), "Arafat Day", "eid"))
            rows += [(first + pd.Timedelta(days=i), "Eid al-Adha", "eid") for i in range(3)]
        for table, name in (
            (ISLAMIC_NEW_YEAR, "Islamic New Year"),
            (PROPHET_BIRTHDAY, "Prophet's Birthday"),
            (SHAM_EL_NESSIM, "Sham El Nessim"),
        ):
            if year in table:
                rows.append((pd.Timestamp(table[year]), name, "religious"))
    table = pd.DataFrame(rows, columns=["date", "holiday_name", "holiday_kind"])
    table = table.drop_duplicates("date", keep="first")
    return table[(table["date"] >= start) & (table["date"] <= end)]


def build_calendar(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Build a daily calendar with every feature the simulator and forecaster need.

    Args:
        start: First date (inclusive).
        end: Last date (inclusive).

    Returns:
        One row per date with day type, holiday, Ramadan, post-Eid and billing-cycle fields.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal["date"].dt.dayofweek  # Monday = 0 ... Sunday = 6
    cal["dow_name"] = cal["date"].dt.day_name().str[:3]
    cal["day_type"] = np.select(
        [cal["dow"] == 4, cal["dow"] == 5], ["friday", "saturday"], default="weekday"
    )
    cal["month"] = cal["date"].dt.month
    cal["day_of_month"] = cal["date"].dt.day
    cal["day_of_year"] = cal["date"].dt.dayofyear
    cal["week_start"] = cal["date"] - pd.to_timedelta((cal["dow"] + 1) % 7, unit="D")

    holidays = _holiday_table(start - pd.Timedelta(days=7), end + pd.Timedelta(days=7))
    cal = cal.merge(holidays, on="date", how="left")
    cal["is_holiday"] = cal["holiday_name"].notna()
    cal["is_eid"] = cal["holiday_kind"].eq("eid")
    cal["holiday_name"] = cal["holiday_name"].fillna("")
    cal["holiday_kind"] = cal["holiday_kind"].fillna("")

    cal["is_ramadan"] = False
    cal["ramadan_day"] = 0
    for first, last in RAMADAN_PERIODS.values():
        in_period = cal["date"].between(pd.Timestamp(first), pd.Timestamp(last))
        cal.loc[in_period, "is_ramadan"] = True
        cal.loc[in_period, "ramadan_day"] = (cal.loc[in_period, "date"] - pd.Timestamp(first)).dt.days + 1

    # Post-Eid bounce-back: the first day after an Eid holiday block.
    eid_dates = set(holidays.loc[holidays["holiday_kind"] == "eid", "date"])
    cal["is_post_eid"] = [
        (d - pd.Timedelta(days=1)) in eid_dates and d not in eid_dates for d in cal["date"]
    ]
    cal["bill_cycle_day"] = np.where(cal["day_of_month"] <= 5, cal["day_of_month"], 0)
    return cal


def profile_day_type(day_type: str, is_holiday: bool) -> str:
    """Map a calendar day to the intraday profile family it follows (holidays behave like Fridays)."""
    return "friday" if is_holiday else day_type
