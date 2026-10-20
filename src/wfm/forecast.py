"""Volume and AHT forecasting: daily models, top-down interval forecast and backtests.

Pipeline:
1. Anomaly treatment: network-outage intervals are replaced by a baseline
   estimate. Outages are unplanned and cannot be forecast, so leaving them in
   would inflate the weekday level and distort the intraday profile. They are
   handled in real time (RTA) and through contingency plans instead.
2. Daily forecast per queue with three models: seasonal naive, Holt-Winters
   and one global LightGBM model on calendar features with a detrended target.
3. Top-down interval forecast: daily forecast x intraday profile for the day
   type (Sun-Thu / Fri / Sat x normal / Ramadan). The Ramadan profile is learned
   from the previous Ramadan. AHT is forecast per interval in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from wfm.config import INTERVAL_MINUTES, INTERVALS_PER_DAY, QUEUE_CODES, open_mask
from wfm.egypt_calendar import profile_day_type

MODELS = ("Seasonal naive", "Holt-Winters", "LightGBM")
PROFILE_WEEKS = 8
FEATURES = [
    "queue_code",
    "dow",
    "month",
    "day_of_year",
    "is_holiday",
    "is_eid",
    "is_ramadan",
    "ramadan_day",
    "is_post_eid",
    "bill_cycle_day",
    "is_campaign",
]
LGB_PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "seed": 0,
    "deterministic": True,
    "num_threads": 1,
    "verbose": -1,
}
LGB_ROUNDS = 600


@dataclass(frozen=True)
class BacktestWindow:
    """A historical period re-forecast as if standing on ``train_end``."""

    name: str
    train_end: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


BACKTESTS = (
    BacktestWindow("Normal (Aug-Sep 2026)", pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02"), pd.Timestamp("2026-09-26")),
    BacktestWindow("Ramadan 2026", pd.Timestamp("2026-02-07"), pd.Timestamp("2026-02-08"), pd.Timestamp("2026-03-28")),
)


# ---------------------------------------------------------------------------
# Anomaly treatment
# ---------------------------------------------------------------------------


def outage_flags(intervals: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """True for intervals inside an outage window on the outage queue or its spillover queue."""
    flag = pd.Series(False, index=intervals.index)
    for ev in events[events["event_type"] == "outage"].itertuples():
        in_window = (intervals["interval_start"] >= ev.start) & (intervals["interval_start"] < ev.end)
        affected = intervals["queue"].isin([ev.queue, ev.spillover_queue])
        flag |= in_window & affected
    return flag


def clean_outages(intervals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add ``offered_clean`` and ``is_outage`` columns, replacing outage-driven volume by a baseline.

    For each affected queue-day, the day's level is estimated from its unaffected
    intervals and the same weekday's average profile over the surrounding four
    weeks. Affected intervals then receive that level times their profile share.
    """
    out = intervals.copy()
    out["is_outage"] = outage_flags(out, events)
    out["offered_clean"] = out["offered"].astype(float)
    hit = out.loc[out["is_outage"], ["date", "queue"]].drop_duplicates()
    outage_days = set(zip(hit["date"], hit["queue"]))
    wide = out.pivot_table(index=["queue", "date"], columns="interval", values="offered", observed=True)
    for date, queue in outage_days:
        refs = [date + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        refs = [d for d in refs if (queue, d) in wide.index and (d, queue) not in outage_days]
        ref = wide.loc[[(queue, d) for d in refs]]
        share = (ref.div(ref.sum(axis=1), axis=0)).mean().to_numpy()
        rows = out.index[(out["date"] == date) & (out["queue"] == queue)]
        day = out.loc[rows].sort_values("interval")
        flagged = day["is_outage"].to_numpy()
        level = day.loc[~flagged, "offered"].sum() / max(share[~flagged].sum(), 1e-9)
        out.loc[day.index[flagged], "offered_clean"] = level * share[flagged]
    return out


def daily_actuals(clean: pd.DataFrame) -> pd.DataFrame:
    """Daily offered (raw and cleaned) and handle-weighted AHT per queue."""
    clean = clean.assign(talk=clean["aht_s"].fillna(0) * clean["handled"])
    daily = clean.groupby(["date", "queue"], observed=True).agg(
        offered=("offered", "sum"),
        offered_clean=("offered_clean", "sum"),
        handled=("handled", "sum"),
        talk=("talk", "sum"),
        is_outage_day=("is_outage", "any"),
    )
    daily["aht_s"] = daily["talk"] / daily["handled"]
    return daily.drop(columns=["talk", "handled"]).reset_index()


# ---------------------------------------------------------------------------
# Daily models
# ---------------------------------------------------------------------------


def campaign_days(events: pd.DataFrame) -> set[pd.Timestamp]:
    """Dates covered by a planned SALES campaign."""
    days: set[pd.Timestamp] = set()
    for ev in events[events["event_type"] == "campaign"].itertuples():
        days.update(pd.date_range(ev.start.normalize(), ev.end - pd.Timedelta(seconds=1), freq="D"))
    return days


def build_features(calendar: pd.DataFrame, dates: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    """Feature frame with one row per (date, queue)."""
    cal = pd.DataFrame({"date": dates}).merge(calendar, on="date", how="left")
    promo = campaign_days(events)
    frames = []
    for code, queue in enumerate(QUEUE_CODES):
        f = cal[["date", "dow", "month", "day_of_year", "is_holiday", "is_eid", "is_ramadan", "ramadan_day", "is_post_eid", "bill_cycle_day"]].copy()
        f["queue"] = queue
        f["queue_code"] = code
        f["is_campaign"] = f["date"].isin(promo) & (queue == "SALES")
        frames.append(f)
    feats = pd.concat(frames, ignore_index=True)
    bool_cols = ["is_holiday", "is_eid", "is_ramadan", "is_post_eid", "is_campaign"]
    feats[bool_cols] = feats[bool_cols].astype(int)
    return feats


def _trend(series: pd.Series, calendar: pd.DataFrame) -> tuple[float, float]:
    """Log-linear trend (level, slope per day) controlling for weekday and month.

    Holidays, Ramadan and post-Eid days are excluded so they do not tilt the slope.
    """
    cal = calendar.set_index("date").loc[series.index]
    keep = ~(cal["is_holiday"] | cal["is_ramadan"] | cal["is_post_eid"]).to_numpy()
    t = (series.index - series.index[0]).days.to_numpy()[keep]
    dummies = pd.get_dummies(cal.loc[keep, ["dow", "month"]].astype(str), drop_first=True).astype(float)
    x = sm.add_constant(np.column_stack([t, dummies.to_numpy()]))
    log_y = np.log(series.to_numpy()[keep])
    slope = sm.OLS(log_y, x).fit().params[1]
    # Level is the mean log volume net of trend, so the residual target is centred on zero.
    return float(np.mean(log_y - slope * t)), float(slope)


def forecast_daily(
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    events: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forecast daily offered volume for every queue with all three models.

    Args:
        daily: Output of :func:`daily_actuals` (uses ``offered_clean``).
        calendar: Calendar covering history and horizon.
        events: Event table (planned campaigns are known in advance).
        train_end: Last date the forecaster is allowed to see.
        horizon: Dates to forecast (must start after ``train_end``).

    Returns:
        Long frame: date, queue, model, forecast.
    """
    train = daily[daily["date"] <= train_end]
    wide = train.pivot(index="date", columns="queue", values="offered_clean")[QUEUE_CODES]
    gap_horizon = pd.date_range(train_end + pd.Timedelta(days=1), horizon.max(), freq="D")
    out = []

    for queue in QUEUE_CODES:
        y = wide[queue]
        last_week = y.iloc[-7:].to_numpy()
        naive = np.resize(last_week, len(gap_horizon))
        hw = ExponentialSmoothing(
            np.log(y.to_numpy()), trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
        ).fit(optimized=True)
        hw_fc = np.exp(hw.forecast(len(gap_horizon)))
        for model, values in (("Seasonal naive", naive), ("Holt-Winters", hw_fc)):
            out.append(pd.DataFrame({"date": gap_horizon, "queue": queue, "model": model, "forecast": values}))

    out.append(_lightgbm_forecast(wide, calendar, events, gap_horizon))
    result = pd.concat(out, ignore_index=True)
    return result[result["date"].isin(horizon)].reset_index(drop=True)


def _lightgbm_forecast(
    wide: pd.DataFrame, calendar: pd.DataFrame, events: pd.DataFrame, horizon: pd.DatetimeIndex
) -> pd.DataFrame:
    """Global LightGBM on calendar features; target is log volume minus each queue's log-linear trend."""
    origin = wide.index[0]
    trends = {q: _trend(wide[q], calendar) for q in QUEUE_CODES}
    train_x = build_features(calendar, wide.index, events)
    t_train = (train_x["date"] - origin).dt.days.to_numpy()
    level = train_x["queue"].map(lambda q: trends[q][0]).to_numpy()
    slope = train_x["queue"].map(lambda q: trends[q][1]).to_numpy()
    y = wide.melt(ignore_index=False, var_name="queue", value_name="y").reset_index()
    train_x = train_x.merge(y, on=["date", "queue"])
    target = np.log(train_x["y"].to_numpy()) - (level + slope * t_train)
    dataset = lgb.Dataset(train_x[FEATURES], target, categorical_feature=["queue_code"])
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=LGB_ROUNDS)

    test_x = build_features(calendar, horizon, events)
    t_test = (test_x["date"] - origin).dt.days.to_numpy()
    level_t = test_x["queue"].map(lambda q: trends[q][0]).to_numpy()
    slope_t = test_x["queue"].map(lambda q: trends[q][1]).to_numpy()
    pred = np.exp(model.predict(test_x[FEATURES]) + level_t + slope_t * t_test)
    return pd.DataFrame({"date": test_x["date"], "queue": test_x["queue"], "model": "LightGBM", "forecast": pred})


# ---------------------------------------------------------------------------
# Top-down interval forecast
# ---------------------------------------------------------------------------

ProfileKey = tuple[str, str, bool]  # (queue, day type, ramadan)


def _weighted_aht(frame: pd.DataFrame) -> np.ndarray:
    """Handle-weighted AHT per interval, lightly smoothed so sparse night intervals stay stable."""
    talk = (frame["aht_s"].fillna(0) * frame["handled"]).groupby(frame["interval"]).sum()
    calls = frame["handled"].groupby(frame["interval"]).sum()
    talk = talk.reindex(range(INTERVALS_PER_DAY), fill_value=0).rolling(3, center=True, min_periods=1).sum()
    calls = calls.reindex(range(INTERVALS_PER_DAY), fill_value=0).rolling(3, center=True, min_periods=1).sum()
    aht = (talk / calls.replace(0, np.nan)).to_numpy()
    return pd.Series(aht).ffill().bfill().to_numpy()


def _share_profile(frame: pd.DataFrame) -> np.ndarray:
    """Average share of daily volume per interval across the days in ``frame``."""
    wide = frame.pivot_table(index="date", columns="interval", values="offered_clean", observed=True)
    wide = wide.reindex(columns=range(INTERVALS_PER_DAY), fill_value=0.0)
    return wide.div(wide.sum(axis=1), axis=0).mean().to_numpy()


def learn_profiles(clean: pd.DataFrame, calendar: pd.DataFrame, train_end: pd.Timestamp) -> dict[ProfileKey, dict[str, np.ndarray]]:
    """Learn volume-share and AHT profiles per queue, day type and Ramadan flag.

    Normal profiles use the last eight weeks before ``train_end``. Ramadan
    profiles come from the most recent Ramadan in the training data; the Ramadan
    AHT uplift is measured against the eight weeks before that Ramadan and applied
    to today's normal AHT, so the AHT level stays current.
    """
    cal = calendar.assign(ptype=[profile_day_type(d, h) for d, h in zip(calendar["day_type"], calendar["is_holiday"])])
    hist = clean[(clean["date"] <= train_end) & ~clean["is_outage"]].merge(
        cal[["date", "ptype", "is_ramadan"]], on="date"
    )
    recent_start = train_end - pd.Timedelta(weeks=PROFILE_WEEKS)
    ram_dates = cal.loc[cal["is_ramadan"] & (cal["date"] <= train_end), "date"]
    last_ram_year = ram_dates.max().year
    last_ram = ram_dates[ram_dates.dt.year == last_ram_year]
    pre_ram_start = last_ram.min() - pd.Timedelta(weeks=PROFILE_WEEKS)

    profiles: dict[ProfileKey, dict[str, np.ndarray]] = {}
    for queue in QUEUE_CODES:
        q = hist[hist["queue"] == queue]
        mask = open_mask(queue)
        for ptype in ("weekday", "friday", "saturday"):
            q_type = q[q["ptype"] == ptype]
            normal = q_type[(q_type["date"] > recent_start) & ~q_type["is_ramadan"]]
            pre_ram = q_type[(q_type["date"] >= pre_ram_start) & (q_type["date"] < last_ram.min())]
            ram = q_type[q_type["date"].isin(last_ram)]
            normal_aht = _weighted_aht(normal)
            ram_uplift = _weighted_aht(ram) / _weighted_aht(pre_ram)
            for is_ram, vol_source, aht in ((False, normal, normal_aht), (True, ram, normal_aht * ram_uplift)):
                share = _share_profile(vol_source) * mask
                profiles[(queue, ptype, is_ram)] = {"share": share / share.sum(), "aht": np.where(mask, aht, np.nan)}
    return profiles


def interval_forecast(
    daily_fc: pd.DataFrame, calendar: pd.DataFrame, profiles: dict[ProfileKey, dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Spread a daily forecast (date, queue, forecast) over 30-minute intervals.

    Returns:
        One row per queue x interval with ``forecast_offered`` and ``forecast_aht_s``.
    """
    cal = calendar.set_index("date")
    rows = []
    for rec in daily_fc.itertuples():
        day = cal.loc[rec.date]
        prof = profiles[(rec.queue, profile_day_type(day["day_type"], day["is_holiday"]), bool(day["is_ramadan"]))]
        rows.append(
            pd.DataFrame(
                {
                    "interval_start": rec.date + pd.to_timedelta(np.arange(INTERVALS_PER_DAY) * INTERVAL_MINUTES, unit="min"),
                    "date": rec.date,
                    "interval": np.arange(INTERVALS_PER_DAY),
                    "queue": rec.queue,
                    "forecast_offered": rec.forecast * prof["share"],
                    "forecast_aht_s": prof["aht"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Accuracy and backtests
# ---------------------------------------------------------------------------


def wape(actual: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: sum |F - A| / sum A."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return float(np.abs(forecast - actual).sum() / actual.sum())


def bias(actual: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Forecast bias: sum (F - A) / sum A; positive means over-forecast."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return float((forecast - actual).sum() / actual.sum())


def _accuracy_table(frame: pd.DataFrame, actual_col: str, fc_col: str, by: list[str]) -> pd.DataFrame:
    """WAPE and bias per group plus an all-queues total per remaining group."""
    def summarise(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"wape": wape(g[actual_col], g[fc_col]), "bias": bias(g[actual_col], g[fc_col])})

    per_queue = frame.groupby(by + ["queue"], observed=True)[[actual_col, fc_col]].apply(summarise).reset_index()
    if by:
        total = frame.groupby(by, observed=True)[[actual_col, fc_col]].apply(summarise).reset_index()
    else:
        total = summarise(frame).to_frame().T
    total["queue"] = "All queues"
    return pd.concat([per_queue, total], ignore_index=True)


@dataclass
class BacktestResult:
    """Everything produced by one backtest window."""

    window: BacktestWindow
    daily: pd.DataFrame  # date, queue, model, forecast, actual, actual_raw
    daily_metrics: pd.DataFrame  # model, queue, wape, bias
    intervals: pd.DataFrame  # interval-level champion forecast vs actual
    interval_metrics: pd.DataFrame  # queue, wape, bias, aht_wape


def run_backtest(
    clean: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    events: pd.DataFrame,
    window: BacktestWindow,
    interval_model: str = "LightGBM",
) -> BacktestResult:
    """Re-forecast a historical window using only data up to ``window.train_end``.

    Accuracy is measured against cleaned actuals (outages removed): outages are
    not forecastable, and scoring against them would reward models that happen
    to over-forecast. Raw actuals are kept for charts.
    """
    horizon = pd.date_range(window.start, window.end, freq="D")
    fc = forecast_daily(daily, calendar, events, window.train_end, horizon)
    actual = daily[["date", "queue", "offered_clean", "offered"]].rename(columns={"offered_clean": "actual", "offered": "actual_raw"})
    fc = fc.merge(actual, on=["date", "queue"])
    daily_metrics = _accuracy_table(fc, "actual", "forecast", ["model"])

    profiles = learn_profiles(clean, calendar, window.train_end)
    ivl = interval_forecast(fc[fc["model"] == interval_model][["date", "queue", "forecast"]], calendar, profiles)
    ivl = ivl.merge(
        clean[["interval_start", "queue", "offered", "offered_clean", "aht_s", "handled", "is_outage"]],
        on=["interval_start", "queue"],
    )
    ivl = ivl[ivl["forecast_aht_s"].notna()]
    interval_metrics = _accuracy_table(ivl, "offered_clean", "forecast_offered", [])
    aht_eval = ivl[(ivl["handled"] > 0) & ~ivl["is_outage"]]
    aht_wape = {
        q: float((np.abs(g["forecast_aht_s"] - g["aht_s"]) * g["handled"]).sum() / (g["aht_s"] * g["handled"]).sum())
        for q, g in aht_eval.groupby("queue", observed=True)
    }
    aht_wape["All queues"] = float(
        (np.abs(aht_eval["forecast_aht_s"] - aht_eval["aht_s"]) * aht_eval["handled"]).sum()
        / (aht_eval["aht_s"] * aht_eval["handled"]).sum()
    )
    interval_metrics["aht_wape"] = interval_metrics["queue"].map(aht_wape)
    return BacktestResult(window, fc, daily_metrics, ivl, interval_metrics)


def select_champion(results: list[BacktestResult]) -> str:
    """Model with the lowest average all-queue daily WAPE across backtests."""
    scores = pd.concat([r.daily_metrics[r.daily_metrics["queue"] == "All queues"] for r in results])
    return str(scores.groupby("model")["wape"].mean().idxmin())


def forecast_forward(
    clean: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    events: pd.DataFrame,
    train_end: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Production forecast: daily (all models) and interval-level (chosen model) from ``start`` to ``end``."""
    horizon = pd.date_range(start, end, freq="D")
    fc = forecast_daily(daily, calendar, events, train_end, horizon)
    profiles = learn_profiles(clean, calendar, train_end)
    ivl = interval_forecast(fc[fc["model"] == model][["date", "queue", "forecast"]], calendar, profiles)
    return fc, ivl
