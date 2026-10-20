# Methodology

Formulas, modelling choices and assumptions behind each module. NileConnect is fictional and every input is a modelling assumption chosen to resemble a mid-size Egyptian telecom contact centre.

## 1. Definitions used throughout

| Metric | Definition |
|---|---|
| Offered | Calls that reached the queue |
| Handled | Offered minus abandoned |
| Service level (SL) | Calls answered within T seconds / calls offered (T = 20). Abandoned calls count against SL. |
| ASA | Average wait of answered calls |
| Abandon rate | Abandoned / offered |
| AHT | (Talk + hold + after-call work) / handled |
| Occupancy | Handle time / staffed (logged-in, available) time. Measures how busy agents are while available. |
| Shrinkage | Paid time in which agents are not available for calls (leave, training, coaching, sickness...). Measures how much paid time never reaches the phones. |
| Adherence | Share of scheduled time spent in the scheduled state at the scheduled time, in 5-minute slots |
| Conformance | Phone time actually worked / phone time scheduled, regardless of when |
| WAPE | sum abs(F - A) / sum A |
| Bias | sum (F - A) / sum A (positive = over-forecast) |

All interval metrics are volume-weighted when aggregated, never averaged across intervals.

## 2. Simulated data (`simulate.py`)

**Daily volume** for queue q on day d:

```
V(q,d) = base(q) × (1 + g_q)^years × weekday(q,d) × season(q,d) × billcycle(q,d)
         × campaign(q,d) × special(q,d) × exp(e_common(d) + e_q(d))
```

- **Weekday factors (Sunday to Saturday):** 1.13, 1.06, 1.03, 1.01, 0.96, 0.70, 0.86, normalised to a mean of 1. SALES has its own set.
- **Seasonality:** a mild annual cosine. AR_TECH gets a summer bump of +14% at the late-July peak; EN_SUPPORT gets +6%.
- **Billing cycle:** days 1-5 multiply AR_BILLING by 1.38, 1.30, 1.20, 1.12 and 1.06, with smaller effects on the other queues.
- **Special days:**
  - national holidays: 0.45-0.74 depending on the queue
  - Eid: 0.30-0.62
  - the first day after an Eid block: 1.10-1.24
  - Ramadan: 0.86-1.03, falling a further 14% in the last days before Eid
- **Noise:** AR(1) day-level noise with σ = 3.5% per queue (φ = 0.4), plus a common 2% shock shared by all queues.
- **Interval volume:** `Poisson(V × profile(q, daytype, ramadan)[i] × outage_multiplier[i])`.

**Intraday profiles** are sums of Gaussian bumps on a 24-hour circle, so late-night peaks wrap past midnight.

- **Friday:** the morning peak shifts about 1.5 hours later and there is a dip around Friday prayer.
- **Ramadan:** later morning peak, a sharp (~60%) dip centred on iftar at about 18:15-18:30, and a strong peak from 21:00 to 01:00.
- **SALES:** zero outside 09:00-23:00.

**Outages:** Poisson(14 a year) on AR_TECH, plus one fixed outage on the RTA replay day.
- Each lasts 2-6 hours, with peak volume of 2-3.5x and AHT at 82-92% of normal.
- Volume jumps within the first hour, then decays linearly to about 30% of the excess.
- AR_BILLING receives 8-14% of the excess as spillover.

**AHT** = base × 0.97 per year (slight improvement trend) × time-of-day factor (+12% overnight, -3% mid-day) × Ramadan factor (+5%, more late at night) × day noise. Realised interval AHT adds noise of σ ≈ 0.8 / sqrt(handled), reflecting a per-call coefficient of variation of about 0.8.

**Historical staffing and service.** The centre's past planning is simulated so the realised KPIs are believable:
1. Planned volume = true expected volume × daily plan error (σ = 6%) × interval error (σ = 5%).
2. Planned requirement = Erlang C at 80/20 with an 88% occupancy cap.
3. Scheduled = requirement × schedule-fit noise (AR(1), σ = 7%) / (1 - 4% absence allowance).
4. Staffed = Binomial(scheduled, 1 - absence). Absence averages about 7.5%, higher on Sundays, in Ramadan and after Eid.
5. Realised SL, abandon rate and ASA come from **Erlang A**, with mean patience of 300 / 360 / 240 / 180 seconds for the four queues. Counts are then sampled binomially.

Outage intervals are understaffed against their volume, so they show realistic collapses in service and spikes in abandons.

## 3. Queueing models (`erlang.py`)

**Traffic intensity:** A = calls × AHT / 1800 (Erlangs per 30-minute interval).

**Erlang B**, by the stable recursion B(0) = 1, B(n) = A·B(n-1) / (n + A·B(n-1)).

**Erlang C** (M/M/N, no abandonment):

```
P(wait) = N·B / (N - A·(1 - B))                for N > A
SL(T)   = 1 - P(wait) · exp(-(N - A)·T / AHT)
ASA     = P(wait) · AHT / (N - A)
Occupancy = A / N
```

**Required agents:** the smallest N with SL(T) ≥ target and A/N ≤ max occupancy. The search starts at max(floor(A) + 1, ceil(A / occ_max)) and updates Erlang B incrementally, so tens of thousands of intervals are solved in one vectorised pass. Check: 100 calls, 180s AHT means A = 10. At N = 13, SL is 79.56%; at **N = 14, SL is 88.84%**, ASA is 7.8s and occupancy 71.4%.

**Erlang A** (M/M/N+M, exponential patience with mean 1/θ), solved as a birth-death chain:

- arrival rate λ, departure rate μ·min(n, N) + θ·max(n - N, 0)
- p(n) ∝ Π λ / death(k), computed in log space and truncated at N + Q, where Q covers the mean overload queue plus 8 standard deviations
- a caller who finds j callers queued ahead goes through stages i = 0..j, each with success rate s_i = Nμ + iθ and own-abandon rate θ

From that:

- P(answered | j) = Π s_i / (s_i + θ)
- E[wait | answered, j] = Σ 1 / (s_i + θ)
- P(answered within T | j) is computed by uniformisation of the stage chain, at rate Λ = Nμ + Qθ, with Poisson(ΛT) weights
- SL = P(no wait) + Σ_j π_j P(answered ≤ T | j), with π_j = p(N + j) (arrivals see time averages)
- abandon rate = 1 - P(answered)
- occupancy = λ·P(answered) / (Nμ)

Tests confirm two properties:
- With infinite patience, Erlang A reproduces Erlang C to within 1e-4.
- Abandon rate = θ·E[queue length] / λ (Little's law) holds to within 1e-6.

## 4. Forecasting (`forecast.py`)

**Anomaly treatment.** Intervals inside an outage window (AR_TECH and its AR_BILLING spillover) are replaced by a baseline:
1. Take the same weekday's average intraday shares from the surrounding ±4 weeks (excluding other outage days).
2. Scale them to the day's level, estimated from its unaffected intervals.

Outages are unforecastable. Left in, they would raise the weekday level and distort the profile, so accuracy is scored against cleaned actuals and outages are handled by the RTA desk.

**Models (daily, per queue):**

- **Seasonal naive:** repeat the last observed week.
- **Holt-Winters:** additive damped trend plus additive weekly seasonality, fitted on log volume (multiplicative in levels).
- **LightGBM (global, all queues in one model):**
  - Target: log(volume) - (level_q + slope_q · t).
  - The log-linear trend per queue is fitted by OLS with weekday and month dummies, excluding holidays, Ramadan and post-Eid days. Removing the trend first lets the tree model extrapolate growth, which trees cannot do on their own.
  - Features: queue, weekday, month, day of year, holiday, Eid, Ramadan flag, Ramadan day index, post-Eid, billing-cycle day, campaign flag. Campaigns are planned in advance, so they are known for the horizon.
  - L1 objective, 600 rounds, learning rate 0.03, 15 leaves, deterministic.

**Champion:** the lowest average all-queue WAPE across both backtests (LightGBM).

**Top-down interval forecast:**
- interval volume = daily forecast × share profile(queue, day type, Ramadan); holidays use the Friday profile
- **normal profiles:** the last 8 weeks before the forecast origin
- **Ramadan profiles:** the most recent Ramadan in the training data
- **interval AHT:** handle-weighted AHT per interval over the last 8 weeks, smoothed over 3 intervals; in Ramadan it is multiplied by the uplift observed last Ramadan relative to the 8 weeks before it

**Backtests**, with no leakage: every model sees only data up to `train_end`, which a test enforces by multiplying all future actuals by 10 and asserting identical forecasts.
- Normal: train to 1 August 2026, forecast 2 August to 26 September 2026 (8 Sunday-Saturday weeks).
- Ramadan: train to 7 February 2026, forecast 8 February to 28 March 2026 (Ramadan 2026 plus Eid al-Fitr).

**Production forecast:** 26 weeks from Sunday 4 October 2026 to 3 April 2027, covering Ramadan 2027.

## 5. Capacity (`capacity.py`)

**Shrinkage** (defaults, all editable in the app):

| Planned | | Unplanned | |
|---|---|---|---|
| Annual leave | 8.1% (21 days / 260 working days) | Sickness | 4.0% |
| Training | 2.5% | Absence | 3.0% |
| Coaching | 2.0% | Aux overrun | 1.5% |
| Meetings | 1.0% | | |

Compounded: total = 1 - Π(1 - s_i) = 20.3%, against an additive 22.1%. **In-day shrinkage** (coaching, meetings, aux overrun = 4.4%) is what the scheduler must cover inside shifts.

**Required FTE (weekly):** Σ (Erlang C agents × 0.5h) / (1 - total shrinkage) / 40h.

**Supply:**
- Tenured FTE decays at weekly attrition w = 1 - (1 - 3.5%)^(12/52) = 0.82% a week.
- A class of size n starting in week s is 0% productive for 4 training weeks, then 60% for 2 nesting weeks, then 100%.
- 15% of the class is lost during training (spread geometrically over the training weeks); regular attrition applies after that.
- Available FTE = tenured + Σ class heads × productivity.

**Hiring recommender (greedy):**
1. Smooth the requirement with a centred 5-week rolling median (full windows only), so isolated spike weeks do not drive permanent hires.
2. Find the first week, at or after lead time + training, where available FTE is more than 2% below the baseline.
3. Book a class starting max(lead time, week - 6), so it is proficient that week.
4. Size it as the larger of the trigger-week shortfall and the average shortfall over the next 8 weeks, divided by the class's productive yield in those weeks. Bounds are 6 to 25 hires; a full class rolls over to the next week.
5. Repeat until no fixable shortfall remains.

## 6. Scheduling (`scheduling.py`)

**Shift patterns:**
- 18 intervals (9 hours) on site with a 2-interval (1-hour) unpaid break, so 8 paid hours.
- The break may start after 6-10 intervals, so neither working block exceeds 5 hours (labour law). That gives 5 break options per start.
- Any 30-minute start is allowed; a cyclic week has 336 starts, so 1,680 integer variables.

**Programme:**

```
minimise   Σ_k (1 + ε·breakoffset_k)·x_k + p · Σ_t u_t
s.t.       Σ_k a_tk·x_k + u_t ≥ r_t        for all intervals t
           x_k ∈ Z≥0,  u_t ≥ 0
```

- r_t = ceil(Erlang C agents / (1 - in-day shrinkage)).
- a_tk = 1 if pattern k is on the phones (not on break) in interval t.
- p = 0.3: one uncovered agent-interval costs 0.3 shifts, so a shift is added only if it removes at least 4 short agent-intervals.
- The tiny tie-break term ε removes symmetric optima and speeds up branch and bound.
- Solved with scipy's `milp` (HiGHS) using a 1% relative gap and a 30-second limit; the achieved gap is reported.

**Outputs:**
- coverage % = Σ min(cov, r) / Σ r
- efficiency = Σ min(cov, r) / Σ cov
- over/under by interval
- FTE equivalent = shifts / 5

The contract (5 × 8h = 40h, 2 rest days) is checked against the legal 48-hour week and 24-hour weekly rest.

## 7. Real time (`realtime.py`)

**Replay setup:** AR_TECH on Tuesday 22 September 2026.
- **Schedule:** optimised on a forecast made 15 days before the week, the usual publication lead.
- **Intraday plan:** the forecast made the evening before (training ends 21 September).
- **Roster:** every shift overlapping the day becomes a named agent in teams of 14, including the previous night's shifts.
- **Coaching and offline blocks:** 30 or 60 minutes, weighted towards 13:00-17:30, until they absorb the planned coaching and meeting share.

**Agent behaviour (5-minute slots):**

| Behaviour | Agents affected | Size |
|---|---|---|
| No-show | 4% | whole shift |
| Late login | 15% | 5-20 min |
| Early logout | 5% | 5-15 min |
| Break taken early or late | 25% | 5-15 min, full length |
| Extended break | 22% | 5-15 min |
| Unscheduled aux | 35% | 5-20 min |
| Extra system aux during the incident | 10% | short blocks |

Staffed agents per interval = mean on-phone count across its six slots. Realised SL, abandon rate and ASA come from Erlang A on the actual volume and AHT.

**Reforecast at time "now":**
- r = Σ w·actual / Σ w·forecast over completed intervals, with weights halving every 2 intervals.
- **Normally:** surge intervals (≥1.5x forecast) are excluded and the base ratio persists for the rest of the day.
- **During an incident:** factor(t) = base + (r - base)·exp(-steps/4), so the surge decays back to the run rate.
- AHT is reforecast with the same weighted ratio.

**Projected staffing:** scheduled on-phone × the realisation rate (staffed / scheduled) over the last 2 hours, clipped to 0.70-1.05.

**Alerts:**
- interval SL < 70%
- abandon rate > 8%
- AHT drift > 10% (with at least 30 calls)
- volume > 20% over forecast (with at least 30 calls)
- **possible incident:** the last interval at ≥2x forecast, or the last two at ≥1.5x

**Actions,** over the next 4 hours and only in intervals where Erlang A projects SL below 80%, applied cheapest first:
1. **IVR outage message and callback:** deflect 30% of incident-driven excess volume (an assumption).
2. **Cancel coaching or offline sessions** that fall in short intervals; agents are named.
3. **Delay breaks** out of short intervals to the first interval that keeps the break start within 5 hours of the shift start.
4. **Overtime** for the remaining gap: ceil(max gap) agents, up to 2 hours each. The premium is +35% by day and +70% at night (22:00-06:00, a site-policy assumption).
5. **Voluntary time off (VTO)** when the projected surplus is at least max(2, 12% of requirement) for 2 hours or more.

The projected SL before and after the actions is computed with Erlang A.

## 8. Labour-law assumptions (`labour.py`)

These are from public summaries of Egypt's Labour Law No. 14 of 2025 (in force from September 2025). Secondary sources disagree on some figures and the executive regulations were still being finalised, so these are planning assumptions, not legal advice.

- **Hours:** 8 working hours a day, 48 a week.
- **Breaks:** at least 1 hour, with no more than 5 consecutive hours of work.
- **Weekly rest:** at least 24 consecutive hours.
- **Overtime:** at most 2 hours a day, paid at +35% by day and +70% at night.
- **Annual leave:** 21 days.

## 9. Limitations

- **The data is synthetic.** Absolute accuracy figures show the method, not what any real centre would achieve. Real data has messier drivers: marketing, IT releases, weather.
- **Ramadan, Eid and other lunar-calendar dates are approximate.**
- **Erlang models assume Poisson arrivals and steady state within each interval.** Erlang A also assumes exponential patience.
- **Each queue is planned as a separate skill group.** Multi-skill routing and cross-skilled agents are out of scope.
- **The scheduler produces shift counts, not named rosters.** Assigning days off over a multi-week rotation is a separate step.
- **IVR deflection rates and agent-behaviour probabilities are assumptions,** chosen to sit inside typical industry ranges.
