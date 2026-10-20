-- Daily KPI summary with a 7-day rolling service level and week-on-week volume change.
-- The windows look back before :start, so the date filter is applied after them.
-- Parameters: :queue, :start, :end (YYYY-MM-DD)
WITH daily AS (
    SELECT
        date,
        SUM(offered)                                          AS offered,
        SUM(answered_in_sl)                                   AS in_sl,
        SUM(abandoned)                                        AS abandoned,
        SUM(handled)                                          AS handled,
        SUM(aht_s * handled) / NULLIF(SUM(handled), 0)        AS aht_s,
        SUM(asa_s * handled) / NULLIF(SUM(handled), 0)        AS asa_s
    FROM intervals
    WHERE queue = :queue
    GROUP BY date
),
kpis AS (
    SELECT
        date,
        offered,
        ROUND(100.0 * in_sl / offered, 1)                                         AS sl_pct,
        ROUND(100.0 * SUM(in_sl) OVER w7 / SUM(offered) OVER w7, 1)               AS sl_7day_pct,
        ROUND(100.0 * abandoned / offered, 1)                                     AS abandon_pct,
        ROUND(asa_s, 0)                                                           AS asa_s,
        ROUND(aht_s, 0)                                                           AS aht_s,
        ROUND(100.0 * (offered - LAG(offered, 7) OVER (ORDER BY date))
              / LAG(offered, 7) OVER (ORDER BY date), 1)                          AS vol_vs_last_week_pct
    FROM daily
    WINDOW w7 AS (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
)
SELECT *
FROM kpis
WHERE date BETWEEN :start AND :end
ORDER BY date;
