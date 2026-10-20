-- Plan vs actual by week: forecast accuracy and staffing delivery per queue.
-- Parameters: :start, :end (YYYY-MM-DD)
WITH weekly AS (
    SELECT
        i.queue,
        c.week_start,
        SUM(i.planned_offered)                       AS forecast,
        SUM(i.offered)                               AS offered,
        SUM(ABS(i.offered - i.planned_offered))      AS abs_error,
        SUM(i.scheduled_agents) / 2.0                AS scheduled_hours,
        SUM(i.staffed_agents) / 2.0                  AS staffed_hours,
        SUM(i.required_agents) / 2.0                 AS required_hours,
        SUM(i.answered_in_sl)                        AS in_sl
    FROM intervals AS i
    JOIN calendar AS c ON c.date = i.date
    WHERE i.date BETWEEN :start AND :end
    GROUP BY i.queue, c.week_start
)
SELECT
    queue,
    week_start,
    ROUND(forecast, 0)                                         AS forecast,
    offered,
    ROUND(100.0 * (offered - forecast) / forecast, 1)          AS bias_pct,
    ROUND(100.0 * abs_error / offered, 1)                      AS interval_wape_pct,
    ROUND(required_hours, 0)                                   AS required_hours,
    ROUND(staffed_hours, 0)                                    AS staffed_hours,
    ROUND(100.0 * staffed_hours / scheduled_hours, 1)          AS staffed_vs_scheduled_pct,
    ROUND(100.0 * in_sl / offered, 1)                          AS sl_pct
FROM weekly
ORDER BY queue, week_start;
