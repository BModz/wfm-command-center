-- Interval report: one queue, one day, plan vs actual every 30 minutes.
-- Parameters: :queue, :date (YYYY-MM-DD)
SELECT
    substr(interval_start, 12, 5)                                   AS interval,
    ROUND(planned_offered, 0)                                       AS forecast,
    offered,
    ROUND(100.0 * (offered - planned_offered) / NULLIF(planned_offered, 0), 1) AS volume_var_pct,
    handled,
    abandoned,
    ROUND(100.0 * answered_in_sl / NULLIF(offered, 0), 1)           AS sl_pct,
    ROUND(asa_s, 0)                                                 AS asa_s,
    ROUND(aht_s, 0)                                                 AS aht_s,
    required_agents                                                 AS required,
    scheduled_agents                                                AS scheduled,
    staffed_agents                                                  AS staffed,
    staffed_agents - required_agents                                AS over_under,
    CASE
        WHEN offered = 0 THEN 'Closed'
        WHEN 1.0 * answered_in_sl / offered < 0.70 THEN 'Below 70%'
        WHEN 1.0 * answered_in_sl / offered < 0.80 THEN 'Below target'
        ELSE 'On target'
    END                                                             AS status
FROM intervals
WHERE queue = :queue
  AND date = :date
ORDER BY interval_start;
