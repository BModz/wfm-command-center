-- The 10 worst service-level intervals per queue in a period: the starting list for an RCA.
-- Parameters: :start, :end (YYYY-MM-DD)
WITH ranked AS (
    SELECT
        queue,
        interval_start,
        offered,
        ROUND(planned_offered, 0)                                  AS forecast,
        ROUND(100.0 * answered_in_sl / offered, 1)                 AS sl_pct,
        ROUND(100.0 * abandoned / offered, 1)                      AS abandon_pct,
        staffed_agents - required_agents                           AS over_under,
        ROW_NUMBER() OVER (PARTITION BY queue ORDER BY 1.0 * answered_in_sl / offered, offered DESC) AS rn
    FROM intervals
    WHERE date BETWEEN :start AND :end
      AND offered >= 30
)
SELECT
    queue,
    interval_start,
    offered,
    forecast,
    sl_pct,
    abandon_pct,
    over_under,
    CASE
        WHEN offered > 1.5 * forecast THEN 'Demand spike'
        WHEN over_under < 0 THEN 'Understaffed vs requirement'
        ELSE 'Variance (AHT or arrivals)'
    END AS likely_driver
FROM ranked
WHERE rn <= 10
ORDER BY queue, sl_pct;
