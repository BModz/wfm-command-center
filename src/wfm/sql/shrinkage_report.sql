-- Weekly in-day shrinkage (scheduled phone time lost to absence and aux), ranked within each queue.
-- Parameters: :start, :end (YYYY-MM-DD)
WITH weekly AS (
    SELECT
        i.queue,
        c.week_start,
        SUM(i.scheduled_agents) / 2.0  AS scheduled_hours,
        SUM(i.staffed_agents) / 2.0    AS staffed_hours
    FROM intervals AS i
    JOIN calendar AS c ON c.date = i.date
    WHERE i.date BETWEEN :start AND :end
    GROUP BY i.queue, c.week_start
)
SELECT
    queue,
    week_start,
    ROUND(scheduled_hours, 0)                                              AS scheduled_hours,
    ROUND(staffed_hours, 0)                                                AS staffed_hours,
    ROUND(scheduled_hours - staffed_hours, 0)                              AS hours_lost,
    ROUND(100.0 * (1 - staffed_hours / scheduled_hours), 1)                AS in_day_shrinkage_pct,
    ROUND(100.0 * AVG(1 - staffed_hours / scheduled_hours) OVER (PARTITION BY queue), 1) AS queue_average_pct,
    RANK() OVER (PARTITION BY queue ORDER BY 1 - staffed_hours / scheduled_hours DESC) AS worst_week_rank
FROM weekly
ORDER BY queue, week_start;
