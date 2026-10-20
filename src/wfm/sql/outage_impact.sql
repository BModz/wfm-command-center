-- Impact of every network outage in the period: volume, abandons and service level inside the window.
-- Parameters: :start, :end (YYYY-MM-DD)
SELECT
    e.event_id,
    e.start                                                     AS outage_start,
    e.duration_h,
    i.queue,
    SUM(i.offered)                                              AS offered,
    ROUND(SUM(i.planned_offered), 0)                            AS forecast,
    SUM(i.abandoned)                                            AS abandoned,
    ROUND(100.0 * SUM(i.answered_in_sl) / SUM(i.offered), 1)    AS sl_pct,
    ROUND(100.0 * SUM(i.abandoned) / SUM(i.offered), 1)         AS abandon_pct
FROM events AS e
JOIN intervals AS i
  ON i.queue IN (e.queue, e.spillover_queue)
 AND i.interval_start >= e.start
 AND i.interval_start <  e."end"
WHERE e.event_type = 'outage'
  AND date(e.start) BETWEEN :start AND :end
GROUP BY e.event_id, i.queue
ORDER BY e.start, i.queue;
