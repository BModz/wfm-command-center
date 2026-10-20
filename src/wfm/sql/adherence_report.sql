-- Agent adherence and conformance for the RTA replay day, with each agent's team average for comparison.
-- Parameters: none (uses the replay day in agent_states)
WITH per_agent AS (
    SELECT
        s.agent_id,
        a.team,
        SUM(s.scheduled_state <> 'Off / logged out') * 5                                     AS scheduled_min,
        SUM(s.scheduled_state <> 'Off / logged out' AND s.actual_state = s.scheduled_state) * 5 AS in_adherence_min,
        SUM(s.scheduled_state = 'On phone') * 5                                               AS scheduled_phone_min,
        SUM(s.actual_state = 'On phone') * 5                                                  AS actual_phone_min,
        SUM(s.actual_state = 'Unscheduled aux') * 5                                           AS aux_min
    FROM agent_states AS s
    JOIN agents AS a ON a.agent_id = s.agent_id
    GROUP BY s.agent_id, a.team
    HAVING scheduled_min > 0
)
SELECT
    agent_id,
    team,
    scheduled_min,
    scheduled_min - in_adherence_min                                         AS out_of_adherence_min,
    ROUND(100.0 * in_adherence_min / scheduled_min, 1)                       AS adherence_pct,
    ROUND(100.0 * actual_phone_min / NULLIF(scheduled_phone_min, 0), 1)      AS conformance_pct,
    aux_min,
    ROUND(AVG(100.0 * in_adherence_min / scheduled_min) OVER (PARTITION BY team), 1) AS team_adherence_pct
FROM per_agent
ORDER BY adherence_pct, out_of_adherence_min DESC;
