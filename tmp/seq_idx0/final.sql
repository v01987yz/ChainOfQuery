WITH step1 AS (
WITH step1 AS (
  SELECT DISTINCT user_pseudo_id
  FROM `analytics_<property_id>.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
    AND event_name = 'user_activity'
)
),
step2 AS (
SELECT DISTINCT user_pseudo_id
FROM `analytics_<property_id>.events_*`
WHERE _TABLE_SUFFIX IN ('20210106', '20210107')
  AND event_name = 'user_activity'
),
step3 AS (
SELECT user_pseudo_id
FROM step1
WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM step2)
),
step4 AS (
step4 AS (
  SELECT COUNT(DISTINCT user_pseudo_id) AS count_of_pseudo_users
  FROM step3
)
)
SELECT * FROM step4;
