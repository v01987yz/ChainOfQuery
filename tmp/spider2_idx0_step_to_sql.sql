WITH active_users_last_7_days AS (
  SELECT DISTINCT user_pseudo_id AS user_id
  FROM `analytics_<property_id>.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
),
active_users_last_2_days AS (
  SELECT DISTINCT user_pseudo_id AS user_id
  FROM `analytics_<property_id>.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210106' AND '20210107'
),
inactive_users AS (
  SELECT user_id
  FROM active_users_last_7_days
  WHERE user_id NOT IN (SELECT user_id FROM active_users_last_2_days)
)
SELECT COUNT(DISTINCT user_id) AS inactive_user_count
FROM inactive_users;
