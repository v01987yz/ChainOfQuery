WITH step1 AS (
  SELECT DISTINCT user_pseudo_id
  FROM `analytics_<property_id>.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
    AND event_name = 'user_activity'
)
