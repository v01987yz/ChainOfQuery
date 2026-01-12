SELECT DISTINCT user_pseudo_id
FROM `analytics_<property_id>.events_*`
WHERE _TABLE_SUFFIX IN ('20210106', '20210107')
  AND event_name = 'user_activity'
