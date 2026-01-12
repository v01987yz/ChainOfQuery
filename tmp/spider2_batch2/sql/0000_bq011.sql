```sql
WITH step1 AS (
  SELECT DISTINCT user_pseudo_id
  FROM `analytics_<property_id>.events_*`
  WHERE event_date BETWEEN '20210101' AND '20210107'
    AND event_name = 'pseudo_user_activity'
    AND _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
),
step2 AS (
  SELECT DISTINCT user_pseudo_id
  FROM `analytics_<property_id>.events_*`
  WHERE event_date IN ('20210106', '20210107')
    AND event_name = 'pseudo_user_activity'
    AND _TABLE_SUFFIX BETWEEN '20210106' AND '20210107'
),
step3 AS (
  SELECT user_pseudo_id
  FROM step1
  WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM step2)
)
SELECT COUNT(*) AS inactive_user_count
FROM step3;
```
