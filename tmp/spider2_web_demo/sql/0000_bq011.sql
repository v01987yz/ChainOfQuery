```sql
WITH step1 AS (
    SELECT DISTINCT user_id
    FROM `analytics_<property_id>.events_*`
    WHERE event_date BETWEEN '20210101' AND '20210107'
),
step2 AS (
    SELECT DISTINCT user_id
    FROM `analytics_<property_id>.events_*`
    WHERE event_date IN ('20210106', '20210107')
),
step3 AS (
    SELECT s1.user_id
    FROM step1 s1
    LEFT JOIN step2 s2 ON s1.user_id = s2.user_id
    WHERE s2.user_id IS NULL
)
SELECT COUNT(*) AS inactive_users_count
FROM step3;
```
