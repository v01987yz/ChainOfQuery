```sql
WITH step1 AS (
    SELECT DISTINCT user_pseudo_id AS customer_id
    FROM `analytics_<property_id>.events_*`
    WHERE event_name = 'purchase'
      AND totals.product = 'Youtube Men’s Vintage Henley'
      AND _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
),
step2 AS (
    SELECT totals.product, user_pseudo_id AS customer_id
    FROM `analytics_<property_id>.events_*`
    WHERE user_pseudo_id IN (SELECT customer_id FROM step1)
      AND _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
),
step3 AS (
    SELECT product, COUNT(*) AS product_count
    FROM step2
    WHERE product != 'Youtube Men’s Vintage Henley'
    GROUP BY product
),
step4 AS (
    SELECT product
    FROM step3
    ORDER BY product_count DESC
    LIMIT 1
)
SELECT product FROM step4;
```
