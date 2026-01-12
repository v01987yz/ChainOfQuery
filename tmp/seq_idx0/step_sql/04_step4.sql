step4 AS (
  SELECT COUNT(DISTINCT user_pseudo_id) AS count_of_pseudo_users
  FROM step3
)
