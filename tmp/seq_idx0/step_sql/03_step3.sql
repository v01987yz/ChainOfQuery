SELECT user_pseudo_id
FROM step1
WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM step2)
