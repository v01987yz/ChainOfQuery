# Spider2 Sequential Executor Report

- Time: 2026-01-11 15:55:11
- split: train
- dialect: BigQuery
- model_steps: gpt-4o-mini
- model_step_sql: gpt-4o-mini
- prefer_table_suffix: True
- min_call_interval_s: 22.0
- mode: idx
- idx: 0
- instance_id: bq011
- db: ga4
- external_knowledge: ga4_obfuscated_sample_ecommerce.events.md
- temporal: None
- question: How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?
- doc_path: /Users/yangsongzhou/Year3/xlang-spider2/spider2-lite/resource/documents/ga4_obfuscated_sample_ecommerce.events.md

## Doc head

```text
# GA4 - BigQuery Export schema

This article explains the format and schema of the Google Analytics 4 property data and the Google Analytics for Firebase data that is exported to BigQuery.

## Datasets

For each Google Analytics 4 property and each Firebase project that is linked to BigQuery, a single dataset named "analytics_<property_id>" is added to your BigQuery project. Property ID refers to your Analytics Property ID, which you can find in the property settings for your Google Analytics 4 property, and in App Analytics Settings in Firebase. Each Google Analytics 4 property and each app for which BigQuery exporting is enabled will export its data to that single dataset.

## Tables

Within each dataset, a table named `events_YYYYMMDD` is created each day if the Daily export option is enabled.

If the Streaming export option is enabled, a table named`events_intraday_YYYYMMDD`is created. This table is populated continuously as events are recorded throughout the day. This table is deleted at the end of each day once `events_YYYYMMDD` is complete.

Not all devices on which events are triggered send their data to Analytics on the same day the events are triggered. To account for thi

[TRUNCATED]
```

## Steps JSON

```json
{
  "question": "How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?",
  "steps": [
    {
      "step_id": 1,
      "goal": "Identify users active in the last 7 days leading up to January 7, 2021.",
      "tables_or_entities": [
        "events_20210101",
        "events_20210102",
        "events_20210103",
        "events_20210104",
        "events_20210105",
        "events_20210106",
        "events_20210107"
      ],
      "filters_or_conditions": [
        "event_date BETWEEN '2020-12-31' AND '2021-01-07'",
        "event_type = 'user_activity'"
      ],
      "intermediate_output": "list_of_active_users_last_7_days",
      "depends_on_step_ids": []
    },
    {
      "step_id": 2,
      "goal": "Identify users active in the last 2 days (January 6 and 7, 2021).",
      "tables_or_entities": [
        "events_20210106",
        "events_20210107"
      ],
      "filters_or_conditions": [
        "event_date IN ('2021-01-06', '2021-01-07')",
        "event_type = 'user_activity'"
      ],
      "intermediate_output": "list_of_active_users_last_2_days",
      "depends_on_step_ids": [
        1
      ]
    },
    {
      "step_id": 3,
      "goal": "Determine users who were active in the last 7 days but not in the last 2 days.",
      "tables_or_entities": [],
      "filters_or_conditions": [
        "list_of_active_users_last_7_days NOT IN list_of_active_users_last_2_days"
      ],
      "intermediate_output": "list_of_pseudo_users_active_last_7_inactive_last_2",
      "depends_on_step_ids": [
        1,
        2
      ]
    },
    {
      "step_id": 4,
      "goal": "Count the number of pseudo users identified.",
      "tables_or_entities": [],
      "filters_or_conditions": [],
      "intermediate_output": "count_of_pseudo_users",
      "depends_on_step_ids": [
        3
      ]
    }
  ],
  "final_answer_derivation": "The final answer is the count of pseudo users who were active in the last 7 days and inactive in the last 2 days as of January 7, 2021."
}
```

## Per-step CTE SQLs

### step1: Identify users active in the last 7 days leading up to January 7, 2021.

```sql
WITH step1 AS (
  SELECT DISTINCT user_pseudo_id
  FROM `analytics_<property_id>.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
    AND event_name = 'user_activity'
)
```

Diagnostics:
```json
{
  "applied": false,
  "changed_patterns": []
}
```

### step2: Identify users active in the last 2 days (January 6 and 7, 2021).

```sql
SELECT DISTINCT user_pseudo_id
FROM `analytics_<property_id>.events_*`
WHERE _TABLE_SUFFIX IN ('20210106', '20210107')
  AND event_name = 'user_activity'
```

Diagnostics:
```json
{
  "applied": false,
  "changed_patterns": []
}
```

### step3: Determine users who were active in the last 7 days but not in the last 2 days.

```sql
SELECT user_pseudo_id
FROM step1
WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM step2)
```

Diagnostics:
```json
{
  "applied": false,
  "changed_patterns": []
}
```

### step4: Count the number of pseudo users identified.

```sql
step4 AS (
  SELECT COUNT(DISTINCT user_pseudo_id) AS count_of_pseudo_users
  FROM step3
)
```

Diagnostics:
```json
{
  "applied": false,
  "changed_patterns": []
}
```

## Final SQL

```sql
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
```
