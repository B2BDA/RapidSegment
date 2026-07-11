-- sample SQL script to downsample a dataset in BigQuery

CREATE OR REPLACE TABLE `your_dataset.downsampled_training_data` AS
SELECT * 
FROM `your_project.your_dataset.your_table`
WHERE target_y = 1  -- Keep 100% of minority class

UNION ALL

SELECT * 
FROM `your_project.your_dataset.your_table`
WHERE target_y = 0
  -- Generates a deterministic pseudo-random float between 0 and 1 per row
  AND ABS(MOD(FARM_FINGERPRINT(CAST(row_id AS STRING)), 100)) < 10; -- Keeps exactly 10% of Class 0