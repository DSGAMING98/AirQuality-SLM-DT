# Dataset Card: AirQuality-SLM Processed

## Provenance

Derived from the UCI Air Quality dataset (ID 360), DOI `10.24432/C59K5F`. Supplied source SHA-256: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.

The source contains 9,357 dated hourly observations (9,358 rows when the header is counted) and 114 additional blank tail rows in the CSV. Dates span 10 March 2004 18:00 to 4 April 2005 14:00.

## Schema

Each processed row contains `text`, `label`, `task`, `record_id`, `timestamp`, `input_text`, and `target_text`. `text`/`input_text` hold the instruction; `label`/`target_text` hold the deterministic answer. `task` is `classify` or `explain`. `record_id` is the source hour and exists for leakage audits.

## Processing

The parser uses a semicolon delimiter and decimal-comma conversion. Empty and `-200` values become missing. `Date` and `Time` become an ISO timestamp. Rows are deduplicated by timestamp and chronologically sorted. The split occurs before two prompts are generated for each hour. Quartiles for CO, NO₂, and benzene are calculated only from non-missing training readings, then frozen for validation and test.

| Split | Source hours | Examples | SHA-256 |
|---|---:|---:|---|
| train | 6,549 | 13,098 | `d28759a9816ad2d50226fbd2df0eea69e2cbdf64ed04d7127f9e3d9316a587a4` |
| validation | 1,404 | 2,808 | `d684890c81a02fd6f4aa31c2576ce804a34e74764fedc8325bfe06f50b2313d0` |
| test | 1,404 | 2,808 | `470f28ced793f4504bfd3050a9873a5dc28875408de1d7a853f84336649617f9` |

## Missingness

`NMHC(GT)` is missing in 8,443 observations; `CO(GT)` 1,683; `NO2(GT)` 1,642; `NOx(GT)` 1,639. Each of `PT08.S1`, `C6H6`, `PT08.S2`, `PT08.S3`, `PT08.S4`, `PT08.S5`, `T`, `RH`, and `AH` has 366 missing observations.

## Intended use and limitations

Intended for educational text-to-text fine-tuning and reproducibility exercises. Labels are generated from historical distribution quartiles; they are not human annotations, AQI categories, legal thresholds, or medical advice. The dataset is old, single-site, subject to sensor drift, and lacks geospatial/source labels. Check the UCI source terms before redistribution or commercial use.
