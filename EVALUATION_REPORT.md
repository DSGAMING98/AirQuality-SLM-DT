# Evaluation Report

## Status

Data validation is **Completed**. Model evaluation is **Blocked / Not Run** because a trained checkpoint does not exist.

## Verified data results

- 9,357 valid dated observations; 0 timestamp parse failures; 0 duplicates.
- 6,549 / 1,404 / 1,404 chronological source hours.
- 13,098 / 2,808 / 2,808 generated instruction examples.
- Strictly increasing split boundaries and zero `record_id` overlap.
- Train-only quartiles: CO 1.1/2.8 mg/m³, NO₂ 69/123 µg/m³, benzene 5.1/14.6 µg/m³.

## Model evaluation protocol

The held-out test split is generated only after thresholds are learned from train. `src/evaluate.py` performs autoregressive generation and reports normalized exact match and token-level F1. It stores 20 example predictions for audit. Validation loss guides checkpoint selection; the test split is used once for final reporting.

## Actual model results

| Metric | Base FLAN-T5-small | Fine-tuned model |
|---|---:|---:|
| Validation loss | Not Run | Not Run |
| Test loss | Not Run | Not Run |
| Exact match | Not Run | Not Run |
| Token F1 | Not Run | Not Run |

No prediction examples or graphs are available. The quartile generator is a deterministic label-production rule and is not reported as SLM performance.
