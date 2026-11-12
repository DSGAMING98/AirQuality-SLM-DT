# Project Report

## Applying Design Thinking to Build a Small Language Model on Air Quality Data

This research prototype investigates whether a compact instruction-tuned encoder-decoder can translate historical sensor measurements into controlled natural-language interpretations. The contribution completed in the present environment is a reproducible supervision pipeline rather than a claim of trained-model performance.

The UCI Air Quality source contains 9,357 hourly observations across 15 variables and uses `-200` for missing readings. The implementation validates the schema, converts decimal commas, removes the missing sentinel, combines date/time, checks duplicate timestamps, and sorts chronologically. It then allocates 70%/15%/15% of source hours to train, validation, and test. Splitting before prompt generation ensures that the classification and explanation examples derived from one hour cannot appear in multiple splits.

Quartiles are estimated from training data only. CO thresholds are 1.1 and 2.8 mg/m³, NO₂ thresholds are 69 and 123 µg/m³, and benzene thresholds are 5.1 and 14.6 µg/m³. Each measurement is labelled low, typical, elevated, or unavailable. These categories are relative to this historical dataset and are not AQI or medical thresholds. Two text-to-text examples are produced per hour, yielding 13,098 training, 2,808 validation, and 2,808 test examples.

The proposed SLM is FLAN-T5-small, an approximately 80M-parameter encoder-decoder suitable for short conditional generation on an 8 GB GPU. The repository contains full fine-tuning, checkpointing, held-out generation, exact-match/F1 evaluation, and Streamlit inference code. However, the required libraries and base checkpoint could not be installed because network DNS resolution failed. Training, validation loss, test scores, prompt responses, screenshots, and publication therefore remain blocked and are not fabricated.

The verified finding is that the numerical dataset can be transformed into traceable, leakage-safe language supervision with documented provenance and limitations. No experimentally supported conclusion about FLAN-T5 accuracy can be made until the blocked training and evaluation stages run. The next work item is to execute the documented commands in a network-enabled Python environment, audit the resulting JSON metrics and samples, then publish the dataset and successful model separately with verified cards.
