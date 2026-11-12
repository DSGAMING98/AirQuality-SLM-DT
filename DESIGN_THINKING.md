# Design Thinking Methodology

## Empathize

Student persona: needs plain-language interpretation and transparent units. Instructor/evaluator persona: needs evidence that data was genuinely cleaned, split, trained, and tested. Developer persona: needs commands that run on ordinary hardware and honest failure states. Shared problems include unexplained sensor codes, extensive missingness, temporal leakage, and the temptation to present rules as AI.

## Define

Create a small, reproducible language model that explains historical measurements relative to the training distribution without presenting them as live AQI or health advice. Success requires traceable source data, leakage-safe splits, measurable held-out behavior, and a UI that uses the actual checkpoint.

## Ideate

Alternatives considered: (1) rule-only dashboard—easy but not an SLM; (2) tabular classifier—learned but cannot naturally generate explanations; (3) large chat LLM—excessive compute and weak reproducibility; (4) FLAN-T5-small—compact, conditional generation, instruction-tuned, and suitable for short controlled outputs. Option 4 was selected while retaining deterministic preprocessing only for supervision creation.

## Prototype

The prototype includes source parsing, missing-marker conversion, chronological splits, train-only quartiles, classification and explanation prompt pairs, FLAN-T5 training/evaluation scripts, and a Streamlit interface. Model code fails transparently when the checkpoint is absent.

## Test

Completed: parsing, schema checks, counts, missingness, strict temporal ordering, split disjointness, output hashes. Blocked: training, loss collection, held-out generation, six-prompt robustness test, and UI inference. The next test cycle begins after dependency/model access is restored.
