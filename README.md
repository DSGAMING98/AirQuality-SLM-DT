# AirQuality SLM
### From-scratch 30.3M-parameter Transformer, public Ollama model, and local two-model conversational assistant

[![Ollama](https://img.shields.io/badge/Ollama-Public_Model-111111?logo=ollama)](https://ollama.com/db0729713/airquality-slm)
[![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Air_Quality-2563EB)](https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed)
[![License](https://img.shields.io/badge/Model_Weights-Apache--2.0-16A34A)](https://ollama.com/db0729713/airquality-slm)

**Latest project milestone (19 November 2026):** The original MiniQwen-inspired SLM has been trained **from random weights**, evaluated, converted to a working Ollama model, and **published** as [`db0729713/airquality-slm`](https://ollama.com/db0729713/airquality-slm). A separate *local* conversational application combines that classifier with the pretrained [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model via Ollama. The conversational integration and its launcher scripts remain in the local `codex/ollama-demo` workspace and **are not yet committed to this repository's `main` branch**.

> **Scientific boundary:** `low`, `typical`, and `elevated` are labels **relative to the historical training distribution**, *not* official Air Quality Index categories, regulatory limits, medical advice, or a live air-quality forecast.

## Run the public model with Ollama

Install [Ollama](https://ollama.com/download), then run:

```powershell
ollama run db0729713/airquality-slm
```

This command downloads the **original scratch-trained classifier (Model A)**. It is **not a general-purpose conversational assistant**. For a meaningful test, paste a supported historical instruction into the Ollama prompt:

```text
Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3.
```

Expected response for that verified example:

```text
CO: elevated; NO2: elevated; benzene: elevated.
```

Type `/bye` to leave the interactive Ollama session. The standalone public model generates task-specific text; the Python application adds input validation, missing-value safeguards, routing, and conversational context. Do not assume the bare `ollama run` command includes all application-level guardrails.

## Run the full conversational assistant (local development workspace)

The **two-model** system was reported as tested locally on Windows with an RTX 4060 8 GB GPU. From the existing project directory **containing the as-yet-unmerged launcher files**, use:

```powershell
ollama pull db0729713/airquality-slm
ollama pull qwen2.5:1.5b

# In the local development folder with the conversational scripts:
.\launch_conversational.ps1       # Terminal chat
.\launch_conversational_ui.ps1    # Streamlit graphical chat
```

> **Repository availability:** The two `launch_conversational*.ps1` files, `conversational_airquality.py`, `conversational_streamlit.py`, and newer conversational verification documents are **local deliverables reported from the `codex/ollama-demo` workspace**, not files currently available from `git clone` of this `main` branch. To reproduce the *complete chatbot* from GitHub alone, those files and their dependencies must first be committed/pushed. The public Model A is already independently downloadable.

### Project milestones

| Phase | Contribution | Current status |
|---|---|---|
| **1. Data engineering** | Validate UCI data; chronological split; training-only quartiles; missingness handling | Published dataset |
| **2. Original scratch SLM** | Custom 30.3M-parameter decoder-only Transformer; learned classification and explanation | Model checkpoint published on Hugging Face |
| **3. Ollama deployment** | Weight-preserving export, corrected chat-template metadata, clean-pull verification | **Model A publicly available** |
| **4. Conversation layer** | Qwen2.5-1.5B-Instruct, routing, conversation history, validation, CLI and Streamlit UI | **Locally verified; not merged to `main`** |

## Repository and published artifacts

| Artifact | Location | What is verified |
|---|---|---|
| GitHub source and documentation | [DSGAMING98/AirQuality-SLM-DT](https://github.com/DSGAMING98/AirQuality-SLM-DT) | Public data pipeline, custom architecture, checkpoint-loading code, and historical tests |
| Public standalone Ollama model | [db0729713/airquality-slm](https://ollama.com/db0729713/airquality-slm) | Model A; original weights with corrected conversion template |
| Processed dataset | [prajwal90/AirQuality-SLM-Processed](https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed) | Published temporal train/validation/test instruction splits |
| Original PyTorch checkpoint | [prajwal90/AirQuality-SLM-Scratch-Qwen](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen) | Custom state dict, config, tokenizer, metrics, training plot |
| Original standalone validation | [safe_inference.py](safe_inference.py) and [RELEASE_NOTES_v1.1.md](RELEASE_NOTES_v1.1.md) | Selected prompt tests: **6/6**; initial validator tests: **9/9** |
| Dataset provenance | [DATASET_CARD.md](DATASET_CARD.md), [data/processed/dataset_info.json](data/processed/dataset_info.json) | Timestamps, missingness, train-only thresholds, split hashes |
| Conversational model (separate) | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Pretrained 1.5B-class instruction-tuned model, Apache-2.0; not the scratch model |

## System overview

```mermaid
flowchart TD
    A["UCI Air Quality source CSV"] --> B["Validate schema, timestamps and numbers"]
    B --> C["Convert -200, blanks and non-finite to missing"]
    C --> D["Sort and deduplicate source hours"]
    D --> E["Chronological 70/15/15 split"]
    E --> F["Fit CO, NO2 and benzene quartiles on TRAIN only"]
    F --> G["Generate classify + explain supervision"]
    G --> H["Hugging Face processed dataset"]
    H --> I["Train 30.3M-parameter MiniQwen from random weights"]
    I --> J["Published custom model.pt + tokenizer + metrics"]
    K["Historical user instruction"] --> L{"Validate task, date, readings and units"}
    L -->|"Invalid / unsupported"| M["REJECTED, no model call"]
    L -->|"Valid"| N["Custom SLM inference"]
    J --> N
    N --> O{"Check output format and missingness"}
    O -->|"Invalid"| P["ABSTAINED"]
    O -->|"Valid"| Q["ACCEPTED: relative historical description"]
```

---

# Phase 1: Data Validation and Preprocessing

This phase is **deterministic data engineering**, not neural language-model inference. Its role is to convert an imperfect sensor record into a traceable training dataset while preventing obvious temporal leakage.

## 1.1 Source data and schema

Source: [UCI Machine Learning Repository, Air Quality, dataset 360](https://archive.ics.uci.edu/dataset/360/air+quality), DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F). The historical collection spans **10 March 2004, 18:00** through **4 April 2005, 14:00**, from a single monitoring context in an Italian city.

The supplied file has **15 required fields**: two time fields and 13 measurement/sensor/environmental fields.

| Source column | Data description | Used for SLM labels? |
|---|---|---|
| `Date`, `Time` | Hourly timestamp; parsed from `DD/MM/YYYY` and `HH.MM.SS` | Time context and split order |
| `CO(GT)` | Reference carbon monoxide, mg/m³ | **Yes, CO** |
| `NO2(GT)` | Reference nitrogen dioxide, µg/m³ | **Yes, NO2** |
| `C6H6(GT)` | Reference benzene, µg/m³ | **Yes, benzene** |
| `NOx(GT)`, `NMHC(GT)` | Other reference pollutant measures | No |
| `PT08.S1(CO)` through `PT08.S5(O3)` | Five gas-sensor channels | No |
| `T`, `RH`, `AH` | Temperature, relative humidity, absolute humidity | No |

The additional fields are kept in the source audit but not included as targets in this three-pollutant instruction task.

## 1.2 Source-data validation

Implemented in [`src/preprocess.py`](src/preprocess.py). The parser:

1. Reads the semicolon-delimited file with UTF-8-BOM handling and decimal-comma normalization.
2. Verifies that all **15 required column names** are present.
3. Parses valid combined dates/times; skips empty tail rows and records invalid timestamps.
4. Converts `-200` (the UCI missing-data sentinel), blank values and non-finite/non-numeric measurements to missing values rather than treating them as zero.
5. Deduplicates by timestamp, retaining the first observation; sorts all valid observations chronologically.
6. Counts missing values and records SHA-256 hashes for reproducibility.

**Recorded source audit:**

| Check | Result |
|---|---:|
| Parsed, distinct timestamped measurements | **9,357** |
| File rows, including header and blank tail | **9,472** |
| Trailing blank CSV rows | **114** |
| Invalid timestamp rows | **0** |
| Duplicate timestamps removed | **0** |
| UCI metadata/listing context | **9,358** instances/rows reported; supplied CSV audit gives **9,357 observations** |

The 9,357 vs 9,358 distinction is retained rather than quietly changing the source record count. The source CSV SHA-256 recorded in `dataset_info.json` is:

```text
13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996
```

### Missing-value audit

| Source field | Missing after `-200` conversion | Interpretation |
|---|---:|---|
| `NMHC(GT)` | 8,443 | Extensive missingness |
| `CO(GT)` | 1,683 | Retained as unavailable when missing |
| `NO2(GT)` | 1,642 | Retained as unavailable when missing |
| `NOx(GT)` | 1,639 | Other reference gas |
| `C6H6(GT)` (benzene) | 366 | Retained as unavailable when missing |
| `PT08.S1`, `PT08.S2`, `PT08.S3`, `PT08.S4`, `PT08.S5`, `T`, `RH`, `AH` | 366 each | Sensor/environmental missingness |

**No fabricated imputation:** The chosen training targets use an explicit **unavailable** category when CO, NO2 or benzene is missing. This avoids presenting filled-in readings as direct observations.

## 1.3 Chronological train/validation/test split

The crucial rule is **split the unique source hours first, then create examples**. Each original hour produces two prompts; both prompts must belong to the same split.

| Split | Source hours | Instruction/answer examples | Historical period |
|---|---:|---:|---|
| Training | **6,549** | **13,098** | 2004-03-10 18:00 to 2004-12-08 14:00 |
| Validation | **1,404** | **2,808** | 2004-12-08 15:00 to 2005-02-05 02:00 |
| Held-out test | **1,404** | **2,808** | 2005-02-05 03:00 to 2005-04-04 14:00 |
| **Total** | **9,357** | **18,714** | Entire validated period |

This is an ordered approximately **70/15/15** split, not a random split. The timestamps in training end before validation begins, and validation ends before testing begins. `record_id` sets are disjoint across all three splits.

**What this does and does not prove:** it prevents *source-hour overlap* and *future-hour quartile fitting* across splits. It does not eliminate every possible form of temporal correlation, covariate shift, sensor calibration drift or generalization error.

## 1.4 Relative labels from training-only quartiles

For each pollutant, the pipeline calculates the **25th (Q1)** and **75th (Q3)** percentiles using **only non-missing training-period readings**, then freezes them for validation, testing and the generated instruction targets.

| Pollutant | Q1 (25th percentile) | Q3 (75th percentile) | Non-missing training readings |
|---|---:|---:|---:|
| CO | **1.1 mg/m³** | **2.8 mg/m³** | 5,066 |
| NO2 | **69 µg/m³** | **123 µg/m³** | 5,096 |
| Benzene | **5.1 µg/m³** | **14.6 µg/m³** | 6,401 |

Labeling logic used in `src/preprocess.py`:

- **unavailable:** missing sensor value;
- **low:** measured value strictly below Q1;
- **elevated:** measured value strictly above Q3;
- **typical:** value between Q1 and Q3, **including both boundaries**.

These categories describe this historical dataset, **not** a calibrated official AQI, air-quality standard or exposure-safety threshold.

## 1.5 Instruction dataset generation

For every validated hour, the pipeline produces **two supervised examples**:

**Classification instruction**

```text
Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3.
```

Target:

```text
CO: elevated; NO2: elevated; benzene: elevated.
```

**One-sentence explanation instruction**

```text
Explain these historical air-quality measurements in one factual sentence. Use relative historical quartiles only; do not infer official AQI, health safety or a pollution source. Hour: 2005-03-06T09:00:00. CO: 0.70 mg/m3; NO2: 79.00 ug/m3; benzene: 1.50 ug/m3.
```

Target:

```text
Relative to historical training data, CO is low, NO2 is typical, and benzene is low.
```

Each processed CSV stores: `text`, `label`, `task`, `record_id`, `timestamp`, `input_text` and `target_text`. The duplicated text/label columns make the data consumable by common text-to-text loading tools while retaining provenance.

**Important distinction:** quartile rules generate the *supervision labels*. They are not themselves the trained SLM, and a rule-based answer should not be misreported as a neural-model prediction.

## 1.6 Dataset integrity and reproducibility

Evidence is available in [`tests/test_pipeline.py`](tests/test_pipeline.py) and the [machine-readable metadata](data/processed/dataset_info.json).

Verified/recorded invariants cover:

- Exactly **6,549 / 1,404 / 1,404** source hours;
- Exactly **two examples per source hour**;
- No overlapping `record_id` values across splits;
- Strict historical train < validation < test boundaries;
- Source and generated-split SHA-256 values.

| Processed split | SHA-256 |
|---|---|
| `train.csv` | `d28759a9816ad2d50226fbd2df0eea69e2cbdf64ed04d7127f9e3d9316a587a4` |
| `validation.csv` | `d684890c81a02fd6f4aa31c2576ce804a34e74764fedc8325bfe06f50b2313d0` |
| `test.csv` | `470f28ced793f4504bfd3050a9873a5dc28875408de1d7a853f84336649617f9` |

To reproduce this **data-only phase** (assuming the official CSV is placed in `data/raw/`):

```powershell
python src/preprocess.py --csv data/raw/AirQualityUCI.csv --out data/processed
python -m unittest discover -s tests -p test_pipeline.py -v
```

See [DATASET_CARD.md](DATASET_CARD.md) for provenance and restrictions. The published data files are at [Hugging Face](https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed).

---

# Phase 2: From-Scratch Small Language Model and Safe Inference

This phase is **learned text generation plus application-level verification**, building on Phase 1's supervised instruction/answer pairs.

## 2.1 Actual model, not a fine-tuned Qwen checkpoint

The completed experiment uses a **custom decoder-only MiniQwen-style Transformer**, coded in [`airquality_slm_model.py`](airquality_slm_model.py) and **trained from randomly initialized model weights**. Its layers borrow general Qwen-family architectural ideas, but **no pretrained Qwen weights were loaded**. The published **`Qwen/Qwen3-8B` tokenizer** supplies tokenization and a vocabulary, not pretrained neural reasoning or knowledge.

| Model property | Published configuration |
|---|---:|
| Trainable/unique model parameters | **30,301,632** |
| Vocabulary size | **151,669** |
| Transformer decoder layers | **3** |
| Hidden dimension (`d_model`) | **192** |
| Attention query heads | **6** |
| Attention key/value heads | **2** |
| Feed-forward dimension (`d_ff`) | **512** |
| Maximum context length (`max_seq_len`) | **160 tokens** |
| Architectural components | RoPE, grouped-query attention, Q/K RMSNorm, residual blocks, SwiGLU, causal attention |
| Output projection | Weight-tied to input-token embedding |
| Checkpoint | Custom PyTorch state dict `model.pt` |

The architecture is defined directly in PyTorch, not generated by calling an off-the-shelf `AutoModelForCausalLM.from_pretrained` model. The model generates output autoregressively with greedy next-token selection.

### Model structure

```mermaid
flowchart LR
    A["Training instruction + historical readings"] --> B["Qwen3 tokenizer"]
    B --> C["Trainable token embedding"]
    C --> D["3 decoder blocks"]
    D --> E["Final RMSNorm"]
    E --> F["Tied vocabulary projection"]
    F --> G["Autoregressive classification / explanation"]
    D1["RoPE + grouped-query causal attention"] -.-> D
    D2["Q/K RMSNorm + SwiGLU feed-forward"] -.-> D
```

## 2.2 Training setup and checkpoints

| Experiment field | Recorded outcome |
|---|---|
| Training origin | From scratch, random weights |
| Optimizer updates | **4,000** |
| Loaded training examples | **13,098** |
| Loaded validation examples | **2,808** |
| Loaded held-out test examples | **2,808** |
| Per-step batch size | **2** |
| Gradient-accumulation steps | **2** |
| Maximum sequence length | **160** |
| Training machine | NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), local Windows/Python environment |
| Checkpoint artifacts | `model.pt`, `model_config.json`, custom Python model implementation, tokenizer files |
| Publication | [AirQuality-SLM-Scratch-Qwen on Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen) |

The completed training run was performed in the project's notebook workflow. The repo's older `src/train.py` is a **separate, unexecuted FLAN-T5 experimental path** and **must not be used as proof or a reproduction script for this trained MiniQwen checkpoint**.

## 2.3 Evaluation results: measured vs interpreted

Values below come from [`metrics.json`](metrics.json), which mirrors the published checkpoint's metrics.

| Metric | Recorded value |
|---|---:|
| Best sampled validation loss | **0.002819679** |
| Full validation loss | **0.009190715** |
| Held-out test loss | **0.008156747** |
| Optimizer steps | **4,000** |
| Pretrained model weights used | **False** |

These are **cross-entropy losses**, **not** 99% accuracy, 99.9% precision, official AQI agreement or an externally validated pollution-classification score. The low losses should be understood in the context of highly templated, automatically generated supervision. Wider accuracy evaluation, calibration and out-of-distribution tests remain open.

Training graphs: [`training_curves.png` on Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen/blob/main/training_curves.png).

## 2.4 Independent checkpoint verification

The public checkpoint was independently downloaded, loaded on CPU through the repository's custom class, and checked with a strict PyTorch state-dictionary load (`weights_only=True`, `strict=True`). This verifies that the published weights and model code are structurally compatible. It does **not** prove the model is accurate on every possible input.

The portable CLI loads the checkpoint, configuration, tokenizer, and model source lazily through Hugging Face Hub. First use requires an internet connection or a previously downloaded local model folder. There is **no need to retrain**.

## 2.5 Robustness tests: raw model vs safe inference

Three positive and three negative/edge prompts were selected. In the **original model-only experiment**, valid prompts passed but two unsupported/malformed prompts failed. A separate deterministic validation wrapper was then applied **without changing any weights**.

| Case | Question tested | Original SLM alone | SLM with safe-inference layer |
|---|---|---|---|
| P1 | Complete high relative readings | PASS | **PASS** |
| P2 | A missing benzene reading | PASS | **PASS** |
| P3 | Constrained historical explanation | PASS | **PASS** |
| N1 | All three readings missing | PASS | **PASS** |
| N2 | Malformed numbers/units | FAIL: generated labels | **PASS: rejected before model call** |
| N3 | Future official AQI and exercise safety request | FAIL: repeated text | **PASS: rejected before model call** |
| **Selected-prompt total** | | **4/6** | **6/6** |

Additionally, the safe-inference Python module passed **9/9 offline validator unit tests**. On **15 November 2026**, the standalone CLI passed the six selected live-model checks **in a fresh Python process with exit code 0**.

**Do not interpret 6/6 as a population accuracy or safety benchmark.** It is evidence for these six chosen cases only. The two problematic prompts were safely rejected by application logic rather than magically corrected through neural retraining.

## 2.6 The input and output safety layer

Implemented in [`safe_inference.py`](safe_inference.py).

**Before calling the SLM:**

1. Accept only two supported templates: historical `Classify` and `Explain`.
2. Require an ISO timestamp within the dataset's historical range.
3. Require all three pollutant fields, expected units (CO: `mg/m3`; NO2/benzene: `ug/m3`), and finite nonnegative numeric readings.
4. Map `missing` and `-200` markers to **missing**, never to a real concentration.
5. **REJECT** malformed or out-of-scope requests before spending inference compute.

**After generation:**

1. Match the exact three-pollutant output structure or constrained one-sentence explanation.
2. Confirm that present readings are not labeled `unavailable` and missing readings are not assigned a numerical category.
3. Return **ACCEPTED** when the schema checks pass; **ABSTAINED** if model output is malformed/inconsistent; or **ERROR** if inference itself fails.

The validator checks syntax, task scope and missingness consistency. **It does not independently verify that every numeric category selected by the model is correct**, guarantee physical sensor plausibility or provide calibrated confidence.

### Run current verified CLI

Using a fresh Python environment and the published checkpoint:

```powershell
git clone https://github.com/DSGAMING98/AirQuality-SLM-DT.git
cd AirQuality-SLM-DT
python -m pip install torch "transformers>=4.51,<5" "huggingface_hub>=0.30"
python safe_inference.py --self-test --device cpu
python -m unittest discover -s tests -p test_safe_inference.py -v
```

One supported example:

```powershell
python safe_inference.py --device cpu --prompt "Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3."
```

Expected for that **selected** input: `CO: elevated; NO2: elevated; benzene: elevated.`

Use `--model-dir PATH` for a trusted local model folder instead of downloading, or `--revision COMMIT_SHA` to pin a specific Hugging Face revision. A public repository does not need a Hugging Face login just to download files.

**Security note:** the loader imports the model repository's Python implementation, so only run code from sources/revisions you trust.

## 2.7 Application and repository status

The existing `app.py` on GitHub `main` is **legacy FLAN-T5-era Streamlit code**. It does **not** load the published MiniQwen checkpoint and is **not** the verified conversational interface.

A separate local Streamlit application (`conversational_streamlit.py`) and command-line application (`conversational_airquality.py`) were developed and reported as validated in the `codex/ollama-demo` workspace, alongside the launchers listed above. The local Streamlit health check was reported as `ok`; that should not be confused with a remotely deployed public web application or with these files already being present on `main`.

The original `safe_inference.py` CLI on `main` remains the directly available checkpoint test path.

---

# Phase 3: Ollama Conversion, Debugging, and Public Release

## 3.1 Why a conversion layer was required

The original `model.pt` is a custom PyTorch state dictionary, not a native Ollama model artifact. The deployment process used an experimental Qwen3-compatible export path and a GGUF conversion to make the trained model accessible to Ollama. The public model is a packaging/deployment of the **original trained SLM**, not a retraining or silent substitution with pretrained Qwen weights.

The first Ollama import launched but produced malformed repetitions such as `low low` or `is is`, including on a supported classification example that previously worked in PyTorch. Import success alone was therefore **not** accepted as evidence of parity.

## 3.2 Root cause: embedded prompt-template mismatch

The debugging report identified the embedded Qwen3 ChatML-style template as the source of a **four-token input mismatch**:

| Path | Prompt tokens for the diagnostic case | Observation |
|---|---:|---|
| Original training/inference framing | **88** | Correct targeted inference |
| Initial Ollama embedded chat template | **92** | Repetitive or incorrect generation |
| Corrected Ollama GGUF/template | Matched expected formatting | Selected regression suite passed |

The template metadata was corrected while retaining the **same 35 tensor arrays**. This is a prompt-tokenization/serialization integration issue, **not** evidence of defective model weights. The original checkpoint and original GGUF were preserved.

## 3.3 Published artifact and reported verification

**Public model:** https://ollama.com/db0729713/airquality-slm

The publication/verification report supplied for the local deployment records:

| Item | Reported result |
|---|---|
| Ollama model namespace | `db0729713/airquality-slm` |
| Model status | Publicly published |
| Artifact download | Approximately **66 MB**, clean-pulled into an empty Ollama store |
| Public manifest identifier (reported prefix) | `2ed62f4acb9b` |
| License in published model | **Apache-2.0** |
| Corrected prompt-template metadata | Preserved |
| Clean-pulled Model A regression suite | **18/18** selected checks |
| Original checkpoint and GGUF | Not overwritten |

**Evidence distinction:** these are **results reported by the local integration/publication workflow**. The published Hugging Face checkpoint, dataset metadata and earlier unit tests are separately documented above. A selected **18/18** functional suite is *not* equivalent to full held-out classification accuracy, model robustness certification, or validated environmental-safety performance.

The public Ollama registry distributes the standalone model artifact; it does **not** package the Python conversational router, application UI, or external Model B.

# Phase 4: Genuine Multi-Turn Conversational System

## 4.1 Two neural models, distinct responsibilities

| Component | Original classification Model A | Conversation Model B |
|---|---|---|
| Model | `db0729713/airquality-slm` | `qwen2.5:1.5b` via Ollama |
| Origin | Custom **trained from scratch** | **Pretrained + instruction-tuned** Qwen2.5-1.5B-Instruct |
| Approximate scale | **30.3M** parameters | **1.54B** parameters (official model card) |
| Responsibility | CO, NO2, benzene historical-relative labels / constrained explanation | Greetings, natural dialogue, explanations, follow-ups and short-term conversational context |
| License | Apache-2.0 (reported published model artifact) | Apache-2.0 (upstream Qwen model card) |
| Provenance | [Custom model checkpoint](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen) | [Official Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |

Model B is an **existing independently pretrained model**. The present implementation does **not** claim to have trained it from scratch, nor to have fine-tuned it on the AirQuality dataset. The original technical contribution remains Model A, the dataset pipeline, the Ollama conversion/debugging work, and the integration and validation architecture.

## 4.2 Inference and routing architecture

```mermaid
flowchart TD
    U["User message and recent conversation"] --> R{"Input validation and task routing"}
    R -->|"Supported historical measurements"| V["Validate timestamp, fields, units and missingness"]
    V -->|"Valid"| A["Model A: scratch-trained airquality-slm on Ollama"]
    V -->|"Invalid"| X["Clarify / reject unsupported classification"]
    A --> C["Validate structural and missingness consistency"]
    C --> E["Return attributed classification"]
    R -->|"Conversation or explanation"| B["Model B: qwen2.5:1.5b on Ollama"]
    E --> B
    B --> H["Context-aware, attributed natural-language reply"]
    X --> H
```

This diagram depicts the **intended functional separation**. The router and validator are deterministic application components; conversational text originates from Model B, and in-scope classifications originate from Model A. A hardcoded FAQ reply must **not** be described as neural conversation.

Example interactions:

```text
User: What are CO, NO2 and benzene?
Conversational Model B: Explains the pollutants in ordinary language.

User: Explain that as if I'm 10.
Conversational Model B: Uses preceding context to simplify the explanation.

User: Classify 2005-03-08T08:00:00, CO 3.40 mg/m3,
      NO2 182 ug/m3, benzene 18.30 ug/m3.
Classification Model A: CO: elevated; NO2: elevated; benzene: elevated.

User: Does that mean it's officially unsafe outdoors?
Guarded conversation: Explains that historical quartiles are not official safety thresholds.
```

The paraphrased conversational responses above are *illustrative interaction types*, not a verbatim test transcript.

## 4.3 Local application, speed and test evidence

The reported final local implementation includes the Python two-model router, multi-turn history management, a command-line interface and a Streamlit interface. The supplied local verification summary reports:

| Verification scope | Reported result | What this means |
|---|---:|---|
| Final application unit suite | **22/22** | Selected integration/unit tests |
| Model A Ollama regression checks | **18/18** | Selected classification, repeated inference and safety-path checks |
| Conversational checks | **13/13** | Selected multi-turn dialogue, routing and follow-up scenarios |
| Streamlit health check | `ok` | Local app health endpoint responded |
| Model B throughput | **~140 generated tokens/sec** | Reported local measurement, not universal benchmark |

The three test suites may exercise overlapping behavior and should **not** be treated as 53 independent evaluations of model accuracy. Throughput depends on device, generation length, quantization, concurrent load and Ollama configuration.

Model A is optimized for narrow historical classification prompts and has a **160-token maximum context**; it should not be mistaken for the model responsible for multi-turn natural-language conversation. The app can use Model B's separate conversational context policy, subject to its configured runtime.

## 4.4 Safety, limitations and attribution

- The **`-200`** source sentinel always denotes unavailable measurement data, never a negative pollution concentration.
- The classification categories come from **train-only quartiles** and do not represent an official AQI or guaranteed health safety.
- The historical dataset covers **2004–2005**; the system cannot infer real-time air quality, current city conditions or future forecasts.
- Input parsers and output validators can reject malformed, out-of-range and unsupported requests, but they **do not guarantee** that every neural category is numerically correct.
- Model B can generate fluent but incorrect claims. Scientific claims require a defensible reference; conversation quality is not the same as environmental accuracy.
- The user-facing application should identify when content comes from Model A, Model B, or deterministic application safeguards.

## 4.5 Reproducing the public portion vs the local development portion

**Available immediately from the public registry:**

```powershell
ollama run db0729713/airquality-slm
```

**In the original local project folder after installation of both models:**

```powershell
ollama pull db0729713/airquality-slm
ollama pull qwen2.5:1.5b
.\launch_conversational.ps1
# or
.\launch_conversational_ui.ps1
```

**Important:** the launcher commands require the additional *local, unmerged* scripts. A fresh clone of this GitHub `main` branch currently includes the older research pipeline and safe-inference CLI but **not** the complete two-model chatbot source. Full third-party reproducibility requires publishing that application code, version-pinned dependencies and verification reports.


---

# Design Thinking and Engineering Decisions

| Stage | Project-specific action and evidence |
|---|---|
| **Empathize** | Need an understandable explanation of historical CO/NO2/benzene readings and transparent handling of absent sensor measurements |
| **Define** | Translate numerical measurements into short factual, auditable descriptions without pretending they are official AQI or health assessments |
| **Ideate** | Consider rule-only classification vs learned short text generation; use quartile targets for supervision, then implement a compact custom Transformer |
| **Prototype** | Build timestamp-cleaning/splitting code, publish instruction examples, train MiniQwen from scratch, implement CLI inference |
| **Test and iterate** | Audit source/splits, compute held-out losses, record selected prompt outcomes, correct the Ollama template mismatch, and verify two-model routing, dialogue and safety paths |

This framing distinguishes **the deterministic data and label creation**, **the learned SLM**, and **the validation guardrails** as separate pieces of the product.

# Repository Layout

```text
AirQuality-SLM-DT/
├── README.md                      # Four-phase research and deployment documentation
├── src/
│   ├── preprocess.py              # Phase 1: validate/clean/split/label
│   ├── train.py                   # Historical FLAN-T5 prototype, NOT MiniQwen training
│   └── evaluate.py                # Historical FLAN-T5 evaluator
├── data/
│   ├── raw/                       # Original UCI source
│   └── processed/                 # Train/validation/test CSV + metadata
├── train.csv                      # Published-split convenience copy
├── validation.csv
├── test.csv
├── dataset_info.json              # Phase 1 metadata convenience copy
├── DATASET_CARD.md                # Phase 1 provenance and limitations
├── tests/
│   ├── test_pipeline.py           # Phase 1 split invariants
│   └── test_safe_inference.py     # Phase 2 validator tests
├── airquality_slm_model.py        # Custom MiniQwen-style PyTorch architecture
├── model_config.json              # Trained model configuration
├── metrics.json                   # Published loss measurements
├── inference_example.py           # Direct custom-checkpoint inference example
├── safe_inference.py              # Recommended validated v1.1 CLI
├── RELEASE_NOTES_v1.1.md          # v1.1 release evidence
├── PUBLISHING.md                  # Initial publication notes (some steps historical)
├── requirements.txt               # Mixed original project dependencies
└── app.py                         # LEGACY FLAN-T5 Streamlit app, not v1.1-ready
```

The large trained `model.pt` and tokenizer artifacts are hosted in the **Hugging Face model repository**, not in the GitHub source tree. The public Ollama registry also hosts Model A as `db0729713/airquality-slm`.

**Local files not yet on public `main` (as reported by the development workflow):** `conversational_airquality.py`, `conversational_streamlit.py`, `launch_conversational.ps1`, `launch_conversational_ui.ps1`, `CONVERSATIONAL_SLM_ARCHITECTURE.md`, `CONVERSATIONAL_VERIFICATION_REPORT.md`, and `OLLAMA_PUBLICATION_PREPARATION.md`. Their names are documented here for the local workspace, not falsely linked as committed files.

## Current vs historical repository documentation

Some pre-v1.1 files such as `TRAINING_REPORT.md`, `EVALUATION_REPORT.md`, `MODEL_CARD.md`, `PROMPT_TEST_RESULTS.md`, `PROJECT_REPORT.md` and `REPRODUCIBILITY.md` were written for the **earlier unexecuted FLAN-T5 proposal**. Their descriptions of model training being blocked, no weights being published, or FLAN-T5 being the final architecture **are historical and do not describe the published MiniQwen run**. They have been left intact as development records; this README, `metrics.json`, `safe_inference.py`, and `RELEASE_NOTES_v1.1.md` describe the published from-scratch experiment. The Ollama/publication/conversation sections above describe subsequent work and explicitly distinguish reported local results from files committed to `main`.

## Limitations and future work

- **Historical, single-context dataset (2004–2005):** cannot stand in for current air quality in Bengaluru or any other location.
- **Training-relative quartiles:** not regulatory limits, clinical thresholds or official AQI categories.
- **Automatically generated labels:** good for controlled educational language supervision, but not human-annotated pollution expertise.
- **Small, from-scratch model:** susceptible to poor generalization and incorrect outputs on unseen numeric combinations, even when the format looks right.
- **Guardrails are partial:** they enforce supported input shape, dates, units, missingness and output syntax, not ground-truth numerical correctness.
- **Limited evaluation:** six chosen functional/robustness cases and loss values do not substitute for full held-out classification accuracy, multiple seeds, calibration or adversarial security testing.
- **Repository synchronization gap:** The Streamlit app on `main` is legacy; a distinct local conversational UI passed a reported health check, but its code has not been committed to this public branch. No public hosting or end-to-end independent UI reproduction is established.
- **Training provenance gap:** the actual from-scratch training notebook is separate from the old GitHub FLAN-T5 training script; a clean end-to-end training reproduction and archived environment should be added.

**Future work:** commit the tested conversational application, pinned dependencies and verification logs to a reviewable GitHub branch; publish the trained-model-specific reproduction notebook/scripts; evaluate exact-match and per-label accuracy on the full held-out set; run multiple seeds, OOD stress tests and numeric consistency checks; independently benchmark Ollama inference and GPU utilization; and seek domain review before any higher-stakes use.

## Presentation release status and Ollama import (18 November 2026)

**Assessment dates:** 19–20 November 2026. The from-scratch model and published dataset are available on Hugging Face; the tested **PyTorch safe-inference CLI** is the verified demonstration path. Direct Ollama import of the custom checkpoint has **not yet been verified**.

**Experimental Ollama route (not a completed demo):**
1. Download the original published \`model.pt\`, model code and tokenizer.
2. Map native MiniQwen weights to Qwen3-compatible Hugging Face tensors, including the interleaved-to-split RoPE reordering.
3. **Stop unless** the converted model reproduces native PyTorch next-token logits on selected historical instructions.
4. Run \`llama.cpp\`'s \`convert_hf_to_gguf.py\`, then create an Ollama \`Modelfile\` using that actual GGUF.
5. Run \`ollama create airquality-slm -f Modelfile\`, then \`ollama run airquality-slm\`, checking its outputs against the original model. This **must be tested on the user's machine**; it cannot be claimed completed from the current repository evidence.

See [Ollama compatibility and chat demo guide](docs/OLLAMA_DEMO_GUIDE.md). A draft [IEEE conference-style research article source](research/AirQuality_SLM_IEEE_Research_Article.tex) is also included for the university assessment. Neither these experimental conversion instructions nor the draft manuscript imply IEEE publication or a verified Ollama import.

**Outstanding code provenance:** \`src/train.py\` and \`src/evaluate.py\` are older FLAN-T5 examples. The completed MiniQwen training notebook should be archived separately before claiming the entire trained model can be reconstructed from GitHub alone.

## References

1. UCI Machine Learning Repository, [Air Quality dataset (ID 360)](https://archive.ics.uci.edu/dataset/360/air+quality), DOI: [10.24432/C59K5F](https://doi.org/10.24432/C59K5F).
2. De Vito et al., *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*, Sensors and Actuators B, 2008.
3. [PyTorch](https://pytorch.org/) documentation, custom model definition, state dictionaries and causal attention.
4. [Hugging Face Hub](https://huggingface.co/docs/hub/index) documentation, model repositories and tokenizer assets.
5. [Qwen/Qwen3-8B tokenizer repository](https://huggingface.co/Qwen/Qwen3-8B) (tokenizer used; **pretrained Model A weights not used**).
6. [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (separate instruction-tuned conversational Model B).
7. [Ollama library](https://ollama.com/library) and [published AirQuality model](https://ollama.com/db0729713/airquality-slm).
8. [Hugging Face processed AirQuality dataset](https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed) and [original scratch checkpoint](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen).

---

**Summary:** **Phase 1** validates 9,357 historical observations and generates 18,714 chronologically separated training/evaluation examples. **Phase 2** trains a 30.3M-parameter Transformer from scratch with a validated safe-inference path. **Phase 3** fixes the Ollama prompt-template mismatch and publishes the original model under `db0729713/airquality-slm`. **Phase 4** uses a **separate pretrained Qwen2.5-1.5B-Instruct model** for genuine multi-turn conversation, plus deterministic routing and guardrails, currently available only in the local unmerged development workspace. Neither historical-relative classification nor conversational fluency should be confused with official AQI or verified health-safety advice.
