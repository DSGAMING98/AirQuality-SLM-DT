---
language:
- en
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
tags:
- air-quality
- from-scratch
- miniqwen
- ollama
- historical-classification
datasets:
- prajwal90/AirQuality-SLM-Processed
---

# Model A — AirQuality MiniQwen classifier

Model A is the project's original research contribution: a custom MiniQwen-inspired,
decoder-only transformer trained from random initialization for historical air-quality
classification.

## Model details

| Property | Value |
|---|---|
| Parameters | 30,301,632 |
| Layers | 3 |
| Context length | 160 tokens |
| Tokenizer | Qwen3 tokenizer |
| Pretrained model weights | None |
| Training updates | 4,000 |
| Ollama tag | `db0729713/airquality-slm:classifier` |

The model classifies CO, NO2 and benzene readings as `low`, `typical`, `elevated`, or
`unavailable` relative to quartiles derived from its historical training distribution.
The source UCI dataset uses `-200` as a missing-value sentinel.

## Verified Ollama example

```powershell
ollama run db0729713/airquality-slm:classifier "Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3."
```

Expected output:

```text
CO: elevated; NO2: elevated; benzene: elevated.
```

## Files and provenance

- Hugging Face weights: <https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen>
- Dataset: <https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed>
- Ollama: <https://ollama.com/db0729713/airquality-slm:classifier>
- Original dataset: UCI Air Quality, Saverio De Vito, DOI `10.24432/C59K5F`, CC BY 4.0

The root project preserves the PyTorch checkpoint, tokenizer, conversion scripts and verified
F16 GGUF. Large weights are not duplicated in this Git folder.

## Converted artifacts

The complete converted artifacts are hosted on Hugging Face because the weight files are too
large for normal GitHub storage. Checksums and machine-readable metadata are in
[`conversion_manifest.json`](conversion_manifest.json).

| Artifact | Purpose | Download |
|---|---|---|
| `model.safetensors` | Qwen3-compatible Transformers export of Model A | [Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen/resolve/main/model.safetensors) |
| `airquality-miniqwen3-f16-ollama.gguf` | Production F16 GGUF for Ollama | [Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen/resolve/main/airquality-miniqwen3-f16-ollama.gguf) |
| `airquality-miniqwen3-f16.gguf` | Earlier diagnostic GGUF retained for reproducibility | [Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen/resolve/main/airquality-miniqwen3-f16.gguf) |

`CONVERSION_PARITY.json` records the Transformers conversion parity checks. The production
Ollama build is the file whose name ends in `-ollama.gguf`.

## Limitations

This model is a narrow historical classifier, not a conversational assistant. Its labels are
not official AQI values, regulatory limits, forecasts, live conditions or health-safety advice.

