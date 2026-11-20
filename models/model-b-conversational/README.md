---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- air-quality
- conversational
- ollama
- qwen2.5
---

# Model B — AirQuality conversational configuration

Model B is the conversational component of the AirQuality two-model application. It uses
**Qwen2.5 1.5B Instruct**, a separate pretrained model created by Alibaba Cloud and distributed
under Apache-2.0.

This repository/folder contains the AirQuality system prompt, Ollama configuration, attribution
and integration documentation. **It does not claim that Qwen2.5 was trained by this project and
does not duplicate or rebrand the upstream weights.**

## Responsibilities

- Natural greetings and general conversation
- Air-quality and project explanations
- Bounded multi-turn context and follow-up questions
- Safe explanation of verified Model A results

Model B must not calculate or guess Model A classifications. The complete Python application
routes valid historical measurement requests to Model A.

## Run

Direct conversational tag:

```powershell
ollama run db0729713/airquality-slm
```

Complete two-model router from the project root:

```powershell
.\launch_conversational.ps1
```

## Attribution

- Upstream model: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- Published Model B mirror: <https://huggingface.co/prajwal90/AirQuality-SLM-Model-B-Qwen2.5-1.5B-Instruct>
- Ollama upstream package: <https://ollama.com/library/qwen2.5:1.5b>
- License: Apache-2.0
- AirQuality integration: <https://github.com/DSGAMING98/AirQuality-SLM-DT>

## Limitations

This conversational model has no live monitoring, official AQI feed or forecasting source.
It cannot execute the independent Model A weights from a single Ollama manifest.

