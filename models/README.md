# AirQuality two-model system

This directory separates the two independently attributed models used by the project.

| Folder | Model | Purpose | Weights |
|---|---|---|---|
| `model-a-airquality-classifier` | Custom 30.3M MiniQwen-inspired model | Historical CO, NO2 and benzene classification | Trained from scratch by this project |
| `model-b-conversational` | Qwen2.5 1.5B Instruct | Conversation, explanations and multi-turn context | Upstream Alibaba Cloud model; not trained by this project |

The complete application uses `../launch_conversational.ps1` to route validated measurement
requests to Model A and all other conversation to Model B. A single Ollama manifest cannot run
both independent weight sets.

To publish both organized folders to their Hugging Face model repositories after authenticating
the official CLI, run `./models/publish_to_huggingface.ps1` from the project root.

