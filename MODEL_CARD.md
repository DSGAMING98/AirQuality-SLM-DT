# Model Card: AirQuality-SLM FLAN-T5

## Status

**Blocked / no fine-tuned checkpoint exists as of 12 November 2026.** Do not cite this repository as a trained model release.

## Intended model

Base checkpoint: `google/flan-t5-small`, an approximately 80M-parameter instruction-tuned T5 encoder-decoder. Intended task: map formatted historical CO, NO₂, and benzene readings to a short relative classification or constrained factual sentence.

## Configuration

Full fine-tuning; 3 epochs; AdamW through Hugging Face Trainer; learning rate `5e-5`; batch size `8`; seed `42`; max source length `256`; max target length `64`; FP16 on CUDA; epoch evaluation/checkpointing; best checkpoint selected on validation loss.

## Current evidence

The training and evaluation scripts are implemented, but the environment lacked Transformers, Datasets, Accelerate, SentencePiece, and the base checkpoint. Network DNS resolution failed, so the experiment did not start. Training loss, validation loss, test metrics, sample generations, and checkpoint size are **Not Run / unavailable**.

## Risks

Even after training, the model can hallucinate, alter units, output unsupported health claims, or follow prompt injection. It must not be used for live monitoring, regulatory compliance, diagnosis, or emergency guidance. Generated supervision carries deterministic-label bias and does not prove scientific validity.
