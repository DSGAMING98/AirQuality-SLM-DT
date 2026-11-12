# Training Report

## Experiment status

**Blocked / Not Run — 12 November 2026.**

Hardware detection succeeded: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM; Python 3.12.10; PyTorch 2.14.1+cu126. CUDA-capable hardware is available.

Training did not begin because `transformers`, `datasets`, `accelerate`, `sentencepiece`, and `streamlit` were not installed and are not cached. `pip install` retried five times but DNS could not resolve `pypi.org`. The `google/flan-t5-small` base checkpoint was also absent from the local Hugging Face cache.

## Intended configuration

| Setting | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Parameters | approximately 80M |
| Method | full seq2seq fine-tuning |
| Epochs | 3 |
| Learning rate | `5e-5` |
| Batch size | 8 per device |
| Optimizer | AdamW via Trainer default |
| Seed | 42 |
| Precision | FP16 when CUDA is available |
| Max source / target | 256 / 64 tokens |
| Checkpoints | each epoch, retain 2, load best at end |

## Results

Training loss: **Not Run**. Validation loss: **Not Run**. Best checkpoint: **Not Created**. Performance graphs: **Not Available**. Once dependencies are installed, run the exact commands in `REPRODUCIBILITY.md`; `src/train.py` will save `training_results.json` and the actual best model.
