# AirQuality SLM v1.1: Validated Standalone Inference

Date: 2026-11-15  
Model: https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen  
Dataset: https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed

## What changed

- `safe_inference.py` is a standalone command-line tool for the **trained, custom** MiniQwen architecture, not the repository's original FLAN-T5 proposal.
- Downloads the published model code, config, tokenizer, and PyTorch checkpoint through Hugging Face Hub. Local `--model-dir` also works.
- Validates exact supported historical prompt templates, timestamp range, pollutant names, nonnegative numeric values, and units.
- Converts UCI `-200` missing sentinels to the training-time `missing` text marker before generation.
- Rejects malformed inputs and out-of-scope future AQI/health-safety requests **without invoking the SLM**.
- Validates generated output syntax and consistency with missing measurements, abstaining when the response is invalid.
- Keeps model weights **unchanged**; the validation layer is separate from the 30,301,632-parameter neural network.

## Training and evaluated outcomes

- From-scratch model: 30,301,632 parameters; 4,000 optimizer updates; 13,098 train, 2,808 validation, and 2,808 test instruction examples loaded.
- Full validation cross-entropy: 0.0091907; held-out test cross-entropy: 0.0081567. These losses are **not classification accuracy**.
- Original model-only six-prompt evaluation: **4/6** passed, including **3/3** positive tests, and 1/3 negative robustness tests.
- Jupyter safe-layer evaluation: **6/6** passed on the same six selected prompts, including **3/3** positive regression tests and **3/3** negative tests. Two invalid/out-of-scope prompts were rejected before the model was called.
- These are **six selected prompts**, not a statistically representative robustness or accuracy benchmark. The safe layer does not establish correctness for every valid numeric measurement.

## Reproduce

```bash
python -m pip install torch 'transformers>=4.51,<5' 'huggingface_hub>=0.30'
python safe_inference.py --self-test --device cpu
python -m unittest discover -s tests -p 'test_safe_inference.py' -v
```

Single-prompt example:

```bash
python safe_inference.py --device cpu --prompt "Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3."
```

The first supported inference downloads approximately 130 MB of public model files. No Hugging Face token is required for a public repository. For deterministic reproduction across later revisions, supply `--revision <commit-sha>`.

### Supported scope

Only the exact dataset-style `Classify ...` and `Explain ...` prompts are supported for hourly records between 2004-03-10 18:00 and 2005-04-04 14:00. Units must be `mg/m3` for CO and `ug/m3` for NO2/benzene. Missing readings are `missing` or `-200`; `-200` is never interpreted as actual pollution. Numeric readings must be finite and nonnegative. This gate does not establish physical plausibility, confidence calibration, official AQI, live forecasts, source attribution, or health safety.

### Model loading note

This is an educational custom transformer (`airquality_slm_model.py`, `model.pt`). It cannot be loaded with standard `AutoModelForCausalLM.from_pretrained`; use `safe_inference.py` or the included custom class, config, and state dict. Downloaded Python code is executed, so use only trusted repositories/revisions.

### Publication checklist

- [x] Existing public trained model, tokenizer, and config verified independently in Jupyter
- [x] Six selected Jupyter checks passed with the safe layer
- [x] Run `python safe_inference.py --self-test` in a fresh process against the public checkpoint (6/6 PASS; exit code 0, 2026-11-15)
- [ ] Publish `safe_inference.py` and this document to the model repository
- [ ] Merge the GitHub v1.1 release pull request

No training weights need to be replaced for v1.1.