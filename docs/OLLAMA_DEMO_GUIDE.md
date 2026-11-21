# AirQuality SLM | Ollama import experiment (NOT yet verified)

**Presentation: 19-20 Nov 2026.** We must preserve the trained model identity. Do **not** demonstrate a stock `qwen3` download and claim it is the AirQuality SLM. The published custom `model.pt` file cannot be used as an Ollama `FROM` source by itself.

## Route A: test a real weight-preserving conversion

The included exporter maps the custom model's weight names into Qwen3's Hugging Face model structure. Native RoPE uses interleaved even/odd pairs while standard HF Qwen3 uses split-half pairing; the script permutes Q/K projection rows and Q/K RMSNorm weights accordingly. It sets the original 10,000 RoPE theta and checks **three full next-token logit vectors** against the original PyTorch model.

**EXPERIMENTAL**: no trained checkpoint weights were available in the paper-authoring environment to run this exporter end-to-end. GGUF and Ollama are also not yet tested. Do not record "imported successfully" unless you see all of these checks pass.

### 1. Make a clean Windows environment (PowerShell)

```powershell
cd "C:\Users\Prajwal\Desktop\python\AirQuality-SLM"
python -m pip install --upgrade torch "transformers>=4.51,<5" huggingface_hub safetensors
python export_for_ollama_EXPERIMENTAL.py --output-dir "ollama_export\qwen3_hf"
```

The first run downloads the existing public HF weights. If you already have that full model folder, add `--model-dir "C:\path\to\downloaded-model"`. Look for **3 parity passes** and `CONVERSION_PARITY.json`. If the exporter prints `PARITY FAILED`, stop. Do not proceed with GGUF.

### 2. Convert parity-tested model to GGUF

Install Git for Windows, then:

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
python -m pip install -r "llama.cpp\requirements\requirements-convert_hf_to_gguf.txt"
python "llama.cpp\convert_hf_to_gguf.py" "ollama_export\qwen3_hf" --outfile "airquality-miniqwen3-f16.gguf" --outtype f16
```

If llama.cpp changes its requirements path, consult its current conversion instructions. Do not rename another model's GGUF as if it were this checkpoint. A successful command does not by itself prove numerical equivalence.

### 3. Import to Ollama and verify manually

Install Ollama from <https://ollama.com>. Make sure `ollama --version` works. Copy the provided `Modelfile.EXPERIMENTAL` to a file named **`Modelfile`** in the same folder as the GGUF. It deliberately applies the `Instruction:\n...\nAnswer:\n` prompt framing used by training.

```powershell
ollama create airquality-slm -f Modelfile
ollama run airquality-slm
```

For `ollama run`, paste **only** the question below, with no `Instruction:` prefix (the Modelfile supplies that prefix):

```text
Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3.
```

Expected **reference output from the previously verified PyTorch SLM**:

```text
CO: elevated; NO2: elevated; benzene: elevated.
```

Also test two more positive prompts and one all-missing prompt from `safe_inference.py`. If any outputs differ, record the discrepancy; do not claim equality or full support. Preserve a screenshot of actual terminal output for the presentation.

**Important**: This Ollama chat is **raw model inference**. It does not include the Python safety wrapper, so unsupported forecasts and malformed inputs may give poor or unsafe responses. Only demonstrate historical, supported prompts. The independent PyTorch CLI remains the validated release.

## Route B: reliable backup if GGUF conversion fails

Run your actual trained model in the terminal through the previously verified safety wrapper:

```powershell
python safe_inference.py --self-test --device cpu
```

For a single historical prompt, call `python safe_inference.py --device cpu --prompt "<exact supported historical prompt>"`. Explain to the examiner that **Ollama's import step has not been completed**, rather than misrepresenting another model. A backup SLM demo does not satisfy the Ollama-specific checklist item, but it is an honest demonstration of trained and tested work.

## What counts as complete

- [ ] Exporter loaded the published weights strictly.
- [ ] All three HF vs original PyTorch next-token parity checks passed.
- [ ] GGUF conversion completed.
- [ ] `ollama create airquality-slm -f Modelfile` completed.
- [ ] `ollama run airquality-slm` reproduced at least the four selected supported reference cases.
- [ ] `ollama show airquality-slm` and screenshots captured.
- [ ] Actual demo results documented in GitHub README **after** verification.

## References

- [Ollama model import documentation](https://github.com/ollama/ollama/blob/main/docs/import.mdx)
- [Qwen3 + llama.cpp conversion](https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md)
- [Source model repository](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen)