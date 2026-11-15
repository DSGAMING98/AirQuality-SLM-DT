# Publishing the verified AirQuality SLM v1.1 release

**Important:** The v1.1 package is prepared and its nine offline tests passed, and the new CLI **passed its own real-checkpoint test** on 2026-11-15 (6/6, exit code 0). The already-published model weights are unchanged.

## 1. Extract into your local AirQuality-SLM project

Extract the archive into `C:\Users\Prajwal\Desktop\python\AirQuality-SLM`. It should contain `safe_inference.py`, `tests/test_safe_inference.py`, `RELEASE_NOTES_v1.1.md`, `publish_to_hf.py` and this file. If that directory is not a Git checkout, clone the GitHub repository separately for publication.

## 2. Independent clean-process verification

In your existing JupyterLab notebook, run this in a new cell:

```python
import subprocess, sys
r = subprocess.run([sys.executable, "safe_inference.py", "--self-test", "--device", "cpu"],
                   text=True, capture_output=True)
print(r.stdout)
print(r.stderr)
print("Exit code:", r.returncode)
assert r.returncode == 0, "The standalone script did not pass all six tests"
```

The check uses **new process state** and fetches the public HF model as needed. It does **not** require training or a Hugging Face token. Expected result: `Safe-inference release self-test: 6/6 passed`. The previously completed notebook six-prompt test is valuable but is not a substitute for this new CLI test.

Also run the offline test suite:

```python
r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_safe_inference.py", "-v"], text=True, capture_output=True)
print(r.stdout, r.stderr)
print("Exit code:", r.returncode)
assert r.returncode == 0
```

## 3. Publish files to GitHub

The GitHub repository `https://github.com/DSGAMING98/AirQuality-SLM-DT` still contains an earlier FLAN-T5-small *plan*, which is outdated as the main status. Update its README and applicable project/model/testing reports to correctly indicate the trained **from-scratch Qwen-inspired MiniQwen** release. Do not claim FLAN-T5 was trained.

From a local Git clone of the project, copy `safe_inference.py`, `tests/test_safe_inference.py`, `RELEASE_NOTES_v1.1.md`, and this publishing guide into the repository root (test goes in `tests/`). Then:

```powershell
git switch -c airquality-safe-inference-v1.1
git add safe_inference.py tests/test_safe_inference.py RELEASE_NOTES_v1.1.md PUBLISHING.md
git commit -m "Add validated v1.1 standalone AirQuality SLM inference"
git push -u origin airquality-safe-inference-v1.1
```

Open a pull request from `airquality-safe-inference-v1.1` to `main`, review the changed files, and merge once the fresh-process tests pass. If the branch already exists, choose a different name.

Suggested README notice (place at the very top, and revise/retire contradictory historical sections):

> **Current release (2026-11-15): AirQuality SLM v1.1.** A trained-from-scratch, 30,301,632-parameter Qwen-inspired decoder-only model is available at [Hugging Face](https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen). Training finished for 4,000 optimizer updates using 13,098 / 2,808 / 2,808 instruction examples. Full validation loss was 0.00919 and held-out test loss 0.00816. The public checkpoint and tokenizer passed independent CPU loading/inference. A standalone `safe_inference.py` adds input rejection and output schema/missingness checks; the notebook's six selected functional/robustness cases passed 6/6 with the safe layer (versus 4/6 model-only). See `RELEASE_NOTES_v1.1.md`. The older FLAN-T5-small plan further below is historical and was not trained.

## 4. Publish only updated inference code to Hugging Face

The published model already contains the trained weights; **do not replace them**. Sign into Hugging Face with a write token using `hf auth login` (or use the notebook's existing credential). Then from the extracted release folder:

```powershell
python publish_to_hf.py --publish
```

This uploads only `safe_inference.py` and `RELEASE_NOTES_v1.1.md` to `prajwal90/AirQuality-SLM-Scratch-Qwen`. No token should be copied into a public script or chat.

## 5. Confirm remote files

- https://github.com/DSGAMING98/AirQuality-SLM-DT
- https://huggingface.co/prajwal90/AirQuality-SLM-Scratch-Qwen/tree/main
- https://huggingface.co/datasets/prajwal90/AirQuality-SLM-Processed

If the new CLI cannot reproduce 6/6, **don't publish the v1.1 verified claim**. Inspect the exact generated output; the validator cannot guarantee every numeric classification is correct.