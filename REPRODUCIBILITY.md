# Reproducibility Guide (Windows PowerShell)

## Environment

```powershell
git clone https://github.com/DSGAMING98/AirQuality-SLM-DT.git
cd AirQuality-SLM-DT
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the official `AirQualityUCI.csv` under `data\raw\`. Verify SHA-256:

```powershell
Get-FileHash data\raw\AirQualityUCI.csv -Algorithm SHA256
```

Expected source hash: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.

## Data pipeline and tests

```powershell
python src\preprocess.py --csv data\raw\AirQualityUCI.csv --out data\processed
python -m unittest discover -s tests -v
```

Expected examples: train 13,098; validation 2,808; test 2,808. Inspect `data\processed\dataset_info.json` for boundaries and hashes.

## Train, validate, test

```powershell
python src\train.py `
  --data-dir data\processed `
  --output-dir artifacts\flan-t5-airquality `
  --epochs 3 --batch-size 8 --learning-rate 5e-5 --seed 42

python src\evaluate.py `
  --model artifacts\flan-t5-airquality\best_model `
  --test data\processed\test.csv `
  --out artifacts\evaluation_results.json
```

Do not add losses/scores to documentation until the JSON result files exist and have been reviewed.

## Application

```powershell
streamlit run app.py
```

The UI must point at `artifacts\flan-t5-airquality\best_model`. A preprocessing preview is not model inference.

## Hugging Face

After creating authorized repositories, upload `train.csv`, `validation.csv`, `test.csv`, and `DATASET_CARD.md` to the dataset repository. Upload only the trained best-model directory plus `MODEL_CARD.md` to the model repository. Verify remotely:

```python
from datasets import load_dataset
data = load_dataset("YOUR_USERNAME/AirQuality-SLM-Processed")
print(data)

from transformers import pipeline
generator = pipeline("text2text-generation", model="YOUR_USERNAME/AirQuality-SLM-FLAN-T5")
print(generator("Classify CO: 2.0 mg/m3; NO2: 100 ug/m3; benzene: 10 ug/m3."))
```

Record only tested URLs and outputs in `SUBMISSION.md`.
