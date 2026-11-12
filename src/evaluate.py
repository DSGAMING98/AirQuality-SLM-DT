#!/usr/bin/env python3
"""Evaluate exact match and token-level F1 on the held-out test split."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def token_f1(prediction, reference):
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref:
        return float(pred == ref)
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/flan-t5-airquality/best_model")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--out", default="artifacts/evaluation_results.json")
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Evaluation blocked: missing dependency {exc.name}.")
    if not Path(args.model).exists():
        raise SystemExit("Evaluation blocked: trained checkpoint does not exist. Run src/train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    rows = list(csv.DictReader(Path(args.test).open(encoding="utf-8")))
    predictions = []
    for row in rows:
        encoded = tokenizer(row["input_text"], return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=64)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append({"input": row["input_text"], "reference": row["target_text"], "prediction": prediction})
    exact = sum(normalize(x["prediction"]) == normalize(x["reference"]) for x in predictions) / len(predictions)
    f1 = sum(token_f1(x["prediction"], x["reference"]) for x in predictions) / len(predictions)
    result = {"status": "Completed", "examples": len(predictions), "exact_match": exact, "token_f1": f1,
              "sample_predictions": predictions[:20]}
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
