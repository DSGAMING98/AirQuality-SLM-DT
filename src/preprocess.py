#!/usr/bin/env python3
"""Convert UCI Air Quality measurements into leakage-safe text-to-text supervision."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

COLUMNS = [
    "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
    "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
MEASURES = {
    "CO": ("CO(GT)", "mg/m3"),
    "NO2": ("NO2(GT)", "ug/m3"),
    "benzene": ("C6H6(GT)", "ug/m3"),
}
FIELDNAMES = ["text", "label", "task", "record_id", "timestamp", "input_text", "target_text"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_number(value: str | None) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return None
        number = float(text)
        return None if not math.isfinite(number) or number == -200 else number
    except ValueError:
        return None


def load_rows(path: Path):
    source = path.read_bytes()
    decoded = source.decode("utf-8-sig", errors="replace")
    physical_rows = list(csv.reader(io.StringIO(decoded), delimiter=";"))
    table = csv.DictReader(io.StringIO(decoded), delimiter=";")
    missing = set(COLUMNS) - set(table.fieldnames or [])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = {}
    parsed_timestamp_rows = 0
    invalid_timestamp_rows = 0
    missing_counts = Counter()
    for raw in table:
        if not (raw.get("Date") or "").strip() and not (raw.get("Time") or "").strip():
            continue
        try:
            when = dt.datetime.strptime(
                raw["Date"].strip() + " " + raw["Time"].strip(), "%d/%m/%Y %H.%M.%S"
            )
        except (TypeError, ValueError):
            invalid_timestamp_rows += 1
            continue
        parsed_timestamp_rows += 1
        row = {"timestamp": when}
        for column in COLUMNS[2:]:
            row[column] = parse_number(raw.get(column))
            if row[column] is None:
                missing_counts[column] += 1
        rows.setdefault(when, row)

    ordered = sorted(rows.values(), key=lambda item: item["timestamp"])
    quality = {
        "physical_csv_rows_including_header_and_blank_tail": len(physical_rows),
        "documented_rows_including_header": len(ordered) + 1,
        "measurement_observations": len(ordered),
        "blank_trailing_rows": max(0, len(physical_rows) - len(ordered) - 1),
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "duplicate_timestamps_removed": parsed_timestamp_rows - len(ordered),
        "missing_by_variable_after_minus_200_conversion": dict(missing_counts),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }
    return ordered, quality


def percentile(values, p):
    values = sorted(values)
    h = (len(values) - 1) * p
    i, j = math.floor(h), math.ceil(h)
    return values[i] if i == j else values[i] * (j - h) + values[j] * (h - i)


def build_thresholds(train):
    result = {}
    for name, (column, unit) in MEASURES.items():
        values = [row[column] for row in train if row[column] is not None]
        result[name] = {
            "q25": percentile(values, 0.25),
            "q75": percentile(values, 0.75),
            "unit": unit,
            "training_non_missing": len(values),
        }
    return result


def level(value, threshold):
    if value is None:
        return "unavailable"
    if value < threshold["q25"]:
        return "low"
    if value > threshold["q75"]:
        return "elevated"
    return "typical"


def fmt(value, unit):
    return "missing" if value is None else f"{value:.2f} {unit}"


def examples(rows, thresholds):
    for row in rows:
        timestamp = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")
        values = {name: row[column] for name, (column, _) in MEASURES.items()}
        labels = {name: level(values[name], thresholds[name]) for name in MEASURES}
        readings = "; ".join(
            f"{name}: {fmt(values[name], MEASURES[name][1])}" for name in ("CO", "NO2", "benzene")
        )
        context = f"Hour: {timestamp}. {readings}."
        classify_input = (
            "Classify these historical air-quality measurements as low, typical, elevated or unavailable "
            "relative to training-distribution quartiles. " + context
        )
        classify_target = "; ".join(f"{name}: {labels[name]}" for name in ("CO", "NO2", "benzene")) + "."
        explain_input = (
            "Explain these historical air-quality measurements in one factual sentence. Use relative historical "
            "quartiles only; do not infer official AQI, health safety or a pollution source. " + context
        )
        explain_target = (
            f"Relative to historical training data, CO is {labels['CO']}, NO2 is {labels['NO2']}, "
            f"and benzene is {labels['benzene']}."
        )
        for task, source, target in (
            ("classify", classify_input, classify_target),
            ("explain", explain_input, explain_target),
        ):
            yield {
                "text": source,
                "label": target,
                "task": task,
                "record_id": timestamp,
                "timestamp": timestamp,
                "input_text": source,
                "target_text": target,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/AirQualityUCI.csv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    source_path, out = Path(args.csv), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, quality = load_rows(source_path)
    n = len(rows)
    train_end = int(n * 0.70)
    validation_size = round(n * 0.15)
    split_rows = {
        "train": rows[:train_end],
        "validation": rows[train_end:train_end + validation_size],
        "test": rows[train_end + validation_size:],
    }
    assert split_rows["train"][-1]["timestamp"] < split_rows["validation"][0]["timestamp"]
    assert split_rows["validation"][-1]["timestamp"] < split_rows["test"][0]["timestamp"]
    thresholds = build_thresholds(split_rows["train"])

    metadata = {
        "source": "UCI Air Quality Dataset (ID 360)",
        "source_file": str(source_path.as_posix()),
        "quality": quality,
        "split_strategy": "chronological 70/15/15 on unique source hours before prompt generation",
        "random_seed": None,
        "label_thresholds_fitted_on": "train only",
        "thresholds": thresholds,
        "splits": {},
    }
    id_sets = []
    for name, subset in split_rows.items():
        path = out / f"{name}.csv"
        generated = list(examples(subset, thresholds))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(generated)
        ids = {item["record_id"] for item in generated}
        id_sets.append(ids)
        metadata["splits"][name] = {
            "source_hours": len(subset),
            "examples": len(generated),
            "date_start": subset[0]["timestamp"].isoformat(),
            "date_end": subset[-1]["timestamp"].isoformat(),
            "file": path.as_posix(),
            "sha256": sha256(path),
        }
    assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])
    (out / "dataset_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("VALIDATION PASSED: chronological boundaries and record IDs are disjoint.")


if __name__ == "__main__":
    main()
