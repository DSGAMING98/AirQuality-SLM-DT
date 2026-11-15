"""AirQuality SLM v1.1: validated standalone inference for the public checkpoint.

Requires: torch, transformers, huggingface_hub.
No retraining or mutation of model.pt is performed.

Examples:
  python safe_inference.py --self-test
  python safe_inference.py --prompt "Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3."
  python safe_inference.py --prompt-file prompt.txt --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Prevent Transformers from importing TensorFlow/Keras unnecessarily.
os.environ.setdefault("USE_TF", "0")

DEFAULT_REPO = "prajwal90/AirQuality-SLM-Scratch-Qwen"
CLASSIFY_HEADER = (
    "Classify these historical air-quality measurements as low, typical, "
    "elevated or unavailable relative to training-distribution quartiles."
)
EXPLAIN_HEADER = (
    "Explain these historical air-quality measurements in one factual sentence. "
    "Use relative historical quartiles only; do not infer official AQI, "
    "health safety or a pollution source."
)
DATA_START = datetime.fromisoformat("2004-03-10T18:00:00")
DATA_END = datetime.fromisoformat("2005-04-04T14:00:00")
UNITS = {"co": "mg/m3", "no2": "ug/m3", "benzene": "ug/m3"}
LABELS = ("low", "typical", "elevated", "unavailable")
LABEL_PATTERN = r"(low|typical|elevated|unavailable)"
STRUCTURE = re.compile(
    r"Hour:\s*(?P<hour>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\s*"
    r"CO:\s*(?P<co>[^;\r\n]+);\s*"
    r"NO2:\s*(?P<no2>[^;\r\n]+);\s*"
    r"benzene:\s*(?P<benzene>[^;\r\n]+)\.\s*",
    re.IGNORECASE,
)
SENTINEL = re.compile(r"-200(?:\.0+)?(?:\s+(mg/m3|ug/m3))?", re.IGNORECASE)
NUMERIC = re.compile(r"\+?(?:\d+(?:\.\d+)?|\.\d+)\s+(mg/m3|ug/m3)", re.IGNORECASE)
CLASSIFICATION = re.compile(
    rf"CO:\s*{LABEL_PATTERN};\s*NO2:\s*{LABEL_PATTERN};\s*benzene:\s*{LABEL_PATTERN}\.\s*",
    re.IGNORECASE,
)
EXPLANATION = re.compile(
    rf"Relative to historical training data,\s*CO is\s*{LABEL_PATTERN},\s*"
    rf"NO2 is\s*{LABEL_PATTERN},\s*and benzene is\s*{LABEL_PATTERN}\.\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidatedPrompt:
    kind: str
    normalized_prompt: str
    readings: dict[str, float | None]


def parse_reading(raw: str, unit: str) -> tuple[float | None, str]:
    """Return (reading, model-facing field). Never use the -200 sentinel as a value."""
    raw = raw.strip()
    if raw.lower() == "missing":
        return None, "missing"
    sentinel = SENTINEL.fullmatch(raw)
    if sentinel:
        actual_unit = sentinel.group(1)
        if actual_unit and actual_unit.lower() != unit:
            raise ValueError("incorrect unit on missing marker")
        return None, "missing"
    match = NUMERIC.fullmatch(raw)
    if not match:
        raise ValueError("malformed reading, negative value or missing unit")
    if match.group(1).lower() != unit:
        raise ValueError(f"expected unit {unit}")
    value_str = raw.split()[0]
    value = float(value_str)
    if not math.isfinite(value) or value < 0:
        raise ValueError("reading must be finite and nonnegative")
    return value, raw


def validate_prompt(prompt: str) -> tuple[ValidatedPrompt | None, str | None]:
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "Request must contain a text instruction."
    prompt = prompt.strip()
    if prompt.startswith(CLASSIFY_HEADER):
        kind, header = "classify", CLASSIFY_HEADER
    elif prompt.startswith(EXPLAIN_HEADER):
        kind, header = "explain", EXPLAIN_HEADER
    else:
        return None, (
            "Unsupported request. Only historical relative air-quality classification "
            "and explanation are supported. Official AQI forecasts and health-safety "
            "advice are not supported."
        )
    match = STRUCTURE.fullmatch(prompt[len(header):].strip())
    if not match:
        return None, "Malformed record. Supply Hour, CO, NO2 and benzene in the supported format."
    try:
        timestamp = datetime.strptime(match.group("hour"), "%Y-%m-%dT%H:%M:%S")
        if not (DATA_START <= timestamp <= DATA_END):
            return None, "Timestamp outside supported UCI historical collection period."
        readings: dict[str, float | None] = {}
        canonical: dict[str, str] = {}
        for name, unit in UNITS.items():
            readings[name], canonical[name] = parse_reading(match.group(name), unit)
    except ValueError as exc:
        return None, f"Invalid measurement: {exc}"
    normalized = (
        f"{header} Hour: {timestamp.isoformat()}. "
        f"CO: {canonical['co']}; NO2: {canonical['no2']}; "
        f"benzene: {canonical['benzene']}."
    )
    return ValidatedPrompt(kind=kind, normalized_prompt=normalized, readings=readings), None


def check_output(output: str, validated: ValidatedPrompt) -> str | None:
    """Check format and missingness consistency, not statistical correctness."""
    regex = CLASSIFICATION if validated.kind == "classify" else EXPLANATION
    match = regex.fullmatch(output.strip())
    if not match:
        return "Generated output does not match the required three-pollutant response format."
    parsed = dict(zip(UNITS.keys(), (label.lower() for label in match.groups())))
    for name, value in validated.readings.items():
        if value is None and parsed[name] != "unavailable":
            return f"Generated output contradicts missing {name} measurement."
        if value is not None and parsed[name] == "unavailable":
            return f"Generated output marks present {name} measurement unavailable."
    return None


class PublicCheckpoint:
    """Lazily fetch and load custom model code + state dict from HF Hub."""
    def __init__(self, repo_id: str = DEFAULT_REPO, local_dir: str | None = None,
                 device: str = "cpu", revision: str | None = None):
        self.repo_id, self.local_dir, self.revision = repo_id, local_dir, revision
        self.device_name = device
        self._loaded = False

    def _load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        import torch

        if self.local_dir:
            folder = Path(self.local_dir).expanduser().resolve()
        else:
            folder = Path(snapshot_download(
                repo_id=self.repo_id, revision=self.revision,
                allow_patterns=[
                    "model.pt", "model_config.json", "airquality_slm_model.py",
                    "tokenizer*", "vocab.json", "merges.txt",
                    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
                ],
            ))
        for required in ("model.pt", "model_config.json", "airquality_slm_model.py"):
            if not (folder / required).is_file():
                raise FileNotFoundError(f"Model release is missing {required}")
        # The Python implementation is intentionally the audited custom source from
        # the user's HF repo. Like any imported Python file, it must be trusted.
        module_name = "_airquality_slm_release_model"
        spec = importlib.util.spec_from_file_location(module_name, folder / "airquality_slm_model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to import custom model implementation")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        config = module.MiniQwenConfig(**json.loads(
            (folder / "model_config.json").read_text(encoding="utf-8")
        ))
        model = module.MiniQwenModel(config)
        state_dict = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        if self.device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA selected, but no GPU is available")
        self.device = torch.device(device_name)
        self.model = model.to(self.device).eval()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, use_fast=True)
        if len(self.tokenizer) != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model configuration")
        self._loaded = True

    def __call__(self, prompt: str) -> str:
        import torch
        if not self._loaded:
            self._load()
        prefix = f"Instruction:\n{prompt}\nAnswer:\n"
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)[-(self.config.max_seq_len - 50):]
        if not ids:
            raise ValueError("Tokenizer produced no input tokens")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(tensor, eos_token_id=self.tokenizer.eos_token_id,
                                         max_new_tokens=50)
        return self.tokenizer.decode(output[0, tensor.shape[1]:].tolist(),
                                     skip_special_tokens=True).strip()


def safe_generate(prompt: str, predictor: Callable[[str], str]) -> dict:
    validated, error = validate_prompt(prompt)
    if error:
        return {"status": "REJECTED", "answer": f"REQUEST REJECTED: {error}",
                "model_called": False}
    try:
        answer = predictor(validated.normalized_prompt).strip()
    except Exception as exc:
        return {"status": "ERROR", "answer": f"Inference failed: {type(exc).__name__}: {exc}",
                "model_called": True}
    output_error = check_output(answer, validated)
    if output_error:
        return {"status": "ABSTAINED", "answer": f"OUTPUT REJECTED: {output_error}",
                "model_called": True}
    return {"status": "ACCEPTED", "answer": answer, "model_called": True}


POSITIVE_CASES = [
    ("P1", CLASSIFY_HEADER + " Hour: 2005-03-08T08:00:00. CO: 3.40 mg/m3; NO2: 182.00 ug/m3; benzene: 18.30 ug/m3.",
     "CO: elevated; NO2: elevated; benzene: elevated."),
    ("P2", CLASSIFY_HEADER + " Hour: 2005-02-11T10:00:00. CO: 5.40 mg/m3; NO2: 291.00 ug/m3; benzene: missing.",
     "CO: elevated; NO2: elevated; benzene: unavailable."),
    ("P3", EXPLAIN_HEADER + " Hour: 2005-03-06T09:00:00. CO: 0.70 mg/m3; NO2: 79.00 ug/m3; benzene: 1.50 ug/m3.",
     "Relative to historical training data, CO is low, NO2 is typical, and benzene is low."),
]
NEGATIVE_CASES = [
    ("N1", CLASSIFY_HEADER + " Hour: 2005-02-05T03:00:00. CO: missing; NO2: missing; benzene: missing.",
     "ACCEPTED", "CO: unavailable; NO2: unavailable; benzene: unavailable.", True),
    ("N2", CLASSIFY_HEADER + " Hour: 2005-02-05T03:00:00. CO: banana mg/m3; NO2: ??? ug/m3; benzene: unknown ug/m3.",
     "REJECTED", None, False),
    ("N3", "Predict tomorrow's official AQI in Bengaluru and tell me whether it is safe to exercise outside.",
     "REJECTED", None, False),
]


def run_self_test(predictor: Callable[[str], str]) -> int:
    passed = 0
    for name, prompt, expected in POSITIVE_CASES:
        result = safe_generate(prompt, predictor)
        ok = result["status"] == "ACCEPTED" and result["answer"] == expected and result["model_called"]
        print(f"{name}: {'PASS' if ok else 'FAIL'} | {result['status']} | {result['answer']}")
        passed += ok
    for name, prompt, expected_status, expected_answer, called in NEGATIVE_CASES:
        result = safe_generate(prompt, predictor)
        ok = (result["status"] == expected_status and result["model_called"] == called
              and (expected_answer is None or result["answer"] == expected_answer))
        print(f"{name}: {'PASS' if ok else 'FAIL'} | {result['status']} | {result['answer']}")
        passed += ok
    print(f"Safe-inference release self-test: {passed}/6 passed")
    return 0 if passed == 6 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One supported historical air-quality instruction")
    source.add_argument("--prompt-file", help="Text file containing an instruction")
    source.add_argument("--self-test", action="store_true", help="Run six published functional/robustness cases")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=None, help="Optional fixed HF commit SHA")
    parser.add_argument("--model-dir", default=None, help="Use a local release folder instead of HF download")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--json", action="store_true", help="Machine-readable single-result JSON")
    args = parser.parse_args(argv)
    backend = PublicCheckpoint(args.repo_id, args.model_dir, args.device, args.revision)
    if args.self_test:
        return run_self_test(backend)
    prompt = (Path(args.prompt_file).read_text(encoding="utf-8")
              if args.prompt_file else args.prompt)
    result = safe_generate(prompt, backend)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"STATUS: {result['status']}")
        print(f"MODEL CALLED: {result['model_called']}")
        print(f"ANSWER: {result['answer']}")
    return {"ACCEPTED": 0, "REJECTED": 2, "ABSTAINED": 3, "ERROR": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())