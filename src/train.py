#!/usr/bin/env python3
"""Fine-tune google/flan-t5-small. Requires dependencies from requirements.txt."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/flan-t5-airquality")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=64)
    args = parser.parse_args()
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
            Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
        )
    except ImportError as exc:
        raise SystemExit(f"Training blocked: missing dependency {exc.name}. Run: pip install -r requirements.txt")

    set_seed(args.seed)
    files = {name: str(Path(args.data_dir) / f"{name}.csv") for name in ("train", "validation", "test")}
    dataset = load_dataset("csv", data_files=files)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    def tokenize(batch):
        inputs = tokenizer(batch["input_text"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=args.max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output), learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, predict_with_generate=True,
        fp16=torch.cuda.is_available(), seed=args.seed, report_to="none", logging_steps=100,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"], processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    train_result = trainer.train()
    validation = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    test = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(str(output / "best_model"))
    tokenizer.save_pretrained(str(output / "best_model"))
    result = {
        "status": "Completed", "model": "google/flan-t5-small", "seed": args.seed,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "configuration": vars(args), "train_metrics": train_result.metrics,
        "validation_metrics": validation, "test_metrics": test,
    }
    (output / "training_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
