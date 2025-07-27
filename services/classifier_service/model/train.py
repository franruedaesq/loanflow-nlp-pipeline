"""
Train a DistilBERT text-classifier on the synthetic dataset produced in Stage 0.
Outputs:
  • ./saved_model/            – Hugging Face fine-tuned checkpoint
  • ./onnx_model/             – ONNX-Runtime exported model
  • ./outputs/                – trainer logs, metrics CSV / JSON
Run:
  python -m model.train --data ../../data/processed/train.jsonl
"""

import argparse
import json
import os
import pathlib
import random
from collections import Counter
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_dataset(jsonl_path: Path) -> Dataset:
    records = [json.loads(line) for line in jsonl_path.open()]
    ds = Dataset.from_list(records)
    ds = ds.shuffle(seed=42)
    return ds.train_test_split(test_size=0.2, seed=42)


def encode(ds, tokenizer, label2id):
    def tokenize(batch):
        return tokenizer(
            batch["free_text"], truncation=True, padding="max_length", max_length=128
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("reason", "label")
    ds = ds.map(lambda x: {"label": label2id[x["label"]]}, batched=False)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def train(args):
    ds = load_dataset(args.data)

    labels = sorted(set(ds["train"]["reason"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Label distribution:", Counter(ds["train"]["reason"]))

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    ds_encoded = encode(ds, tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1": metric_f1.compute(
                average="weighted", predictions=preds, references=labels
            )["f1"],
        }

    training_args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        run_name="phase1_classifier_v1",
        report_to=[],
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_encoded["train"],
        eval_dataset=ds_encoded["test"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    resume = Path("outputs").glob("checkpoint-*")
    resume = next(resume, None)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    Path("saved_model").mkdir(exist_ok=True)
    trainer.save_model("saved_model")
    tokenizer.save_pretrained("saved_model")

    # Export to ONNX
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        "saved_model", export=True
    )
    ort_model.save_pretrained("onnx_model")
    tokenizer.save_pretrained("onnx_model")

    # Save training logs
    log_history = trainer.state.log_history
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../../data/raw/all_examples.jsonl"),
        help="Path to merged JSONL dataset (Stage 0 output)",
    )
    parser.add_argument("--model-ckpt", default="distilbert-base-uncased")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
