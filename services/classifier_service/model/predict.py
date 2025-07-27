"""
Lightweight ONNXRuntime inference wrapper with threshold logic.
Usage:
    predictor = Predictor("onnx_model")
    res = predictor(text)
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

THRESHOLD = 0.65
TOPK = 3


class Predictor:
    def __init__(self, model_dir: str | Path = "onnx_model"):
        model_dir = Path(model_dir)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Load label-map saved in config.json
        config = (model_dir / "config.json").read_text()
        id2label = {
            int(k): v for k, v in __import__("json").loads(config)["id2label"].items()
        }
        self.id2label = id2label

    def __call__(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        logits = self.session.run(None, dict(enc))[0]
        probs = softmax(logits, axis=-1)[0]

        top_idx = int(np.argmax(probs))
        top_score = float(probs[top_idx])
        top_label = self.id2label[top_idx]

        if top_score >= THRESHOLD:
            return {"label": top_label, "score": top_score}
        order = probs.argsort()[::-1][:TOPK]
        return {
            "label": None,
            "score": top_score,
            "candidates": [(self.id2label[i], float(probs[i])) for i in order],
        }
