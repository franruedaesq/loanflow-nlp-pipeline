from pathlib import Path

from .generator import Generator


def merge_jsonl_files(output: str = "all_examples.jsonl"):
    Generator().merge(Path(output))
