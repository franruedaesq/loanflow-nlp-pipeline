import json
import random
from pathlib import Path
from typing import List

from .config import EXAMPLES_PER_REASON, RANDOM_SEED
from .models import Example, Reason, Step, StructuredExample
from .openai_client import chat_completion

# --------------------------------------------------------------------------- #
OUT_DIR = Path("generated_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)


def _prompt(reason: Reason, n: int) -> str:
    return (
        "You are a data generator producing realistic messages from mortgage web applicants, the user is Loan Officer"
        "explaining why they abandoned the process. The process consist of pricing, credit report generation,"
        "liabilities, declarations, repricing, rate lock, loan submission, compliance, disclosure"
        "Return ONLY a JSONL block with the "
        "keys 'free_text' and 'step'. Do NOT add anything else. Use English, first‑person, colloquial. "  # Removed 'id' from the keys to include instruction
        f"The abandonment reason is: {reason.value}. Generate exactly {n} distinct examples."
    )


def generate_examples(reason: Reason, n: int) -> List[Example]:
    examples: List[Example] = []
    print(len(examples), n)
    # print("Generating examples for reason:", reason.value)
    while len(examples) < n:
        print(f"Generating {n} examples for reason: {reason.value}")
        try:
            record = chat_completion(_prompt(reason, n), StructuredExample)
            print("Record received:", record)
            print("Record type:", type(record))

            ex = Example(
                reason=reason, free_text=record["free_text"], step=Step(record["step"])
            )
            print(f"Generated example: {ex}")
            examples.append(ex)
            print(f"Current count: {len(examples)}")
        except Exception as e:
            print(f"Error generating example: {e}")
            print(f"Error type: {type(e)}")
            continue
    return examples


class Generator:
    """High-level helper to create per-reason files and a unified JSONL."""

    def __init__(
        self, n_per_reason: int = EXAMPLES_PER_REASON, out_dir: Path = OUT_DIR
    ):
        self.n = n_per_reason
        self.out_dir = out_dir

    # --------------------------------------------------------------------- #
    def run(self):
        total_reasons = len(Reason)
        total_expected = total_reasons * self.n
        global_count = 0

        for reason in Reason:
            chunk = generate_examples(reason, self.n)
            file_path = self.out_dir / f"{reason.value}.jsonl"

            with file_path.open("w", encoding="utf-8") as f:
                for i, ex in enumerate(chunk, 1):
                    f.write(ex.model_dump_json())
                    f.write("\n")
                    global_count += 1
                    print(
                        f"{global_count}/{total_expected} saved "
                        f"({i}/{self.n} in {file_path.name})"
                    )

            print(f"✅ File ready: {file_path} with {len(chunk)} examples\n")

    # --------------------------------------------------------------------- #
    def merge(self, dst_path: Path | None = None):
        """Concatenate all *.jsonl into one file."""
        dst_path = dst_path or Path("all_examples.jsonl")
        files = sorted(self.out_dir.glob("*.jsonl"))
        count = 0

        with dst_path.open("w", encoding="utf-8") as out:
            for file in files:
                with file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        out.write(line + "\n")
                        count += 1
                        print(f"{count} lines written " f"(last from {file.name})")

        print(f"✅ Merged {count} lines from {len(files)} files into {dst_path}")


# Convenience functional wrappers ------------------------------------------------
def main():
    gen = Generator()
    gen.run()
    gen.merge()


if __name__ == "__main__":  # allows `python -m app.generator`
    main()
