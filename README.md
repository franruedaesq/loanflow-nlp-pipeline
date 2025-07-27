# NLP-Driven Decision System • Synthetic-Data-First Portfolio Project

> Modular AI stack for analyzing why loan applicants abandon a mortgage workflow,  
> featuring synthetic-data generation, multi-model NLP serving, full observability, and AWS-ready deployment.

---

## 1 Project Goals

| Goal                                              | Why it matters                                                                                                                                       |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Generate realistic synthetic data**             | Demonstrates ability to create custom datasets for rare or private scenarios – a frequent requirement in regulated industries (finance, healthcare). |
| **Serve multiple NLP models behind a single API** | Mirrors production patterns: lightweight classifier, fine-tuned NER, quantised LLM for explanations.                                                 |
| **Full MLOps/DevOps pipeline**                    | Shows expertise in Docker, CI/CD, Terraform (ECS ≈ production), monitoring with Prometheus + Grafana + Evidently.                                    |
| **Portfolio-friendly**                            | Each micro-service runs independently. Reviewers can spin up only what they want to inspect.                                                         |

---

## 2 Repository Layout (excerpt)

```
nlp-decision-system/
├── data/ # generated datasets end up here
│ └── raw/
├── services/
│ ├── data_generator/ # Stage 0 – synthetic data
│ │ ├── app/ # Python package
│ │ ├── cli.py # `python cli.py -n 50 -o /opt/output`
│ │ ├── requirements.txt
│ │ └── Dockerfile
│ └── ... # classifier_service, ner_service, etc.
├── infra/ # Terraform for AWS (ECS + ECR)
├── monitoring/ # Prometheus + Grafana dashboards
├── docker-compose.yml # local orchestration
└── Makefile # convenience targets
```

A full tree is in [`docs/architecture.md`](docs/architecture.md).

---

## 3 Stage 0 – Synthetic Data Generator

### 3.1 Purpose

Creates English, first-person, colloquial messages that mimic loan officers explaining **why** a borrower could not continue.
Labels follow the `Reason` enum (e.g. `tech_error`, `missing_info`, …); each example also records the UI **step** where the abandonment occurred.

### 3.2 Key files

- `app/models.py` – Pydantic models (`Reason`, `Step`, `Example`).
- `app/generator.py` – Core logic; calls OpenAI with exponential back-off.
- `cli.py` – Command-line wrapper; accepts `--n-per-reason` and `--outdir`.
- `Dockerfile` – Turns the generator into a reproducible one-shot container.

---

## 4 Running Locally

### 4.1 Prerequisites

- **Python 3.11+** or **Docker 23+**
- OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

### 4.2 Option A – Python virtualenv

```bash
# repo root
cd services/data_generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py -n 50 -o ../../../data/raw          # creates JSONL files
```

### 4.3 Option B – Docker Compose (recommended)

```bash
# repo root
docker compose build data_generator
docker compose run --rm data_generator
# Output appears under ./data/raw/
```

---

## 5 Makefile Shortcuts

```bash
make generate-data      # Build image + run service with default params
make clean-data         # Remove ./data/raw/generated_data
```

_(See `Makefile` for details.)_

---

## 6 Roadmap

| Stage | Deliverable                                | Status   |
| ----- | ------------------------------------------ | -------- |
| 0     | Synthetic-data micro-service               | ✅ Ready |
| 1     | PyTorch text classifier + API              | 🚧 WIP   |
| 2     | LoRA fine-tuned NER model + API            | Planned  |
| 3     | Quantised LLM explainer + API              | Planned  |
| 4     | Model selector micro-service               | Planned  |
| 5     | FastAPI gateway + Prometheus metrics       | Planned  |
| 6     | AWS Terraform (ECR + ECS Fargate) pipeline | Planned  |

---

## 7 Contributing / Questions

This project is part of a personal portfolio; constructive feedback is welcome.
Open an issue or reach out on **LinkedIn**: `francisco-rueda-esq`.

---

## 8 Licence

MIT licence © 2025 Francisco Rueda.
Generated examples are synthetic; no real borrower data is included.
