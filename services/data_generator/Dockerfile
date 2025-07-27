FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION= \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/app

# ----- Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# ----- Source code
COPY app/ app/
COPY cli.py .

# The output folder will be mounted by docker-compose
VOLUME ["/opt/output"]

# ENTRYPOINT ["python", "cli.py"]
