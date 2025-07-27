FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /opt/app

# ----- install deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# ----- copy source
COPY model/ model/
COPY api/ api/

# expose port for FastAPI
EXPOSE 8000

# default CMD serves the API; training can be run by overriding
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
