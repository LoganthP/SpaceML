# Space ML Mission Control - one image: built dashboard + FastAPI on the same origin (port 8000).
#   docker compose up -d                 -> http://localhost:8080
# Build args:
#   TORCH_INDEX   CPU PyTorch wheels by default (~1 GB image instead of ~6 GB with CUDA);
#                 "none" builds a lite image without deep learning (telemetry LSTM, Mars, exoplanet CNN off)
#   TORCH_VERSION / TORCHVISION_VERSION   a matched pair from the same index (default 2.6.0 / 0.21.0, the
#                 documented CPU pair). They are installed together and then locked with a constraints file,
#                 so segmentation-models-pytorch/timm can never pull a different (e.g. CUDA) torchvision from
#                 PyPI - that mix is what caused "operator torchvision::nms does not exist".
#   NODE_IMAGE / PYTHON_IMAGE   base images (e.g. a registry mirror)
ARG NODE_IMAGE=node:20-alpine
ARG PYTHON_IMAGE=python:3.11-slim

# ---- 1. dashboard (React/Vite) -------------------------------------------------------------
FROM ${NODE_IMAGE} AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. API + pipelines ------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS app
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SPACEML_DATA_DIR=/app/data SPACEML_MODELS_DIR=/app/models SPACEML_STORAGE_DIR=/app/storage \
    SPACEML_FRONTEND_DIST=/app/frontend/dist SPACEML_STARTER_DIR=/app/starter \
    HF_HOME=/app/storage/cache/huggingface TORCH_HOME=/app/storage/cache/torch
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements-core.txt backend/requirements-ml.txt ./
RUN pip install -r requirements-core.txt
COPY backend/scripts/check_torch.py /tmp/check_torch.py
RUN if [ "$TORCH_INDEX" != "none" ]; then \
        pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX" \
        && pip freeze | grep -iE '^(torch|torchvision)==' > /app/torch-constraints.txt \
        && cat /app/torch-constraints.txt \
        && pip install -r requirements-ml.txt -c /app/torch-constraints.txt \
        && pip check \
        && python /tmp/check_torch.py; \
    else touch /app/torch-constraints.txt; \
    fi
# Any later "pip install" inside the image keeps the matched pair too:
ENV PIP_CONSTRAINT=/app/torch-constraints.txt
COPY backend/ ./
COPY starter/ /app/starter/
COPY samples/ /app/samples/
COPY --from=web /web/dist /app/frontend/dist
RUN useradd --create-home --uid 1000 smc && mkdir -p /app/data /app/models /app/storage \
    && chown -R smc:smc /app/data /app/models /app/storage /app/samples
USER smc
VOLUME ["/app/data", "/app/models", "/app/storage"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=8 \
    CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
