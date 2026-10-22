# ── Stage 1: compile dlib + build all wheels ───────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# ── Stage 2: lean runtime image ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Only the shared libraries needed at runtime by dlib / pillow-heif
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0-serial \
    libheif1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels photodate \
    && rm -rf /wheels

COPY photodate/ photodate/

VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "photodate.web:app", "--host", "0.0.0.0", "--port", "8000"]
