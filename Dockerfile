FROM python:3.12-slim

WORKDIR /app

# System dependencies for face_recognition (dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all source files first, then install
COPY pyproject.toml .
COPY photodate/ photodate/
RUN pip install --no-cache-dir .

# Data volume for settings and album data
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "photodate.web:app", "--host", "0.0.0.0", "--port", "8000"]
