FROM python:3.12-slim

WORKDIR /app

# System dependencies: pillow-heif (HEIC support) and exiftool (dates in HEIC/PNG without re-encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libheif-dev \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install dependencies
COPY pyproject.toml .
COPY photodate/ photodate/
RUN pip install --no-cache-dir .

# Data volume for settings and album data
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "photodate.web:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
