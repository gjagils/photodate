FROM python:3.12-slim

WORKDIR /app

# Copy source and install dependencies
COPY pyproject.toml .
COPY photodate/ photodate/
RUN pip install --no-cache-dir .

# Data volume for settings and album data
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "photodate.web:app", "--host", "0.0.0.0", "--port", "8000"]
