FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY photodate/ photodate/

EXPOSE 8000

CMD ["uvicorn", "photodate.web:app", "--host", "0.0.0.0", "--port", "8000"]
