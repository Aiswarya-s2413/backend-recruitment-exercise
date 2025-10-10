FROM python:3.13-slim
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* /app/

# Install Poetry and all dependencies
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root --verbose

# Copy application code
COPY . /app

# Start the FastAPI app
CMD ["uvicorn", "local_stub:app", "--host", "0.0.0.0", "--port", "9000"]
