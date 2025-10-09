FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-interaction || true
COPY . /app
CMD ["uvicorn", "local_stub:app", "--host", "0.0.0.0", "--port", "9000"]
