# Multi-stage build to reduce image size
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
# Copy dependency manifest first to leverage Docker cache
COPY backend/pyproject.toml backend/requirements.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:$PATH \
    PYTHONPATH=/app/backend/src

WORKDIR /app
COPY --from=builder /install /usr/local
# Frontend/backend split: backend/ (src layout) + frontend/ (static)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
EXPOSE 8000
# Static frontend directory points to /app/frontend
ENV FRONTEND_DIR=/app/frontend
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
