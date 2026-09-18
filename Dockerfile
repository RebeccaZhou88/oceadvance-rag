# 多阶段构建，减小镜像体积
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
# 先复制依赖声明，利用 Docker 缓存
COPY backend/pyproject.toml backend/requirements.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:$PATH \
    PYTHONPATH=/app/backend/src

WORKDIR /app
COPY --from=builder /install /usr/local
# 前后端分离：backend/ (src layout) + frontend/ (静态)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
EXPOSE 8000
# 静态前端目录指向 /app/frontend
ENV FRONTEND_DIR=/app/frontend
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
