# autoCrypto Docker 镜像
FROM python:3.12-slim

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先装依赖（利用 Docker 缓存）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 复制源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/
COPY main.py ./

# 输出目录
RUN mkdir -p /app/outputs

# 默认启动纸上交易（安全）
CMD ["uv", "run", "python", "scripts/run_production.py", "--dry-run"]
