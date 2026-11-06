# =========================================================
# Stage 1 — Builder Stage: install Python dependencies
# =========================================================
FROM python:3.12.13-slim AS builder

WORKDIR /app

COPY 98proxy /etc/apt/apt.conf.d/98proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_INSTALLER_METADATA=1 \
    UV_LINK_MODE=copy \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH="/app" \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-dev --no-install-project


# =========================================================
# Stage 2 — Runtime Stage: lightweight final image
# =========================================================
FROM python:3.12.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH="/app" \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/app \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1

WORKDIR $HOME

COPY 98proxy /etc/apt/apt.conf.d/98proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    && rm -f /etc/apt/apt.conf.d/98proxy \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
COPY --from=builder /app/.venv $HOME/.venv

COPY celery_tasks $HOME/celery_tasks
COPY src $HOME/src
# COPY config $HOME/config

CMD ["celery", "-A", "celery_tasks.celery_app:celery_app", "worker", \
     "--loglevel=INFO", "--pool=solo", \
     "-Q", "docassembler,celery"]
# CMD ["celery", "-A", "celery_tasks.celery_app:celery_app", "worker", "--loglevel=DEBUG", "-P", "solo", "-Q", "docassembler,celery"]
