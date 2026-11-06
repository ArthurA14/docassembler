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

# RUN --mount=type=cache,target=/root/.cache/uv \
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=0 \
    UV_HTTP_TIMEOUT=300 \
    uv sync --frozen --no-dev --no-install-project


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
    FASTAPI_ENV=dev \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1

WORKDIR $HOME

COPY 98proxy /etc/apt/apt.conf.d/98proxy

ARG GROUPUSER=appgroup \
    APPUSER=appuser \
    GROUPID=1000 \
    USERID=1000

RUN groupadd -g ${GROUPID} ${GROUPUSER} \
    && useradd -g ${GROUPUSER} \
        -d ${HOME} \
        -m -u ${USERID} ${APPUSER}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    && rm -f /etc/apt/apt.conf.d/98proxy \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
COPY --from=builder --chown=${APPUSER}:${GROUPUSER} /app/.venv $HOME/.venv

RUN mkdir -p \
    $HOME/config \
    $HOME/prompts \
    && chown ${APPUSER}:${GROUPUSER} \
    $HOME \
    $HOME/config \
    $HOME/prompts

COPY --chown=${APPUSER}:${GROUPUSER} api_gateway $HOME/api_gateway
COPY --chown=${APPUSER}:${GROUPUSER} celery_tasks $HOME/celery_tasks
# COPY --chown=${APPUSER}:${GROUPUSER} src $HOME/src
# COPY --chown=${APPUSER}:${GROUPUSER} config $HOME/config
# COPY --chown=${APPUSER}:${GROUPUSER} prompts $HOME/prompts

USER ${APPUSER}

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:${API_PORT:-5000}/health || exit 1

ENTRYPOINT ["sh", "-c"]
CMD ["if [ \"$FASTAPI_ENV\" = \"dev\" ]; then \
    exec /app/.venv/bin/python -m uvicorn api_gateway.main:app --host 0.0.0.0 --port ${API_PORT:-5000} --reload; \
else \
    exec /app/.venv/bin/python -m uvicorn api_gateway.main:app --host 0.0.0.0 --port ${API_PORT:-5000} --workers ${UVICORN_WORKERS:-1}; \
fi"]
