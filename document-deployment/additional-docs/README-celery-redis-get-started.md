# Celery + Redis Integration Guide (Get Started)

This guide explains how to integrate **Celery** (distributed task queue) and **Redis** (message broker & result backend) into your project, ensuring version stability and reproducibility using [`uv`](https://github.com/astral-sh/uv) and Docker.

**Why Celery + Redis ?**  
- Celery pushes tasks into a Redis queue.
- Workers pick up and execute tasks asynchronously.
- Ideal for decoupling workloads and scaling horizontally.

---------------------------------------------------------------------------

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv) installed (v0.9.5+)
- Docker (28.0.2) & Docker Compose (v2.34.0)
- Python ≥3.12

---

## Step 1: Update Dependencies in `pyproject.toml`

Add the following **exact versions** to ensure compatibility-especially with `chonkie`, which is version-sensitive:

```toml
[project.dependencies]
celery = ">=5.6.2"
redis = ">=7.3.0"
chonkie = "1.4.0"  # FIXED: Prevents accidental upgrade (e.g. 1.5+ would break existing code)
```

**Critical Note:**<br>
Without setting `chonkie==1.4.0`, resolving `celery/redis` may pull a newer `chonkie` version. Since `chonkie` APIs change across minor versions, this will cause **runtime errors** unless code is updated accordingly.

---------------------------------------------------------------------------

## Step 2: Lock & Sync Dependencies

Generate/update your lock file with deterministic builds:
```bash
uv lock
uv sync
```

→ This ensures `uv.lock` reflects **all pinned versions** (including transitive deps) and avoids drift between environments.

---------------------------------------------------------------------------

## Step 3: Dockerize the services - example of implementation from a import service
### Producer (FastAPI App) (`./api_gateway/import/Dockerfile_producer`)

Producer Dockerfile - **key additions** and clarifications:

- `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` already set in `runtime` stage:
    ```dockerfile
    ENV CELERY_BROKER_URL=redis://redis:6379/0 \
        CELERY_RESULT_BACKEND=redis://redis:6379/1
    ```   
- `pyproject.toml` and `uv.lock` are copied *before* dependency sync → ensures reproducible build.
- `--mount=type=cache,target=/root/.cache/uv` speeds up rebuilds.

→ The builder/runtime split keeps the final image lean.

### ‍Worker (./worker/Dockerfile_worker)

Worker Dockerfile - highlight key points:

- Uses the **same builder stage** (shared dependencies & lock file).
- Runs the Celery worker explicitly:
    ```dockerfile
    CMD ["celery", "-A", "celery_tasks.celery_app:celery_app", "worker", \
        "--loglevel=INFO", "--concurrency=2", \
        "-Q", "import,celery"]
    ```  
- Supports both `import` and `celery` queues (you can extend this).
- Alternative debug mode (commented out): `--loglevel=DEBUG -P solo` (useful for local troubleshooting).

---------------------------------------------------------------------------

## Step 4: Orchestrate with Docker Compose
Let's take a look at the `docker-compose.yaml` - here’s a concise summary:
```yaml
version: "3.9"

services:
  redis:
    image: redis:8.4.2-trixie
    ports:
      - "6379:6379"

  import-api-producer:
    build:
      context: .
      dockerfile: api_gateway/Dockerfile_import_producer
    environment:                                       # "localhost" for remote testing
      CELERY_BROKER_URL: redis://redis:6379/0          # redis://redis:6379/0 | redis://localhost:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
      PYTHONPATH: /app
      FASTAPI_ENV: dev
      API_PORT: 5001
    volumes:
      - ./api_gateway:/app/api_gateway
      # ... (other local mounts for hot reload - useful during developing/debugging)
    depends_on:
      - redis
    # network_mode: "host"   # Enables "localhost" resolution in remote/cloud environments

  import-worker:
    build:
      context: .
      dockerfile: worker/Dockerfile_worker
    environment:
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
      PYTHONPATH: /app
    depends_on:
      - redis
    # network_mode: "host"   # Same reason - for remote testing
```

### Why `network_mode: "host"` + `localhost` ?
- In `non-enterprise networks` (e.g. remote dev machines, cloud VMs), Docker’s default bridge networking breaks service discovery (e.g. `redis` hostname not resolvable).
- `network_mode: "host"` allows services to communicate via `localhost`, bypassing container DNS limitations.
- For local dev on a *single machine*, this works seamlessly.

→ When deploying to Kubernetes or managed infra, or when you are **on premise**, revert to DNS-based names (`redis:6379`) and omit `network_mode`.

### How Celery works here
1. **Producer** (FastAPI app) calls `my_task.delay(...)` → Celery serializes & enqueues task in Redis.
2. Redis holds the task queue (scalable, persistent if configured).
3. `Worker(s)` poll Redis → pick tasks → execute in isolation (e.g. document assembly, LLM calls).
4. Results stored in Redis backend (`celery_task_results` hash set) or external DB (e.g. ES).

→ This decouples request handling from heavy computation - improving throughput and fault tolerance.

### Launching the stack
#### Full stack (build + run)
```bash
# Build all + start
docker compose up --build

# Stop & clean volumes
docker compose down -v --remove-orphans

# Force rebuild (no cache)
docker compose build --no-cache
docker compose up
```

### Service-specific commands
```bash
# Build only the worker - `-d` for "--detached mode"
docker compose build --no-cache import-worker -d

# Re-create and run (e.g. after config change)
docker compose up --force-recreate

# Scale workers/consumers (2 instances)
docker compose up --build --scale import-worker=2 -d

# Monitor
docker compose ps
docker compose logs -f import-worker
```

### Quick Test

After launching:

1. Hit your API: `POST /import` (example).
2. Watch worker logs:
```bash
docker compose logs -f import-worker
```     

→ You should see:<br>
`INFO/MainProcess] Received task: celery_tasks.importation.import_documents_task [...]`

### Summary checklist
| Step | Done ? |
|------|-------|
| `celery>=5.6.2`, `redis>=7.3.0`, `chonkie==1.4.0` in `pyproject.toml` | v |
| `uv lock && uv sync` executed | v |
| `Dockerfile_producer` includes `uv sync` + dependency layers | v |
| `Dockerfile_worker` runs Celery with correct app & queues | v |
| `docker-compose.yaml` sets `network_mode: "host"` for remote testing | v |
| Verified: Producer → Redis → Worker pipeline works | v |

---------------------------------------------------------------------------

## Additional explanations

### `--pool=solo` vs `--concurrency` at the end of the Worker Dockerfile

-  `--pool=solo` = 1 process, 1 thread → **effective concurrency ~ 1** (even if `--concurrency=N` is set).
- To run **2 tasks in parallel inside a single container** (not recommended), you have two clean options:
  - Remove `--pool=solo` (Linux-only) and use `--concurrency=2`  
    → spawns 2 worker processes (multiprocessing pool).
  - Keep `--pool=solo`, but scale via Docker Compose (see below)
    → 2 *separate* containers, each with concurrency=1.

### Have multiple workers to aborsb the concurrent workload (best practice)
→ For POC/demo, **scaling via `--scale` is preferred**:
  easier to manage, reproducible, and mirrors production (e.g. Kubernetes `replicas`).

---------------------------------------------------------------------------

## Launching Multiple workers

### Recommended: Use `--scale` in `docker compose`
In `docker-compose.yaml`, define **one** worker service (as before), then launch:

```bash
docker compose up --build --scale import-worker=2
```

→ This runs **2 independent containers**, each executing the same Celery command from the Dockerfile.<br>
They both connect to the same Redis queue (`import,celery`) and share the workload automatically.

→ `deploy.replicas: 2` is **ignored** by `docker compose` (classic), only active in `Docker Swarm` (see right below).
```yaml
services:
  import-worker:
    build: # ./worker
      context: .
      dockerfile: worker/Dockerfile_worker
    # ...
    # ...
    deploy:
      replicas: 2
```

### Alternative: Multiple worker services (not recommended)

You *could* define two services in `docker-compose.yaml`, each with its own name and identical config:
```yaml
import-worker-1:
  build:
    context: .
    dockerfile: worker/Dockerfile_worker
  environment:
    CELERY_BROKER_URL: redis://redis:6379/0
    CELERY_RESULT_BACKEND: redis://redis:6379/1
  command: ["celery", "-A", "celery_tasks.celery_app:celery_app", "worker",
            "--loglevel=INFO", "--concurrency=2",
            "-Q", "import,celery"]
  depends_on:
    - redis

import-worker-2:
  # identical to above
```

→ Downside: duplicated config, harder to maintain, less flexible (no `--scale`).

### Not recommended: Multiple Celery workers inside one container

You *could* launch multiple Celery processes in a single container via:
```docker
CMD ["sh", "-c", " \
    celery -A celery_tasks.celery_app:celery_app worker --loglevel=INFO --pool=solo -Q import,celery -n worker1@%h & \
    celery -A celery_tasks.celery_app:celery_app worker --loglevel=INFO --pool=solo -Q import,celery -n worker2@%h & \
    wait \
"]
```

→ Why avoid this ?
- Shared resources (memory, CPU) → harder to isolate failures.
- Restart one worker → restarts all.
- No orchestration per worker (scaling, monitoring).
- Violates the "**one process per container**" best practice.

---------------------------------------------------------------------------

## About `CELERY_RESULT_BACKEND` (optional but useful)

- **Broker (e.g. Redis db0)** → transports messages (jobs) to workers.
- **Result Backend (e.g. Redis db1)** → stores state and results (`SUCCESS`, `FAILURE`, return value, tracebacks).

### When do you need a result backend ?
- You need to query task status/results via `AsyncResult(task_id).state/result`, etc.
- You want to track retries, timestamps, or exceptions.

### When can you skip it ?
- If tasks are executed and forget (e.g. "trigger document assembly and forget").
- If task outcomes are **persisted elsewhere** (e.g. Elasticsearch, DB).
- If your workflow **doesn’t need async result polling** (e.g. HTTP callback or webhook instead).

### Example:
In your producer, if you only do `my_task.delay(...)` and don’t store or inspect the `AsyncResult`, you may omit `CELERY_RESULT_BACKEND`.<br>
But for **debugging or orchestration** (e.g. "is this document done ?"), Redis db1 is handy.

### Summary:
- Use `--scale` for multi-worker containers.
- Use `--concurrency=N` (no `solo`) for parallelism *within* a container (Linux), but prefer **one process per container** (best practice).
- Keep Dockerfile simple - don’t nest Celery processes.
- `CELERY_RESULT_BACKEND` is optional unless you inspect task outcomes.

------------------------

### Author

AAllie
