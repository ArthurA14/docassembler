# mock-llm

This folder contains the mock LLM deployment assets for the `doc-assembler-depl` namespace.

The goal is to test the document stack without calling the external vLLM endpoint at `http://vllm.<CLUSTER_IP>.nip.io/v1`, while keeping the existing Python images unchanged. The application keeps using its OpenAI-compatible wrapper, and only the Helm configuration changes:

```yaml
model: "mock-llm"
base_url: "http://mock-llm.doc-assembler-depl.svc.cluster.local:8000/v1"
```

The `mock-llm` service runs a lightweight FastAPI application that exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`) and returns deterministic mock responses — no GPU, no model file required.

## Files

- `mock_llm.py`: FastAPI application exposing mock LLM, embedder, and reranker endpoints.
- `deployment.yaml`: Kubernetes Deployment for the mock LLM service.
- `service.yaml`: internal `ClusterIP` service exposing the mock on port `8000`.
- `values-mock-llm.yaml`: Helm override for the document chart to point LLM, embedder, and reranker settings to the mock service.

## Mock LLM Image

The mock service is packaged as a Docker image and pushed to the private Nexus registry.

Expected local build context:

```text
mock-llm/mock-llm-app/
├── Dockerfile
└── mock_llm.py
```

Build and push the mock image:

```bash
cd mock-llm/mock-llm-app/
docker build -t nexus.docker.registry/document-assembly-gateway/mock-llm:v1 .
docker push nexus.docker.registry/document-assembly-gateway/mock-llm:v1
```

## Deploy mock-llm

Apply only the Kubernetes manifests:

```bash
kubectl apply -f mock-llm/deployment.yaml
kubectl apply -f mock-llm/service.yaml
kubectl rollout status deploy/mock-llm -n doc-assembler-depl
```

The deployment uses `imagePullSecrets: nexusdns` so Kubernetes can pull the mock image from Nexus.

## Deploy the document Chart Override

After `mock-llm` is running, upgrade the document Helm release with the mock override:

```bash
cd helm-deployment
helm upgrade --install v3.1.0 ./mychart \
  -n doc-assembler-depl \
  -f ./mychart/values.yaml \
  -f ../mock-llm/mock-llm-k8s/values-mock-llm.yaml
```

The override changes:

- `llm-config.yaml`: points text assembly to `mock-llm`.
- `embedder-settings.yaml`: points embedding configuration to the mock service path.
- `reranker-settings.yaml`: disables document and citation reranking, pointing to the mock `/v1/rerank` endpoint.
- component resources: raises CPU and memory requests for the mock test path.

## Verify

Check that the mock service is reachable from inside the namespace:

```bash
kubectl run curl-test \
  -n doc-assembler-depl \
  --rm -it \
  --image=curlimages/curl \
  --restart=Never \
  -- sh
```

Inside the temporary pod:

```bash
curl http://mock-llm.doc-assembler-depl.svc.cluster.local:8000/v1/models

curl http://mock-llm.doc-assembler-depl.svc.cluster.local:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-llm",
    "messages": [
      {
        "role": "user",
        "content": "Answer only: OK"
      }
    ],
    "temperature": 0.2,
    "max_tokens": 32
  }'
```

Expected response: a deterministic JSON payload matching the mock logic defined in `mock_llm.py`.
