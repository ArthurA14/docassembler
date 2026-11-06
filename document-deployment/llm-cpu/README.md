# llm-cpu

This folder contains the CPU-only LLM deployment assets for the `document-assembly` namespace.

The goal is to test the Report stack without calling the external vLLM endpoint at `http://vllm.<CLUSTER_IP>.nip.io/v1`, while keeping the existing Python images unchanged. The application keeps using its OpenAI-compatible wrapper, and only the Helm configuration changes:

```yaml
model: "cpu-llm"
base_url: "http://llm-cpu.document-assembly.svc.cluster.local:8080/v1"
```

The `llm-cpu` service runs LocalAI in CPU mode. The model is exposed through the logical alias `cpu-llm`; the GGUF file expected by the provided LocalAI config is `cpu-llm.gguf`.

## Files

- `00-pvc.yaml`: persistent volume claim mounted at `/models` by the LocalAI pod.
<!-- - `01-configmap.yaml`: runtime values consumed by `02-deployment.yaml` for `MODELS_PATH`, `THREADS`, and `CONTEXT_SIZE`. -->
- `02-deployment.yaml`: LocalAI CPU deployment with the `seed-models` init container.
- `03-service.yaml`: internal `ClusterIP` service exposing LocalAI on port `8080`.
- `values-cpu-llm.yaml`: Helm override for the Report chart to point LLM, embedder, and reranker settings to the CPU path.
- `Dockerfile`: model-seed image used by the init container.
- `seed-models/cpu-llm.yaml`: LocalAI model configuration for the `cpu-llm` alias.

## Model Seed Image

The init container expects a private image containing the LocalAI model files under `/seed-models`.

Expected local build context:

```text
llm-cpu/
├── Dockerfile
└── seed-models/
    ├── cpu-llm.yaml
    ├── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    ├── jina-embeddings-v2-base-en-Q4_K_M.gguf
    └── bge-reranker-v2-m3-Q4_K_M.gguf
```

For the first CPU test, use `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` as the source model and place or rename it as:

```text
llm-cpu/seed-models/cpu-llm.gguf
```

Build and push the model-seed image:

```bash
cd llm-cpu
docker build -t nexus.docker.registry/document-assembly-gateway/llm-cpu-models:v1 .
docker push nexus.docker.registry/document-assembly-gateway/llm-cpu-models:v1
```

## Deploy llm-cpu

Because this folder also contains Helm values and LocalAI model YAML files, do not apply the whole folder with `kubectl apply -f llm-cpu/`.

Apply only the Kubernetes manifests:

```bash
kubectl apply -f llm-cpu/00-pvc.yaml
# kubectl apply -f llm-cpu/01-configmap.yaml
kubectl apply -f llm-cpu/02-deployment.yaml
kubectl apply -f llm-cpu/03-service.yaml
kubectl rollout status deploy/llm-cpu -n document-assembly
```

The `seed-models` init container copies `/seed-models/.` from `nexus.docker.registry/document-assembly-gateway/llm-cpu-models:v1` into the `/models` PVC mount, then lists `/models` in the pod logs. The LocalAI container starts afterwards and reads the model files from the same PVC.

The deployment uses `imagePullSecrets: nexusdns` so Kubernetes can pull the private model-seed image from Nexus.

## Deploy the Report Chart Override

After `llm-cpu` is running, upgrade the Report Helm release with the CPU override:

```bash
cd helm-deployment
helm upgrade --install document-assembly ./mychart \
  -n document-assembly \
  -f ./mychart/values.yaml \
  -f ../llm-cpu/values-cpu-llm.yaml
```

The override changes:

- `llm-config.yaml`: points text generation to `llm-cpu`.
- `embedder-settings.yaml`: points embedding configuration to the CPU service path if supported.
- `reranker-settings.yaml`: disables document and citation reranking to avoid external vLLM `/rerank` calls.
- component resources: raises CPU and memory requests for the CPU test path.

## Verify

Check that the model files were copied:

```bash
kubectl exec -n document-assembly deploy/llm-cpu -- ls -lh /models
kubectl exec -n document-assembly deploy/llm-cpu -- cat /models/cpu-llm.yaml
```

Check the LocalAI API from inside the namespace:

```bash
kubectl run curl-test \
  -n document-assembly \
  --rm -it \
  --image=curlimages/curl \
  --restart=Never \
  -- sh
```

Inside the temporary pod:

```bash
curl http://llm-cpu.document-assembly.svc.cluster.local:8080/v1/models

curl http://llm-cpu.document-assembly.svc.cluster.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cpu-llm",
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
