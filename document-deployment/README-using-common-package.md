# Getting Started - `.netrc` setup for private package access

This project uses a shared private library hosted on GitLab. Docker builds fetch it at build time via `.netrc` authentication (injected as a Docker secret - see `docker-compose.yaml`).

> **Context:** this step is a **build-only** operation. The `.netrc` and the `extra_hosts` resolution (into the `docker-compose.yaml`) are only needed to build the images and embed the shared private library into them. Once the images are built and pushed to Nexus, Kubernetes/Helm pulls them directly - no `.netrc` propagation needed at runtime.
>
> A GitLab CI pipeline (currently being finalized) will automate this build + push step in the future.

## 1. Create your `.netrc` file locally

```bash
cd ~
touch .netrc
chmod 600 .netrc
```

## 2. Add your GitLab credentials

Add the following to `~/.netrc`:
```bash
machine gitlab.internal.instance.com
login mylogin
password <my-personal-access-token>
```

## 3. Generate a Personal Access Token on GitLab

1. Go to **GitLab → top-right avatar → Preferences → Personal Access Tokens**
2. Click **Add new token**
3. Fill in a **Token name** (e.g. `netrc-docker`)
4. Under **Select scopes**, check **all scopes**
5. Click **Create personal access token**
6. **Copy the generated token** (it is shown only once)
7. Paste it as the `password` value in `~/.netrc`

## 4. Build the images

```bash
docker compose -f docker-compose.yaml build
# docker compose -f docker-compose.yaml up --build
```

This builds the three service images (`document-gateway`, `import-worker`, `docassembler-worker`), resolving `gitlab.internal.instance.com` via `extra_hosts` and fetching the shared private library via the `.netrc` secret.

## 5. Push the images to Nexus

```bash
docker push nexus.docker.registry/document-assembly-gateway/document-gateway:v3.1.0
docker push nexus.docker.registry/document-assembly-gateway/import-worker:v3.1.0
docker push nexus.docker.registry/document-assembly-gateway/docassembler-worker:v3.1.0
```

## 6. Deploy via Helm

Once the images are available in Nexus, deploy the full stack into the Kubernetes cluster:

```bash
cd helm-deployment
helm upgrade --install v3.1.0 ./mychart \
  -n doc-assembler-depl \
  -f ./mychart/values.yaml
```

Kubernetes pulls the images from Nexus via the `nexusdns` imagePullSecret — no `.netrc` involved at this stage.
