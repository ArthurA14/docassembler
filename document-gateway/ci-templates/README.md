# document-gateway CI template set

Required GitLab CI/CD variables:
- `NEXUS_USER`
- `NEXUS_PASSWORD`
- `document_DEPLOYMENT_PROJECT` (or edit `ci-templates/variables/variables.yml`)

This pipeline:
1. lints/tests the `document-gateway` repo,
2. builds and pushes `document-gateway` to Nexus,
3. triggers the `document-deployment` pipeline and forwards:
   - `UPSTREAM_APP_NAME=document-gateway`
   - `UPSTREAM_IMAGE_NAME`
   - `UPSTREAM_IMAGE_TAG`

The downstream deployment pipeline is responsible for the component deployment.
