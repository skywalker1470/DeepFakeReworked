#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   REGISTRY      e.g. your Docker Hub username
#   IMAGE_NAME    e.g. deepfake-detector
#   VM_HOST       e.g. ubuntu@1.2.3.4 (the EC2 instance)
#   VM_KEY        path to the SSH private key for VM_HOST
#
# Optional:
#   IMAGE_TAG     defaults to "latest"

: "${REGISTRY:?set REGISTRY}"
: "${IMAGE_NAME:?set IMAGE_NAME}"
: "${VM_HOST:?set VM_HOST}"
: "${VM_KEY:?set VM_KEY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "== 1/5 Building image ${FULL_IMAGE} =="
docker build -t "${FULL_IMAGE}" .

echo "== 2/5 Pushing image =="
docker push "${FULL_IMAGE}"

echo "== 3/5 Pulling latest image on VM =="
ssh -i "${VM_KEY}" "${VM_HOST}" "docker pull ${FULL_IMAGE}"

echo "== 4/5 Restarting container on VM =="
ssh -i "${VM_KEY}" "${VM_HOST}" bash -s <<EOF
docker rm -f deepfake-detector 2>/dev/null || true
docker run -d --name deepfake-detector -p 8000:8000 --restart unless-stopped ${FULL_IMAGE}
EOF

echo "== 5/5 Smoke testing /health =="
sleep 3
ssh -i "${VM_KEY}" "${VM_HOST}" "curl -sf http://localhost:8000/health" && echo "Deployment OK" || {
  echo "Smoke test failed" >&2
  exit 1
}
