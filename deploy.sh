#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   REGISTRY      e.g. your Docker Hub username, or an ECR repo URL
#   IMAGE_NAME    e.g. deepfake-detector
#   EC2_HOST      e.g. ec2-user@1.2.3.4
#   EC2_KEY       path to the SSH private key for EC2_HOST
#
# Optional:
#   IMAGE_TAG     defaults to "latest"

: "${REGISTRY:?set REGISTRY}"
: "${IMAGE_NAME:?set IMAGE_NAME}"
: "${EC2_HOST:?set EC2_HOST}"
: "${EC2_KEY:?set EC2_KEY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "== 1/5 Building image ${FULL_IMAGE} =="
docker build -t "${FULL_IMAGE}" .

echo "== 2/5 Pushing image =="
docker push "${FULL_IMAGE}"

echo "== 3/5 Pulling latest image on EC2 =="
ssh -i "${EC2_KEY}" "${EC2_HOST}" "docker pull ${FULL_IMAGE}"

echo "== 4/5 Restarting container on EC2 =="
ssh -i "${EC2_KEY}" "${EC2_HOST}" bash -s <<EOF
docker rm -f deepfake-detector 2>/dev/null || true
docker run -d --name deepfake-detector -p 8000:8000 --restart unless-stopped ${FULL_IMAGE}
EOF

echo "== 5/5 Smoke testing /health =="
sleep 3
ssh -i "${EC2_KEY}" "${EC2_HOST}" "curl -sf http://localhost:8000/health" && echo "Deployment OK" || {
  echo "Smoke test failed" >&2
  exit 1
}
