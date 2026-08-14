#!/usr/bin/env bash
# Run inside the bake pod after cloning this repo to /work.
set -euxo pipefail
cd /work
curl -sL https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C /usr/local/bin crane
mkdir -p /opt/kroot /kaniko/.docker
crane export gcr.io/kaniko-project/executor:v1.23.2 | tar -C /opt/kroot -xf -
AUTH=$(printf 'avocahdoe:%s' "$GITHUB_TOKEN" | base64 -w0)
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}\n' "$AUTH" > /kaniko/.docker/config.json
export DOCKER_CONFIG=/kaniko/.docker
/opt/kroot/kaniko/executor \
  --force \
  --dockerfile=/work/Dockerfile \
  --context=/work \
  --destination=ghcr.io/avocahdoe/ocr-worker:1.0 \
  --compressed-caching=false \
  --cache=false \
  --cleanup \
  --verbosity=info
echo BAKE_PUSH_DONE
