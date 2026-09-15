#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
flp_source=${OMNIRT_FLP_SOURCE_ROOT:-/home/ubuntu/digital-human/fasterliveportrait/FasterLivePortrait}
qt_image=${OMNIRT_QUICKTALK_BASE_IMAGE:-omnirt-quicktalk:cu124}
flp_image=${OMNIRT_FLP_BASE_IMAGE:-shaoguo/faster_liveportrait:v3}

test -d "$flp_source/src"
test -d "$flp_source/configs"
test -z "$(git -C "$flp_source" status --porcelain)"
flp_revision=$(git -C "$flp_source" rev-parse HEAD)
expected_flp_revision=${OMNIRT_EXPECTED_FLP_REVISION:-8aad3602177547aaa5e4beec0c3ef5b7944e7a1f}
if [[ "$flp_revision" != "$expected_flp_revision" ]]; then
  echo "FasterLivePortrait source revision differs from the validated version: $flp_revision" >&2
  exit 1
fi

omnirt_revision=$(git -C "$repo_root" rev-parse HEAD)
qt_image_id=$(docker image inspect "$qt_image" --format '{{.Id}}')
flp_image_id=$(docker image inspect "$flp_image" --format '{{.Id}}')
expected_qt_id=${OMNIRT_EXPECTED_QUICKTALK_BASE_IMAGE_ID:-sha256:8766c1c32ad968b81c761be756ec445f8de4266adf126d4b6335e7fbdac9676b}
expected_flp_id=${OMNIRT_EXPECTED_FLP_BASE_IMAGE_ID:-sha256:a619dd7c30cf24bcc7b1d1266697b7a1332d911af2bdc89a9b3e976496a49457}
if [[ "$qt_image_id" != "$expected_qt_id" || "$flp_image_id" != "$expected_flp_id" ]]; then
  echo "A base image differs from the validated RTX 4090 image IDs." >&2
  echo "QuickTalk: $qt_image_id; FasterLivePortrait: $flp_image_id" >&2
  exit 1
fi

image_tag=${OMNIRT_AVATAR_IMAGE_TAG:-omnirt-avatar-ws:${omnirt_revision:0:12}}
echo "OmniRT=$omnirt_revision FasterLivePortrait=$flp_revision" >&2
echo "QuickTalk image=$qt_image_id FasterLivePortrait image=$flp_image_id" >&2

docker build --network none --progress=plain \
  --build-context "flp_src=$flp_source" \
  --build-arg "QUICKTALK_BASE_IMAGE=$qt_image" \
  --build-arg "FASTERLIVEPORTRAIT_BASE_IMAGE=$flp_image" \
  --build-arg "OMNIRT_REVISION=$omnirt_revision" \
  --build-arg "FASTERLIVEPORTRAIT_REVISION=$flp_revision" \
  --build-arg "QUICKTALK_BASE_IMAGE_ID=$qt_image_id" \
  --build-arg "FASTERLIVEPORTRAIT_BASE_IMAGE_ID=$flp_image_id" \
  -f "$repo_root/deploy/avatar-ws/Dockerfile" \
  -t "$image_tag" \
  "$repo_root"

echo "Built $image_tag" >&2
