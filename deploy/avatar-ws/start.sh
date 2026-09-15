#!/usr/bin/env sh
set -eu

# QuickTalk writes face/template caches next to its model root. Keep that
# root on the shared writable mount while the actual checkpoints stay read-only.
model_root=/tmp/opentalking/quicktalk-model
checkpoint_link="$model_root/checkpoints"
checkpoint_dir=/models/quicktalk/checkpoints

test -f "$checkpoint_dir/quicktalk.pth"
mkdir -p "$model_root" /tmp/opentalking/quicktalk-cache /tmp/opentalking/omnirt-work
if test -L "$checkpoint_link"; then
  test "$(readlink "$checkpoint_link")" = "$checkpoint_dir"
elif test -e "$checkpoint_link"; then
  echo "QuickTalk checkpoint path already exists and is not the expected link: $checkpoint_link" >&2
  exit 1
else
  ln -s "$checkpoint_dir" "$checkpoint_link"
fi

exec "$@"
