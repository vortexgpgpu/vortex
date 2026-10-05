#!/bin/bash

# Copyright © 2019-2026
#
# Licensed under the Apache License, Version 2.0 (the "License").

# Prints the toolchain identity the CI caches are keyed on: <TOOLCHAIN_REV>-<sha>.
# TOOLCHAIN_REV is a rolling tag (e.g. v3.0), so the key must carry the commit it
# currently resolves to: keying on the tag name keeps serving a stale cache after
# the tag is force-moved to new prebuilts. Every workflow that names the toolchain
# cache derives its key here, so a lookup and the save it probes cannot diverge.

set -euo pipefail

source "$(dirname "$0")/../VERSION"

# ls-remote output is refname-sorted, so the peeled ^{} line (annotated tag's
# commit) sorts last; a lightweight tag yields the single commit line.
sha=$(git ls-remote https://github.com/vortexgpgpu/vortex-toolchain-prebuilt \
      "refs/tags/$TOOLCHAIN_REV" "refs/tags/$TOOLCHAIN_REV^{}" \
      | tail -1 | cut -f1)
if [ -z "$sha" ]; then
  echo "error: cannot resolve toolchain tag '$TOOLCHAIN_REV'" >&2
  exit 1
fi
echo "$TOOLCHAIN_REV-$sha"
