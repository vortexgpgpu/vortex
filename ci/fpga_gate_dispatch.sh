#!/bin/bash
# Nightly dispatcher for the FPGA gate (.github/workflows/fpga_gate.yml), run
# from cron on the host of the self-hosted Vivado runner:
#
#   0 3 * * *  <clone>/ci/fpga_gate_dispatch.sh >> ~/.cache/vortex/fpga_gate_dispatch.log 2>&1
#
# GitHub cannot condition a schedule on the repository, so the workflow has no
# schedule of its own. This dispatches it only when a synthesis input changed
# since the last gated commit (the state file the workflow records), so a quiet
# night creates no run. `gh` must be authenticated with a token that can
# dispatch workflows on the repository.
#
# See docs/designs/continuous_integration.md §4.4.

set -euo pipefail

REPO="${REPO:-vortexgpgpu/vortex}"
CLONE="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$HOME/.cache/vortex/fpga_gate.${REPO//\//_}.sha"
INPUTS='^(hw/|VX_config\.toml$|VX_types\.toml$|VERSION$|ci/(fpga_gate|synth_gate|synth_report)\.py$|ci/testcases/fpga_gate\.yaml$|ci/baselines/synthesis/xilinx/|\.github/workflows/fpga_gate\.yml$)'

log() {
    echo "$(date -u +%FT%TZ) fpga_gate_dispatch: $*"
}

# A run nobody can pick up waits in the queue until GitHub cancels it.
if ! pgrep -x Runner.Listener > /dev/null; then
    log "the Actions runner is not running on $(hostname); nothing dispatched"
    exit 1
fi

# The concurrency group would serialize a second run, but it would still sit
# in the history as a duplicate.
for status in queued in_progress; do
    n=$(gh api "repos/$REPO/actions/workflows/fpga_gate.yml/runs?branch=master&status=$status" --jq '.total_count')
    if [ "$n" != "0" ]; then
        log "a gate run is already $status; nothing dispatched"
        exit 0
    fi
done

git -C "$CLONE" fetch --quiet origin master
TIP=$(git -C "$CLONE" rev-parse FETCH_HEAD)
LAST=$(cat "$STATE" 2>/dev/null || true)

# A missing or unknown last commit has no diff to read, so it gates.
if [ -n "$LAST" ] && git -C "$CLONE" cat-file -e "${LAST}^{commit}" 2>/dev/null; then
    if ! git -C "$CLONE" diff --name-only "$LAST" "$TIP" | grep -qE "$INPUTS"; then
        log "no synthesis input changed since $LAST; nothing dispatched"
        exit 0
    fi
fi

gh api -X POST "repos/$REPO/actions/workflows/fpga_gate.yml/dispatches" -f ref=master
log "dispatched the gate for $TIP (last gated: ${LAST:-none})"
