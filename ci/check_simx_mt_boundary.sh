#!/bin/bash
# CI guard for the SimX threading-boundary rule.
#
# Model code under sim/simx/ communicates exclusively through framework
# primitives (SimChannel, SimEventLink, RegSlice) and never carries
# threading vocabulary of its own: no deferred-call plumbing, no domain
# introspection, no atomics, threads, or locks. The framework (sim/common/)
# is the only place that vocabulary may appear, which is what keeps every
# component correct under both serial and parallel execution.
#
# Allowlist: sim/simx/dtm/ hosts the debug-transport TCP server, which is
# host-side tooling with its own service thread — not a timed component.
# sim/simx/regslice.h is a framework element (the registered boundary
# stage), not model code.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

scan() {
    local label="$1"; shift
    local pattern="$1"; shift
    local hits
    hits=$(grep -rnE "$pattern" "$ROOT/sim/simx" \
             --include='*.cpp' --include='*.h' \
             2>/dev/null | grep -v "^$ROOT/sim/simx/dtm/" \
                         | grep -v "^$ROOT/sim/simx/regslice.h:" || true)
    if [ -n "$hits" ]; then
        echo "ERROR: $label" >&2
        echo >&2
        echo "$hits" >&2
        echo >&2
        fail=1
    fi
}

scan "sim/simx uses deferred-call plumbing (use SimChannel/SimEventLink):" \
    '\bcross_call\b|\bcross_pending\b'

scan "sim/simx reads execution-domain internals:" \
    '->domain\(\)|\bexec_domain\b'

scan "sim/simx declares threading state (atomics/threads/locks):" \
    'std::atomic|std::thread|std::mutex|std::shared_mutex|std::lock_guard|std::unique_lock|std::shared_lock|std::scoped_lock|std::condition_variable'

scan "sim/simx includes threading headers:" \
    '#[[:space:]]*include[[:space:]]*<(atomic|thread|mutex|condition_variable|shared_mutex|future|semaphore|barrier|latch)>'

if [ "$fail" -ne 0 ]; then
    echo "SimX threading-boundary check FAILED" >&2
    exit 1
fi
echo "SimX threading-boundary check passed"
