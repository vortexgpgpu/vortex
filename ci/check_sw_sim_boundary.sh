#!/bin/bash
# CI guard for the sw/ ↔ sim/ + hw/ bidirectional isolation rule.
#
# Vortex's source tree separates the install-facing software stack
# (sw/kernel/, sw/runtime/) from the internal hardware (hw/*) and
# simulator (sim/*) implementations. The isolation is bidirectional:
#
#   - sw/kernel and sw/runtime files MUST NOT include or reference
#     anything in hw/* or sim/* (would bleed internals into the SDK).
#
#   - sim/* and hw/* files MUST NOT include or reference anything in
#     sw/kernel/ or sw/runtime/ (would couple the simulator/RTL to
#     the install-facing surface).
#
# sw/common/ is the shared escape hatch — vortex-internal, never
# installed, accessible from all four layers.
#
# See AGENTS.md §6 and docs/coding_guidelines_cpp.md §8.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

###############################################################################
# 1. #include scans
###############################################################################

scan_includes() {
    local label="$1"; shift
    local pattern="$1"; shift
    local hits
    hits=$(grep -rnE "$pattern" "$@" \
             --include='*.c' --include='*.cpp' --include='*.cc' \
             --include='*.h' --include='*.hpp' --include='*.sv' \
             2>/dev/null || true)
    if [ -n "$hits" ]; then
        echo "ERROR: $label" >&2
        echo >&2
        echo "$hits" >&2
        echo >&2
        fail=1
    fi
}

# sw/kernel + sw/runtime must not include from hw/ or sim/.
scan_includes \
    "sw/kernel or sw/runtime references hw/ or sim/ headers:" \
    '#[[:space:]]*include[[:space:]]*[<"]([./]*(hw|sim)/[^">]+)[>"]' \
    "$ROOT/sw/kernel" "$ROOT/sw/runtime"

# sim + hw must not include from sw/kernel or sw/runtime.
# (sw/common is allowed via -Isw/common; sibling files within sw/common
# are themselves not the install-facing layer.)
SW_PUBLIC_HEADERS=$(cd "$ROOT/sw/kernel/include" 2>/dev/null && ls *.h 2>/dev/null | tr '\n' '|' | sed 's/|$//')$(cd "$ROOT/sw/runtime/include" 2>/dev/null && echo -n "|" && ls *.h 2>/dev/null | tr '\n' '|' | sed 's/|$//')

if [ -n "$SW_PUBLIC_HEADERS" ]; then
    scan_includes \
        "sim/ or hw/ references sw/kernel/include or sw/runtime/include headers:" \
        "#[[:space:]]*include[[:space:]]*[<\"]($SW_PUBLIC_HEADERS|sw/(kernel|runtime)/[^\">]+)[>\"]" \
        "$ROOT/sim" "$ROOT/hw"
fi

###############################################################################
# 2. Build-flag scans — Makefiles must not add cross-layer -I paths
###############################################################################

# Build-flag scans use `find` + `grep -l`-style filtering since some
# grep builds handle --include/--exclude inconsistently against
# arbitrary file extensions. Only true Makefile / .mk / .in files are
# scanned; stamp files (configure artifacts) are filtered explicitly.

scan_makefile_flags() {
    local label="$1"; shift
    local pattern="$1"; shift
    local hits=""
    local f
    while IFS= read -r f; do
        local m
        m=$(grep -nE -- "$pattern" "$f" 2>/dev/null || true)
        if [ -n "$m" ]; then
            hits+="${f}:${m}"$'\n'
        fi
    done < <(find "$@" \
              \( -name 'Makefile' -o -name '*.mk' -o -name '*.in' \) \
              -not -name '*.stamp' \
              -type f 2>/dev/null)
    if [ -n "$hits" ]; then
        echo "ERROR: $label" >&2
        echo >&2
        printf '%s' "$hits" >&2
        echo >&2
        fail=1
    fi
}

# sim/* and hw/* Makefiles must not -I into sw/kernel or sw/runtime.
scan_makefile_flags \
    "sim/ or hw/ Makefile adds -Isw/{kernel,runtime}/include:" \
    '-I[^[:space:]]*sw/(kernel|runtime)/include' \
    "$ROOT/sim" "$ROOT/hw"

# sw/kernel and sw/runtime Makefiles must not -I into hw/ or sim/.
# Exception: sw/runtime/opae links against the FPGA AFU shell defined
# in hw/syn/altera/opae/ — a HW-bound integration that can't be
# relocated. Skipped via path exclusion below.
SW_BUILD_DIRS=()
for d in "$ROOT/sw/kernel" "$ROOT/sw/runtime"; do
    [ -d "$d" ] && SW_BUILD_DIRS+=("$d")
done

# Build the exclusion-aware list of files to scan.
sw_files=$(find "${SW_BUILD_DIRS[@]}" \
            \( -name 'Makefile' -o -name '*.mk' -o -name '*.in' \) \
            -not -name '*.stamp' \
            -not -path '*/sw/runtime/opae/*' \
            -type f 2>/dev/null)
if [ -n "$sw_files" ]; then
    hits=""
    while IFS= read -r f; do
        m=$(grep -nE -- '-I[^[:space:]]*(hw|sim)/' "$f" 2>/dev/null || true)
        if [ -n "$m" ]; then
            hits+="${f}:${m}"$'\n'
        fi
    done <<< "$sw_files"
    if [ -n "$hits" ]; then
        echo "ERROR: sw/kernel or sw/runtime Makefile adds -Ihw/ or -Isim/:" >&2
        echo >&2
        printf '%s' "$hits" >&2
        echo >&2
        fail=1
    fi
fi

###############################################################################

if [ "$fail" -ne 0 ]; then
    echo "sw/ ↔ sim/+hw/ boundary check FAILED — see violations above." >&2
    echo "See AGENTS.md §6 and docs/coding_guidelines_cpp.md §8." >&2
    exit 1
fi

echo "sw/ ↔ sim/+hw/ boundary check OK"
