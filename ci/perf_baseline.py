"""Perf-regression baselines: golden rtlsim cycle counts per benchmark.

Baselines live in the SOURCE tree at ci/baselines/perf/<category>.json (canonical
sorted JSON, one file per category). They are read by ci/test_runner.py's
perf_gate check and (re)written only by `pytest --update-baselines`; CI
never writes them. See docs/designs/continuous_integration.md §3.4.

The harness runs from a build tree, so the source root is resolved from the
build tree's config.mk (VORTEX_HOME) — baselines are source data, not copied
into build/, so an update lands directly on the golden a human then reviews.
"""

import collections
import hashlib
import json
import os
import re

# Perf-regression tolerance: cycles must stay within +/-2% of baseline. The
# upper bound is the regression gate; the lower bound is the ratchet — an
# improvement beyond it must update the baseline so the gain is locked in and a
# later silent regression back toward the old number is still caught.
TOLERANCE = 0.02


def _source_root():
    """Source repo root, from the build tree's config.mk VORTEX_HOME."""
    try:
        with open("config.mk") as fh:
            for line in fh:
                m = re.match(r"\s*VORTEX_HOME\s*[?:]?=\s*(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    # Fallback: two levels up from this file (ci/ -> repo root).
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASELINE_DIR = os.path.join(_source_root(), "ci", "baselines", "perf")


def _path(category):
    return os.path.join(BASELINE_DIR, category + ".json")


def config_hash(case):
    """Fingerprint the run inputs; a change invalidates the stored cycles."""
    key = "|".join([case.app, case.args, case.configs,
                    repr(sorted(case.shape.items()))])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load(category):
    try:
        with open(_path(category)) as fh:
            return json.load(fh)
    except OSError:
        return {}


# --update-baselines accumulates here across the session; conftest flushes once.
_pending = collections.defaultdict(dict)


def record(case, xlen, cycles, instrs):
    entry = _pending[case.category].setdefault(case.id, {
        "app": case.app, "args": case.args, "configs": case.configs,
        "config_hash": config_hash(case),
    })
    entry["app"], entry["args"] = case.app, case.args
    entry["configs"], entry["config_hash"] = case.configs, config_hash(case)
    entry[str(xlen)] = {"cycles": int(cycles), "instrs": int(instrs)}


def flush():
    """Merge pending records into the per-category files (preserving other xlens)."""
    for category, entries in _pending.items():
        merged = load(category)
        for cid, entry in entries.items():
            cur = merged.setdefault(cid, {})
            cur.update(entry)  # entry carries this-run's xlen key + metadata
        os.makedirs(BASELINE_DIR, exist_ok=True)
        with open(_path(category), "w") as fh:
            json.dump(merged, fh, indent=2, sort_keys=True)
            fh.write("\n")
    _pending.clear()
