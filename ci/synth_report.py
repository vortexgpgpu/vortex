#!/usr/bin/env python3
"""synth_report — collapse a fan-out of synthesis-gate reports into one summary.

.github/workflows/asic_gate.yml runs one standalone job per DUT, so the run's
verdict is scattered across N `--report` JSONs. This joins them into a single
Markdown table for the workflow summary, and exits with the worst outcome so the
caller can decide whether the commit was actually gated:

  0  every build passed (or is a tracked known issue)
  1  at least one build regressed
  2  at least one build never produced metrics -- the commit is NOT gated

  usage: synth_report.py <dir-of-report-json> [...]
"""

import glob
import json
import os
import sys

# Same contract as synth_gate.PASSING: a tracked known issue and an unexpected
# pass are reported, not failed.
PASSING = ("PASS", "RECORDED", "KNOWN-ISSUE", "XPASS")

# Shown in this order when a build reports them; the rest of the metrics stay in
# the JSON artifacts, which is where a real investigation goes anyway.
COLUMNS = (("Fmax (MHz)", "fmax_mhz", "%.1f"),
           ("Area (um^2)", "cell_area_um2", "%.0f"),
           ("SRAM (um^2)", "sram_area_um2", "%.0f"),
           ("Cells", "cell_count", "%d"),
           ("LUT", "lut", "%d"),
           ("FF", "ff", "%d"))


def load(paths):
    """Every build across every report, newest-wins on a duplicate id."""
    builds, env, tool = {}, {}, ""
    for path in paths:
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            print("WARNING: %s: %s" % (path, e), file=sys.stderr)
            continue
        env = doc.get("env") or env
        tool = doc.get("tool") or tool
        for b in doc.get("builds", []):
            builds[b["id"]] = b
    return builds, env, tool


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit(__doc__.strip().splitlines()[-1].strip())
    paths = []
    for arg in argv:
        paths += sorted(glob.glob(os.path.join(arg, "**", "*.json"),
                                  recursive=True)) if os.path.isdir(arg) else [arg]
    builds, env, tool = load(paths)
    if not builds:
        print("## synthesis gate\n\nNo reports found — every build errored "
              "before it could write one.")
        return 2

    # Only the columns some build actually measured: an FPGA run has no cell
    # area and an ASIC run has no LUTs, and empty columns are noise.
    cols = [c for c in COLUMNS
            if any((b.get("metrics") or {}).get(c[1]) is not None
                   for b in builds.values())]

    print("## %s gate\n" % (tool or "synthesis"))
    if env:
        print("`%s`\n" % "  ".join("%s=%s" % kv for kv in sorted(env.items())))
    print("| build | dut | " + " | ".join(c[0] for c in cols)
          + " | time | verdict |")
    print("|---|---|" + "---|" * (len(cols) + 2))

    worst = 0
    for bid in sorted(builds):
        b = builds[bid]
        m = b.get("metrics") or {}
        cells = []
        for _, key, fmt in cols:
            v = m.get(key)
            cells.append("—" if v is None else fmt % v)
        secs = m.get("build_time_s")
        verdict = b.get("verdict") or "?"
        note = " — %s" % b["known_issue"] if b.get("known_issue") else ""
        print("| %s | %s | %s | %s | %s%s |"
              % (bid, b.get("dut", ""), " | ".join(cells),
                 "—" if secs is None else "%dm" % (secs // 60), verdict, note))
        if verdict not in PASSING:
            worst = max(worst, 2 if verdict == "BUILD-FAIL" else 1)

    if worst:
        print("\n%d of %d builds did not pass."
              % (sum(1 for b in builds.values()
                     if (b.get("verdict") or "?") not in PASSING), len(builds)))
    return worst


if __name__ == "__main__":
    sys.exit(main())
