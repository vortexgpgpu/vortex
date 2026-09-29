#!/usr/bin/env python3
"""synth_report — render synthesis-gate reports as one Markdown summary.

Both gates end a run with `--report` JSON: asic_gate.yml one per DUT job,
fpga_gate.yml one for the whole sweep. This joins them into a single Markdown
document for the workflow summary -- every build's metrics against its
baseline, why each build that did not pass did not, and where each design is
tight -- so a run is diagnosed from its GitHub page rather than from the machine
that built it. It exits with the worst outcome so the caller can decide whether
the commit was actually gated:

  0  every build passed (or is a tracked known issue)
  1  at least one build regressed
  2  at least one build never produced metrics -- the commit is NOT gated

  usage: synth_report.py [--note TEXT] [--annotate LEVEL] [--json FILE]
                         [--build-failed]
                         <report.json | dir-of-report-json> [...]
"""

import argparse
import glob
import json
import os
import sys
import traceback

# Same contract as synth_gate.PASSING: a tracked known issue and an unexpected
# pass are reported, not failed.
PASSING = ("PASS", "RECORDED", "KNOWN-ISSUE", "XPASS")

# Every metric either tool records, shown in this order when a build reports it.
COLUMNS = (("Fmax (MHz)", "fmax_mhz", "%.1f"),
           ("WNS (ns)", "wns_ns", "%.3f"),
           ("TNS (ns)", "tns_ns", "%.1f"),
           ("Area (um^2)", "cell_area_um2", "%.0f"),
           ("Seq (um^2)", "seq_area_um2", "%.0f"),
           ("SRAM (um^2)", "sram_area_um2", "%.0f"),
           ("Cells", "cell_count", "%d"),
           ("Power (mW)", "power_mw", "%.1f"),
           ("LUT", "lut", "%d"),
           ("FF", "ff", "%d"),
           ("LUTRAM", "lutram", "%d"),
           ("BRAM", "bram", "%d"),
           ("URAM", "uram", "%d"),
           ("DSP", "dsp", "%d"))

# Slack sits near zero by design, so a relative change of it says nothing.
NO_DELTA = ("wns_ns", "tns_ns")


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
        if not isinstance(doc, dict) or "builds" not in doc:
            continue
        env = doc.get("env") or env
        tool = doc.get("tool") or tool
        for b in doc["builds"]:
            builds[b["id"]] = b
    return builds, env, tool


def reasons(build):
    """Why a build got its verdict; a report without `reasons` still carries
    the build error."""
    found = build.get("reasons")
    if found is None:
        found = [build["error"]] if build.get("error") else []
    return [r for r in found if r]


def cell(build, key, fmt):
    cur = (build.get("metrics") or {}).get(key)
    if cur is None:
        return "—"
    ref = (build.get("baseline") or {}).get(key)
    if not ref or key in NO_DELTA:
        return fmt % cur
    return "%s (%+.1f%%)" % (fmt % cur, 100.0 * (cur - ref) / ref)


def print_table(builds, cols):
    print("| build | dut | target (MHz) | " + " | ".join(c[0] for c in cols)
          + " | time | verdict |")
    print("|---|---|---|" + "---|" * (len(cols) + 2))
    for bid in sorted(builds):
        b = builds[bid]
        secs = (b.get("metrics") or {}).get("build_time_s")
        verdict = b.get("verdict") or "?"
        if verdict not in PASSING:
            verdict = "**%s**" % verdict
        print("| %s | %s | %s | %s | %s | %s |"
              % (bid, b.get("dut", ""), b.get("clock_mhz", "—"),
                 " | ".join(cell(b, c[1], c[2]) for c in cols),
                 "—" if secs is None else "%dm" % (secs // 60), verdict))


def print_findings(builds):
    found = [(bid, builds[bid]) for bid in sorted(builds) if reasons(builds[bid])]
    if not found:
        return
    print("\n### Findings\n")
    for bid, b in found:
        print("**%s** — %s\n" % (bid, b.get("verdict") or "?"))
        for reason in reasons(b):
            if "\n" in reason:
                print("```\n%s\n```" % reason)
            else:
                print("- %s" % reason)
        print()


def print_configs(builds):
    print("\n### Configurations\n")
    print("| build | configs | strategy | config hash | baseline hash |")
    print("|---|---|---|---|---|")
    for bid in sorted(builds):
        b = builds[bid]
        print("| %s | `%s` | %s | `%s` | `%s` |"
              % (bid, b.get("configs") or " ", b.get("impl_strategy") or "—",
                 b.get("config_hash") or "—",
                 b.get("baseline_config_hash") or "—"))


def print_paths(title, result):
    paths = result.get("critical_paths") or []
    if paths:
        print("%s critical paths\n" % title)
        print("| slack (ns) | levels | logic (ns) | route (ns) | route % "
              "| startpoint | endpoint |")
        print("|---|---|---|---|---|---|---|")
        for p in paths:
            print("| %s | %s | %s | %s | %s | `%s` | `%s` |"
                  % tuple(p.get(k, "—") for k in
                          ("slack_ns", "levels", "logic_ns", "route_ns",
                           "route_pct", "startpoint", "endpoint")))
        print()
    nets = result.get("high_fanout_nets") or []
    if nets:
        print("%s high-fanout nets\n" % title)
        print("| fanout | driver | net |")
        print("|---|---|---|")
        for n in nets:
            print("| %s | %s | `%s` |"
                  % (n.get("fanout", "—"), n.get("driver", "—"),
                     n.get("net", "—")))
        print()


def print_diagnostics(builds):
    """Where each design is tight, measured and baseline side by side: a
    regression is read off what moved between the two."""
    shown = [(bid, builds[bid]) for bid in sorted(builds)
             if (builds[bid].get("metrics") or {}).get("critical_paths")]
    if not shown:
        return
    print("\n### Critical paths\n")
    for bid, b in shown:
        base = b.get("baseline") or {}
        print("<details><summary>%s — worst slack %s ns</summary>\n"
              % (bid, b["metrics"]["critical_paths"][0].get("slack_ns", "?")))
        print_paths("Measured", b["metrics"])
        print_paths("Baseline", base)
        if base.get("env"):
            print("Baseline recorded with `%s`\n" % "  ".join(
                "%s=%s" % kv for kv in sorted(base["env"].items())))
        print("</details>\n")


def escape(text, prop=False):
    """Workflow-command encoding; a property also reserves ':' and ','."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def annotate(builds, tool, level):
    """One workflow annotation per reason, so the run page names the build and
    the metric. On stderr: stdout is the Markdown document."""
    for bid in sorted(builds):
        b = builds[bid]
        verdict = b.get("verdict") or "?"
        if verdict in PASSING:
            continue
        title = "%s gate: %s %s" % (tool or "synthesis", bid, verdict)
        for reason in reasons(b) or [verdict]:
            print("::%s title=%s::%s"
                  % (level, escape(title, prop=True), escape(reason)),
                  file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render synthesis-gate reports as one Markdown summary")
    ap.add_argument("reports", nargs="+", metavar="PATH",
                    help="report JSON, or a directory searched for them")
    ap.add_argument("--note", metavar="TEXT",
                    help="a line printed under the heading, e.g. which run "
                         "these results came from")
    ap.add_argument("--annotate", choices=("error", "warning", "notice"),
                    metavar="LEVEL",
                    help="emit a workflow annotation (error, warning or notice) "
                         "per reason a build did not pass")
    ap.add_argument("--json", metavar="FILE",
                    help="write the joined reports as one JSON document")
    ap.add_argument("--build-failed", action="store_true",
                    help="print `<id> <dut>` for each build that never "
                         "produced metrics, one per line, and exit")
    args = ap.parse_args(argv)

    paths = []
    for arg in args.reports:
        paths += sorted(glob.glob(os.path.join(arg, "**", "*.json"),
                                  recursive=True)) if os.path.isdir(arg) else [arg]
    builds, env, tool = load(paths)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"tool": tool, "env": env,
                       "builds": [builds[b] for b in sorted(builds)]},
                      fh, indent=2)
            fh.write("\n")

    if args.build_failed:
        for bid in sorted(builds):
            if builds[bid].get("verdict") == "BUILD-FAIL":
                print(bid, builds[bid].get("dut", ""))
        return 0

    print("## %s gate\n" % (tool or "synthesis"))
    if args.note:
        print("%s\n" % args.note)
    if not builds:
        print("No reports found — every build errored before it could write "
              "one.")
        return 2
    if env:
        print("`%s`\n" % "  ".join("%s=%s" % kv for kv in sorted(env.items())))

    # Only the columns some build actually measured: an FPGA run has no cell
    # area and an ASIC run has no LUTs, and empty columns are noise.
    cols = [c for c in COLUMNS
            if any((b.get("metrics") or {}).get(c[1]) is not None
                   for b in builds.values())]
    print_table(builds, cols)

    failed = [b for b in builds.values()
              if (b.get("verdict") or "?") not in PASSING]
    if failed:
        print("\n%d of %d builds did not pass." % (len(failed), len(builds)))
    print_findings(builds)
    print_configs(builds)
    print_diagnostics(builds)
    if args.annotate:
        annotate(builds, tool, args.annotate)

    if any(b.get("verdict") == "BUILD-FAIL" for b in failed):
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    # A renderer that dies must not read as a regression verdict (exit 1): the
    # caller records the commit as gated on anything but 2.
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
