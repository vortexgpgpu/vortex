#!/usr/bin/env python3
"""fpga_gate — FPGA synthesis-regression gate (Xilinx/Vivado DUT builds).

Same shape as the perf_gate (ci/perf_baseline.py): a checked-in golden baseline
per build, a hard threshold around it, and baselines that only a human updates.
The assertion differs: instead of rtlsim cycles it compares THIS commit's
post-implementation Fmax and LUT count against the recorded numbers, so an RTL
change that quietly costs timing closure or area cannot land green.

  Spec      : ci/testcases/fpga_gate.yaml                 (what to build)
  Baselines : ci/baselines/synthesis/xilinx/<group>.json  (last accepted metrics)
  Builds    : hw/syn/xilinx/dut/<dut>/<PREFIX>            (Vivado, out-of-tree)
  Metrics   : <build>/synth_summary.csv, timing.rpt, high_fanout_nets.rpt
              (all three already written by hw/syn/xilinx/dut/project.tcl)

Same division as every other check in the catalog: the yaml is hand-authored,
commented and reviewed; the baseline is machine-written and never hand-edited.
A `config_hash` (dut/clock/configs/device/opt-level/xlen) ties the two together,
so metrics recorded under one config can never be compared against another.

Usage (from a source checkout, with the Xilinx env sourced):
  source ~/dev/xilinx_setup.sh
  ci/fpga_gate.py --list
  ci/fpga_gate.py                       # all builds, gate against baselines
  ci/fpga_gate.py -b tcu -b rtu -j 2    # explicit builds, 2 in parallel
  ci/fpga_gate.py -b graphics           # a whole group
  ci/fpga_gate.py --update-baseline     # re-record (human-reviewed, never in CI)

See docs/designs/continuous_integration.md 3.5.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

# Fmax/LUT must stay within +/-5% of baseline. The regression side is the gate;
# the improvement side is the ratchet -- a win beyond it must be recorded so the
# gain is locked in and a later silent slide back to the old number still fails.
#
# This is the GLOBAL default. It is overridden per-metric on the command line
# (--metric-threshold lut=0.10) and per-build in the catalog ("thresholds":
# {"fmax_mhz": 0.03}) -- most specific wins. Same shape as model_parity's
# `tolerance` (per-case entry > category defaults > DEFAULT_PARITY_TOLERANCE).
TOLERANCE = 0.05

# Metrics gated by default; the rest are recorded and reported, not asserted.
GATED_METRICS = ("fmax_mhz", "lut")

# Higher-is-better metrics. Everything else is lower-is-better.
HIGHER_IS_BETTER = ("fmax_mhz",)

# Verdicts that do not fail the run. KNOWN-ISSUE is a tracked expected failure
# (see gate()); XPASS is one that stopped reproducing -- reported loudly, but a
# gate that hard-failed on an unexpected PASS would be the wrong incentive.
PASSING = ("PASS", "RECORDED", "KNOWN-ISSUE", "XPASS")

# Reported (and recorded) in this order. build_time_s is bookkeeping -- it is how
# long the build took, and it is what the scheduler uses to order the queue.
METRICS = ("fmax_mhz", "wns_ns", "lut", "lutram", "ff", "bram", "uram", "dsp",
           "build_time_s")

# Top critical paths, read from the timing.rpt project.tcl already writes.
# Recorded whether or not timing closed, and never gated: they are the diagnostic
# that says WHERE the design is tight, so a Fmax regression comes with its
# location instead of just its magnitude.
TIMING_RPT = "timing.rpt"
NUM_PATHS = 10

# Top high-fanout nets (high_fanout_nets.rpt). Recorded, never gated -- the
# "why" behind a route-bound critical path.
FANOUT_RPT = "high_fanout_nets.rpt"
NUM_FANOUT = 10

# Scheduling weights (seconds) for a build with no recorded build_time_s yet.
# Only ever used to ORDER the queue, so a rough magnitude is enough.
DEFAULT_WEIGHTS = {"top": 14400, "core": 7200, "tcu": 3600, "rtu": 3600,
                   "vortex": 3600, "cache": 1800, "raster": 1800, "dxa": 1800,
                   "tex": 1200, "om": 1200}
FALLBACK_WEIGHT = 1800

# Vivado jobs per build, clamped so N parallel builds do not oversubscribe.
MIN_VIVADO_JOBS = 2
MAX_VIVADO_JOBS = 8

# Vivado's own log lines around the end of RTL elaboration -- the point past
# which a build can no longer die of a config/RTL typo (see watch()). Vivado
# prints the ELAB_DONE timing line on the way out of elaboration EITHER WAY, so
# a clean pass is ELAB_DONE with no ELAB_FAIL and no error behind it.
ELAB_DONE = "Finished RTL Elaboration"
ELAB_FAIL = "RTL Elaboration failed"

# Vivado log lines that mark a phase transition, in order. Progress is reported
# against these, so a multi-hour build says where it actually is.
PHASES = (
    ("create_project", "setup"),
    ("Starting synth_design", "elaboration"),
    (ELAB_DONE, "synthesis"),
    ("Synth Design complete", "opt"),
    ("Starting Logic Optimization Task", "opt"),
    ("Starting Placer Task", "placement"),
    ("Starting Routing Task", "routing"),
    ("report_timing", "reporting"),
)

POLL_INTERVAL = 5
HEARTBEAT = 300  # seconds between "still alive, here is where it is" lines

# Per-build resume stamp, written into the build dir (not a central session
# file): the state belongs next to the build tree it describes, so it survives a
# kill, a reboot, or a concurrent run touching a different build.
STAMP = "fpga_gate.json"


class BuildError(Exception):
    pass


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def source_root():
    """Source checkout root (has configure + VX_config.toml), not a build tree."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != "/":
        if os.path.exists(os.path.join(d, "configure")) and \
           os.path.exists(os.path.join(d, "VX_config.toml")):
            return d
        d = os.path.dirname(d)
    sys.exit("ERROR: not inside a Vortex source checkout")


def spec_path(root, gate):
    return os.path.join(root, "ci", "testcases", gate + ".yaml")


def baseline_dir(root, tool):
    return os.path.join(root, "ci", "baselines", "synthesis", tool)


def baseline_path(root, tool, group):
    return os.path.join(baseline_dir(root, tool), group + ".json")


def load_catalog(root, gate, tool):
    """Builds keyed by id: the SPEC from ci/testcases/<gate>.yaml, joined to its
    recorded metrics from ci/baselines/synthesis/<tool>/<group>.json.

    Same division as every other check in the catalog: the yaml says what to run
    (and is hand-authored, commented, reviewed), the baseline says what the last
    accepted answer was (and is machine-written). Neither file can be mistaken
    for the other's job.
    """
    with open(spec_path(root, gate)) as fh:
        doc = yaml.safe_load(fh)
    defaults = doc.get("defaults") or {}
    # `builds:`, not `tests:` -- a Vivado build is not a pytest case (no driver,
    # no sim build tree, no xlen expansion, no app). The catalog's `tests:` block
    # holds the ONE case that invokes this script; the build list is ours.
    spec = doc.get("builds")
    if not spec:
        sys.exit("ERROR: no `builds:` block in %s" % spec_path(root, gate))

    baselines = {}
    for group in {e.get("group", "default") for e in spec}:
        path = baseline_path(root, tool, group)
        if os.path.exists(path):
            with open(path) as fh:
                baselines[group] = json.load(fh)

    builds = {}
    for entry in spec:
        bid = entry["id"]
        if bid in builds:
            sys.exit("ERROR: duplicate build id '%s' in %s"
                     % (bid, spec_path(root, gate)))
        missing = {"dut", "clock_mhz", "configs"} - set(entry)
        if missing:
            sys.exit("ERROR: %s: missing %s" % (bid, ", ".join(sorted(missing))))
        group = entry.get("group", "default")
        build = dict(entry)
        build["group"] = group
        build["known_issue"] = entry.get("known_issue",
                                         defaults.get("known_issue", ""))
        build["thresholds"] = entry.get("thresholds",
                                        defaults.get("thresholds", {}))
        recorded = baselines.get(group, {}).get(bid, {})
        result = recorded.get("result") or {}
        # An all-null skeleton means "never recorded" -- the file ships with the
        # shape visible, not with numbers.
        build["baseline"] = result if result.get("fmax_mhz") is not None else None
        build["baseline_env"] = result.get("env") or {}
        build["config_hash"] = recorded.get("config_hash")
        builds[bid] = build
    return builds


def select(builds, selectors):
    """Resolve --build selectors: an exact build id wins over a group of the
    same name; otherwise a group selects all its builds."""
    if not selectors:
        return [builds[b] for b in sorted(builds)]
    picked = {}
    for sel in selectors:
        hits = [b for b in builds.values() if b["id"] == sel]
        if not hits:
            hits = [b for b in builds.values() if b["group"] == sel]
        if not hits:
            sys.exit("ERROR: no build or group named '%s' (see --list)" % sel)
        for b in hits:
            picked[b["id"]] = b
    return [picked[b] for b in sorted(picked)]


def config_hash(build, env):
    """Fingerprint the synthesis inputs; a change invalidates the stored metrics."""
    key = "|".join([build["dut"], str(build["clock_mhz"]), build["configs"],
                    env["device"], str(env["opt_level"]), str(env["xlen"])])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def save_baselines(root, tool, results, env):
    """Write measured metrics to the golden baselines (never done by CI).

    Each entry records the GPU configuration it represents -- dut, target clock,
    and configs -- so a baseline is meaningful on its own: a set of metrics with
    no identifiable config is worthless. The yaml `builds:` block stays the
    source of truth for what to build; config_hash fingerprints that spec and
    invalidates the metrics if it drifts, so the recorded config cannot be
    hand-edited to disagree with the numbers next to it without failing the gate.
    """
    by_group = {}
    for r in results:
        by_group.setdefault(r["build"]["group"], []).append(r)
    os.makedirs(baseline_dir(root, tool), exist_ok=True)
    for group, rs in by_group.items():
        path = baseline_path(root, tool, group)
        entries = {}
        if os.path.exists(path):
            with open(path) as fh:
                entries = json.load(fh)
        for r in rs:
            b = r["build"]
            entries[b["id"]] = {
                "dut": b["dut"],
                "clock_mhz": b["clock_mhz"],
                "configs": b["configs"],
                "config_hash": config_hash(b, env),
                "result": dict(r["metrics"], env=dict(env)),
            }
        with open(path, "w") as fh:
            json.dump(entries, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("updated %s" % path)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def prepare_build_tree(root, build_dir, xlen):
    """Configure the gate's own out-of-tree build if it is not there yet."""
    if os.path.exists(os.path.join(build_dir, "config.mk")):
        return
    tooldir = os.environ.get("TOOLDIR") or os.path.expanduser("~/tools")
    os.makedirs(build_dir, exist_ok=True)
    print("configuring build tree %s (xlen=%d, tooldir=%s)"
          % (build_dir, xlen, tooldir))
    subprocess.run([os.path.join(root, "configure"),
                    "--tooldir=%s" % tooldir, "--xlen=%d" % xlen],
                   cwd=build_dir, check=True)


def read_summary(path):
    """Parse project.tcl's synth_summary.csv into the metric dict."""
    with open(path) as fh:
        row = next(csv.DictReader(fh))
    out = {}
    for m in METRICS:
        if m in row:
            out[m] = round(float(row[m]), 3) if m in ("fmax_mhz", "wns_ns") \
                else int(float(row[m]))
    return out


def read_paths(path):
    """Top critical paths, parsed from Vivado's timing.rpt.

    project.tcl already writes one (`report_timing -unique_pins -nworst 100`), so
    this needs no extra Vivado work and can be re-read from any finished build.
    -unique_pins is what makes the list 10 REAL paths rather than 10 views of the
    same one through different pins.

    The logic/route split is kept because it is the diagnosis, not decoration: a
    path that is 66% route is placement/congestion, and no amount of retiming the
    logic will fix it.
    """
    if not os.path.exists(path):
        return []
    paths, cur = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"Slack \((MET|VIOLATED)\)\s*:\s*(-?[\d.]+)ns", line)
            if m:
                if cur:
                    paths.append(cur)
                    if len(paths) >= NUM_PATHS:
                        break
                cur = {"slack_ns": float(m.group(2))}
            elif cur is None:
                continue
            elif line.startswith("Source:"):
                cur["startpoint"] = line.split(None, 1)[1]
            elif line.startswith("Destination:"):
                cur["endpoint"] = line.split(None, 1)[1]
            elif line.startswith("Path Group:"):
                cur["group"] = line.split(None, 2)[2]
            elif line.startswith("Logic Levels:"):
                cur["levels"] = int(line.split()[2])
            elif line.startswith("Data Path Delay:"):
                d = re.search(r"logic ([\d.]+)ns .*?route ([\d.]+)ns \(([\d.]+)%\)",
                              line)
                if d:
                    cur["logic_ns"] = float(d.group(1))
                    cur["route_ns"] = float(d.group(2))
                    cur["route_pct"] = float(d.group(3))
    if cur and len(paths) < NUM_PATHS:
        paths.append(cur)
    return paths


def read_fanout(path):
    """Top high-fanout nets, from the high_fanout_nets.rpt project.tcl writes.

    Recorded, never gated. A net fanning out to thousands of loads is what forces
    the placer to spread its cone across the die, which is the routing delay the
    critical paths then eat -- so this is the other half of the same diagnosis:
    the paths say WHERE the design is tight, these say WHY.
    """
    if not os.path.exists(path):
        return []
    nets = []
    for line in open(path, errors="replace"):
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) != 3 or not cols[1].isdigit():
            continue                      # header, rule, or a non-table line
        nets.append({"net": cols[0], "fanout": int(cols[1]), "driver": cols[2]})
        if len(nets) >= NUM_FANOUT:
            break
    return nets


def threshold(build, metric, args):
    """Gate tolerance for one metric: entry > per-metric global > global."""
    return float((build.get("thresholds") or {}).get(
        metric, args.metric_threshold.get(metric, args.threshold)))


def say(build, tag, msg, t0=None):
    at = "" if t0 is None else " (%dm)" % ((time.monotonic() - t0) // 60)
    print("  [%-5s] %-10s %s%s" % (tag, build["id"], msg, at))


def watch(proc, log, build, timeout, verbose, seen=0):
    """Follow a running build's log: report phases, announce elaboration, collect errors.

    A config mistake -- a bad define, a missing source, a parameter or hierarchy
    error -- dies during RTL elaboration, seconds into synthesis. Passing
    ELAB_DONE is therefore the point where a build stops being a typo risk and
    starts being an hours-long run worth waiting for, so it is announced as soon
    as it lands rather than at the end. Note Vivado prints ELAB_DONE on the way
    out of elaboration whether it passed or failed, so a clean pass also requires
    no ELAB_FAIL and no error behind it.

    Returns (rc, reached_elab, errors).
    """
    reached = failed = False
    errors, phase, t0, beat = [], None, time.monotonic(), time.monotonic()
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            with open(log, errors="replace") as fh:
                fh.seek(seen)
                for line in fh:
                    if verbose:
                        print("  | %-10s %s" % (build["id"], line.rstrip()))
                    if ELAB_FAIL in line:
                        failed = True
                    elif ELAB_DONE in line and not (failed or errors):
                        reached = True
                        say(build, "elab", "past RTL elaboration -- into "
                            "synthesis", t0)
                    for marker, label in PHASES:
                        if marker in line and label != phase:
                            phase = label
                            if not verbose:   # verbose already streams the line
                                say(build, "phase", label, t0)
                    if line.startswith("ERROR:") and len(errors) < 5:
                        errors.append(line.strip())
                seen = fh.tell()
        if rc is not None:
            return rc, reached, errors
        now = time.monotonic()
        if now - t0 > timeout:
            proc.kill()
            raise BuildError("timed out after %ds (elaboration %s)"
                             % (timeout, "passed" if reached else "NOT reached"))
        if not verbose and now - beat >= HEARTBEAT:
            beat = now
            say(build, "...", "%s, still running" % (phase or "starting"), t0)
        time.sleep(POLL_INTERVAL)


def load_stamp(work, chash):
    """This build dir's resume state, if it belongs to the CURRENT config."""
    try:
        with open(os.path.join(work, STAMP)) as fh:
            stamp = json.load(fh)
    except (OSError, ValueError):
        return None
    # A stamp from a different config means the checkpoints in this tree are for
    # a design we are no longer building -- resuming from them would silently
    # synthesize the OLD config. Treat as absent: rebuild from scratch.
    return stamp if stamp.get("config_hash") == chash else None


def save_stamp(work, chash, status, metrics=None):
    with open(os.path.join(work, STAMP), "w") as fh:
        json.dump({"config_hash": chash, "status": status, "metrics": metrics},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")


def run_build(build, build_dir, env, args):
    """Synthesize one DUT; return its metrics. Raises BuildError on failure."""
    dut_root = os.path.join(build_dir, "hw", "syn", "xilinx", "dut")
    src = os.path.join(dut_root, build["dut"])
    if not os.path.isdir(src):
        raise BuildError("no DUT target '%s' under %s" % (build["dut"], dut_root))

    # Unique PREFIX per build id: its own build tree and its own log, so parallel
    # builds (and other synthesis runs on this machine) cannot collide.
    prefix = "fpga_gate_" + build["id"]
    work = os.path.join(src, prefix)
    chash = config_hash(build, env)

    # Resume: a finished build of THIS config is reused as-is; an unfinished one
    # keeps its build tree so Vivado picks up from its post-synth/post-impl
    # checkpoint (project.tcl resumes on either) instead of re-running hours of
    # work. Anything else -- no stamp, or a stamp for a different config -- is a
    # clean rebuild.
    resume = False
    if args.resume:
        stamp = load_stamp(work, chash)
        if stamp and stamp["status"] == "done" and stamp["metrics"]:
            # A build finished before critical-path reporting existed has no
            # paths recorded. Rather than reuse an incomplete result, resume from
            # its post-impl checkpoint: project.tcl re-runs only run_report, so
            # the paths are backfilled in minutes instead of hours.
            if stamp["metrics"].get("critical_paths") and \
               stamp["metrics"].get("high_fanout_nets") is not None:
                say(build, "cache", "already built for this config -- reusing")
                return stamp["metrics"]
            say(build, "resum", "built, but no critical paths recorded -- "
                                "re-reporting from checkpoint")
            resume = True
        elif stamp:
            resume = True
            say(build, "resum", "reusing build tree, resuming from checkpoint")

    os.makedirs(work, exist_ok=True)
    shutil.copy2(os.path.join(src, "Makefile"), work)

    benv = dict(os.environ)
    benv.update(CONFIGS=build["configs"],
                CLK_FREQ_MHZ=str(build["clock_mhz"]),
                OPT_LEVEL=str(env["opt_level"]),
                MAX_JOBS=str(args.vivado_jobs))
    if resume:
        benv["RESUME"] = "1"   # make clean is a no-op; keeps project_1 + *.dcp

    log = os.path.join(work, "build.log")
    t0 = time.monotonic()
    if not resume:
        subprocess.run(["make", "-C", work, "clean"], env=benv,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       check=True)
    # Stamp BEFORE building: a run killed mid-flight leaves a resumable marker.
    save_stamp(work, chash, "running")

    if args.verbose:
        say(build, "run", "CONFIGS=%s CLK_FREQ_MHZ=%d OPT_LEVEL=%d MAX_JOBS=%d"
            % (build["configs"], build["clock_mhz"], env["opt_level"],
               args.vivado_jobs))
    # On resume the log is appended, so start watching past what is already
    # there -- otherwise the previous attempt's phases replay as if they were
    # this one's (and its old ERROR: lines resurface as this run's failure).
    seen = os.path.getsize(log) if resume and os.path.exists(log) else 0
    with open(log, "a" if resume else "w") as fh:
        proc = subprocess.Popen(["make", "-C", work], env=benv, stdout=fh,
                                stderr=subprocess.STDOUT)
        rc, reached_elab, errors = watch(proc, log, build, args.timeout,
                                         args.verbose, seen)
    elapsed = int(time.monotonic() - t0)

    if rc != 0:
        raise BuildError("%s (rc=%d) after %dm%s\n      see %s"
                         % ("FAILED BEFORE SYNTHESIS -- config/RTL error"
                            if not reached_elab else "build failed",
                            rc, elapsed // 60,
                            "".join("\n      | " + e for e in errors), log))

    summary = os.path.join(work, "synth_summary.csv")
    if not os.path.exists(summary):
        raise BuildError("no synth_summary.csv (Vivado did not reach report), "
                         "see %s" % log)
    metrics = read_summary(summary)
    metrics["build_time_s"] = elapsed
    metrics["critical_paths"] = read_paths(os.path.join(work, TIMING_RPT))
    metrics["high_fanout_nets"] = read_fanout(os.path.join(work, FANOUT_RPT))
    save_stamp(work, chash, "done", metrics)
    return metrics


def weight(build):
    """Scheduling cost: the recorded build time, else a per-DUT estimate."""
    recorded = (build.get("baseline") or {}).get("build_time_s")
    return recorded or DEFAULT_WEIGHTS.get(build["dut"], FALLBACK_WEIGHT)


def run_all(builds, build_dir, env, args):
    """Longest-processing-time-first list scheduling.

    Submitting longest-first to a fixed pool means a long build is always in
    flight from t=0 (it is dispatched first) and the remaining slots churn
    through the short ones -- so the makespan is bounded by the longest build
    rather than by a short build starting last and running past everything.
    """
    order = sorted(builds, key=weight, reverse=True)
    print("schedule (%d builds, %d parallel, %d Vivado jobs each):"
          % (len(order), args.jobs, args.vivado_jobs))
    for b in order:
        print("  %-10s dut=%-8s %4d MHz  ~%dm"
              % (b["id"], b["dut"], b["clock_mhz"], weight(b) // 60))
    if args.dry_run:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_build, b, build_dir, env, args): b
                   for b in order}
        for fut in concurrent.futures.as_completed(futures):
            b = futures[fut]
            try:
                metrics = fut.result()
                say(b, "done", "Fmax=%.1f MHz  LUT=%d  (%dm)"
                    % (metrics["fmax_mhz"], metrics["lut"],
                       metrics["build_time_s"] // 60))
                results.append({"build": b, "metrics": metrics, "error": None})
            except (BuildError, subprocess.CalledProcessError) as e:
                say(b, "FAIL", str(e))
                results.append({"build": b, "metrics": None, "error": str(e)})
    return sorted(results, key=lambda r: r["build"]["id"])


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate(result, env, args, gated):
    """Compare one result against its baseline; return (verdict, [reasons]).

    A build carrying a `known_issue` reason is tracked-expected-failure: it still
    builds, still reports, and its numbers still land in the table -- but its
    verdict does not fail the run. Same contract as a `known_issue:` test case
    (conftest marks those xfail strict=False), including that an unexpected pass
    is surfaced (XPASS) rather than converted into a hard failure.
    """
    build = result["build"]
    known = build.get("known_issue")

    def verdict(name, reasons):
        if not known:
            return name, reasons
        return "KNOWN-ISSUE", reasons + ["known issue: " + known]

    if result["error"]:
        return verdict("BUILD-FAIL", [result["error"]])

    base = build.get("baseline") or {}
    if not any(base.get(m) is not None for m in gated):
        return verdict("NO-BASELINE",
                       ["no recorded baseline (run --update-baseline)"])
    if build.get("config_hash") != config_hash(build, env):
        return verdict("STALE", ["config changed since the baseline was recorded "
                                 "(re-record with --update-baseline)"])

    reasons = []
    for m in gated:
        cur, ref = result["metrics"].get(m), base.get(m)
        if ref is None:
            reasons.append("%s: missing from baseline" % m)
            continue
        tol = threshold(build, m, args)
        delta = (cur - ref) / ref if ref else 0.0
        if abs(delta) <= tol:
            continue
        better = (delta > 0) == (m in HIGHER_IS_BETTER)
        reasons.append("%s: %g -> %g (%+.1f%%, tolerance %.0f%%) -- %s"
                       % (m, ref, cur, 100 * delta, 100 * tol,
                          "improvement, update the baseline to lock it in"
                          if better else "REGRESSION"))
    if not reasons:
        return ("XPASS", ["known issue no longer reproduces -- clear "
                          "`known_issue` from the catalog"]) if known \
            else ("PASS", [])
    return verdict("IMPROVED" if all("improvement" in r for r in reasons)
                   else "FAIL", reasons)


def report(results, env, args, gated, record=False):
    """Print the metric table; return the list of failing verdicts.

    In record mode the table is the review artifact for the baseline update, so
    the per-metric deltas are still shown (that is the diff a human signs off on)
    but they are not verdicts -- recording IS the intent.
    """
    print("\n%-10s %-8s %9s %9s %9s %9s %8s %6s %6s %6s  %s"
          % ("build", "dut", "fmax", "lut", "ff", "lutram", "bram", "uram",
             "dsp", "time", "verdict"))
    print("-" * 108)
    failures = []
    for r in results:
        b = r["build"]
        if record:
            verdict, reasons = ("RECORDED", []) if r["metrics"] \
                else ("BUILD-FAIL", [r["error"]])
        else:
            verdict, reasons = gate(r, env, args, gated)
        r["verdict"] = verdict
        if reasons:   # includes KNOWN-ISSUE/XPASS: reported, but not failing
            failures.append((b["id"], verdict, reasons))
        if r["metrics"] is None:
            print("%-10s %-8s %s" % (b["id"], b["dut"], verdict))
            continue
        m, base = r["metrics"], b.get("baseline") or {}

        def cell(key, fmt="%d"):
            cur = m.get(key)
            if cur is None:
                return "-"
            ref = base.get(key)
            if not ref:
                return fmt % cur
            return "%s(%+.0f%%)" % (fmt % cur, 100 * (cur - ref) / ref)

        print("%-10s %-8s %9s %9s %9s %9s %8s %6s %6s %5dm  %s"
              % (b["id"], b["dut"], cell("fmax_mhz", "%.0f"), cell("lut"),
                 cell("ff"), cell("lutram"), cell("bram"), cell("uram"),
                 cell("dsp"), m["build_time_s"] // 60, verdict))

    for bid, verdict, reasons in failures:
        print("\n%s: %s" % (bid, verdict))
        for reason in reasons:
            print("  - %s" % reason)
    return failures


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="FPGA synthesis-regression gate (Fmax/area vs golden baseline)")
    ap.add_argument("-b", "--build", action="append", default=[], metavar="ID",
                    help="build id or group to run (repeatable; default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=2, metavar="N",
                    help="max parallel Vivado builds (default: 2)")
    ap.add_argument("--vivado-jobs", type=int, default=0, metavar="N",
                    help="Vivado jobs per build (default: cores/jobs, clamped to "
                         "%d..%d)" % (MIN_VIVADO_JOBS, MAX_VIVADO_JOBS))
    ap.add_argument("--tool", default="xilinx", choices=("xilinx",),
                    help="synthesis toolchain; selects the golden file tree "
                         "ci/baselines/synthesis/<tool>/")
    ap.add_argument("--threshold", type=float, default=TOLERANCE, metavar="F",
                    help="global gate tolerance, fraction (default: %.2f). "
                         "Overridden per-metric by --metric-threshold and "
                         "per-build by the catalog's \"thresholds\"."
                         % TOLERANCE)
    ap.add_argument("--metric-threshold", action="append", default=[],
                    metavar="M=F",
                    help="global tolerance for one metric, e.g. lut=0.10 "
                         "(repeatable); still overridden per-build")
    ap.add_argument("--gate", default=",".join(GATED_METRICS), metavar="M,M",
                    help="metrics to assert on (default: %s); all metrics are "
                         "always recorded and reported" % ",".join(GATED_METRICS))
    ap.add_argument("--update-baseline", action="store_true",
                    help="record this run's metrics as the new golden baseline. "
                         "Human-reviewed; CI must never pass this.")
    ap.add_argument("--build-dir", metavar="DIR",
                    help="build tree to synthesize in (default: "
                         "<source>/build<xlen>_fpga_gate, configured on demand)")
    ap.add_argument("--xlen", type=int, default=32, choices=(32, 64),
                    help="ISA width (default: 32)")
    ap.add_argument("--opt-level", type=int, default=3, metavar="N",
                    help="Vivado optimization level, 0=fastest 3=default")
    ap.add_argument("--timeout", type=int, default=8 * 3600, metavar="SEC",
                    help="per-build timeout (default: 8h)")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted session: builds already finished "
                         "for this config are reused, unfinished ones pick up "
                         "from their Vivado checkpoint, the rest run. A build "
                         "whose config changed is rebuilt clean.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream each build's Vivado log to stdout (prefixed by "
                         "build id) instead of phase/heartbeat lines")
    ap.add_argument("--report", metavar="FILE",
                    help="write the full run (metrics + verdicts) as JSON")
    ap.add_argument("--list", action="store_true", help="list builds and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule without running Vivado")
    args = ap.parse_args(argv)

    # Line-buffer stdout: progress is the point of the elaboration watch, and it
    # is worthless if it sits in a block buffer until the run ends (which is what
    # happens whenever this is piped -- CI, nohup, a log file).
    sys.stdout.reconfigure(line_buffering=True)

    overrides = {}
    for spec in args.metric_threshold:
        metric, _, value = spec.partition("=")
        if metric not in METRICS or not value:
            sys.exit("ERROR: --metric-threshold expects <metric>=<fraction>, "
                     "metric one of: %s" % ", ".join(METRICS))
        overrides[metric] = float(value)
    args.metric_threshold = overrides

    root = source_root()
    builds = load_catalog(root, "fpga_gate", args.tool)

    if args.list:
        for b in select(builds, args.build):
            base = b.get("baseline") or {}
            print("%-10s [%s] dut=%-8s %4d MHz  %s%s"
                  % (b["id"], b["group"], b["dut"], b["clock_mhz"],
                     "baseline: Fmax=%.0f MHz LUT=%d"
                     % (base["fmax_mhz"], base["lut"])
                     if base.get("fmax_mhz") else "no baseline",
                     "  KNOWN ISSUE: " + b["known_issue"]
                     if b.get("known_issue") else ""))
            print("    %s" % b["configs"])
        return 0

    if not os.environ.get("XILINX_VIVADO"):
        sys.exit("ERROR: XILINX_VIVADO is not set -- source the Xilinx env first "
                 "(source ~/dev/xilinx_setup.sh)")

    if args.jobs < 1:
        sys.exit("ERROR: --jobs must be >= 1")
    if not args.vivado_jobs:
        args.vivado_jobs = max(MIN_VIVADO_JOBS,
                               min(MAX_VIVADO_JOBS, os.cpu_count() // args.jobs))

    build_dir = args.build_dir or os.path.join(
        root, "build%d_fpga_gate" % args.xlen)
    env = {"device": os.environ.get("DEVICE", "xcu55c-fsvh2892-2L-e"),
           "vivado": os.path.basename(os.environ["XILINX_VIVADO"].rstrip("/")),
           "opt_level": args.opt_level, "xlen": args.xlen}

    selected = select(builds, args.build)
    for b in selected:
        ref = b.get("baseline_env", {}).get("vivado")
        if ref and ref != env["vivado"]:
            print("WARNING: %s baseline was recorded on Vivado %s, running %s -- "
                  "results across tool versions are not comparable"
                  % (b["id"], ref, env["vivado"]))

    if not args.dry_run:
        prepare_build_tree(root, build_dir, args.xlen)
    results = run_all(selected, build_dir, env, args)
    if args.dry_run:
        return 0

    gated = tuple(m for m in args.gate.split(",") if m)
    outcomes = report(results, env, args, gated, args.update_baseline)
    failures = [o for o in outcomes if o[1] not in PASSING]

    ok = [r for r in results if r["metrics"]]
    if args.update_baseline and ok:
        save_baselines(root, args.tool, ok, env)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"env": env, "tolerance": args.threshold,
                       "metric_thresholds": args.metric_threshold,
                       "gated": list(gated),
                       "builds": [{"id": r["build"]["id"],
                                   "group": r["build"]["group"],
                                   "dut": r["build"]["dut"],
                                   "clock_mhz": r["build"]["clock_mhz"],
                                   "known_issue": r["build"].get("known_issue", ""),
                                   "metrics": r["metrics"],
                                   "error": r["error"],
                                   "verdict": r["verdict"]}
                                  for r in results]}, fh, indent=2)
            fh.write("\n")
        print("\nwrote %s" % args.report)

    if args.update_baseline:
        # Recording IS the intent; a delta is what we just wrote down. Only a
        # build that never produced metrics is still a failure -- unless it is a
        # tracked known issue.
        broken = [r for r in results
                  if not r["metrics"] and not r["build"].get("known_issue")]
        return 2 if broken else 0
    if any(v == "BUILD-FAIL" for _, v, _ in failures):
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
