#!/usr/bin/env python3
"""synth_gate — synthesis-regression gate over a checked-in golden baseline.

Same shape as the perf_gate (ci/perf_baseline.py): a golden baseline per build,
a hard threshold around it, and baselines that only a human updates. The
assertion differs: instead of rtlsim cycles it compares THIS commit's
post-synthesis timing and area against the recorded numbers, so an RTL change
that quietly costs timing closure or area cannot land green.

Two backends, one gate. Everything that is not "how do I drive the tool and what
does it call its numbers" is shared:

  --tool=xilinx  Vivado synthesis + place-and-route      -> ci/fpga_gate.py
  --tool=yosys   Yosys + OpenSTA on an ASIC PDK          -> ci/asic_gate.py

  Spec      : ci/testcases/<gate>.yaml                  (what to build)
  Baselines : ci/baselines/synthesis/<tool>/<group>.json (last accepted metrics)
  Metrics   : <build>/synth_summary.csv, written by the flow itself

Same division as every other check in the catalog: the yaml is hand-authored,
commented and reviewed; the baseline is machine-written and never hand-edited.
A `config_hash` (dut/clock/configs/tool env/xlen) ties the two together, so
metrics recorded under one config can never be compared against another.

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

# Gated metrics must stay within +/-5% of baseline. The regression side is the
# gate; the improvement side is the ratchet -- a win beyond it must be recorded
# so the gain is locked in and a later silent slide back to the old number still
# fails.
#
# This is the GLOBAL default. It is overridden per-metric on the command line
# (--metric-threshold lut=0.10) and per-build in the catalog ("thresholds":
# {"fmax_mhz": 0.03}) -- most specific wins. Same shape as model_parity's
# `tolerance` (per-case entry > category defaults > DEFAULT_PARITY_TOLERANCE).
TOLERANCE = 0.05

# Higher-is-better metrics. Everything else is lower-is-better.
HIGHER_IS_BETTER = ("fmax_mhz",)

# Verdicts that do not fail the run. KNOWN-ISSUE is a tracked expected failure
# (see gate()); XPASS is one that stopped reproducing -- reported loudly, but a
# gate that hard-failed on an unexpected PASS would be the wrong incentive.
PASSING = ("PASS", "RECORDED", "KNOWN-ISSUE", "XPASS")

POLL_INTERVAL = 5
HEARTBEAT = 300  # seconds between "still alive, here is where it is" lines

# Tool jobs per build, clamped so N parallel builds do not oversubscribe.
MIN_TOOL_JOBS = 2
MAX_TOOL_JOBS = 8


class BuildError(Exception):
    pass


# ---------------------------------------------------------------------------
# tools
#
# A backend supplies four things and nothing else: what it measures, how it is
# invoked, how its log reads, and what makes two runs comparable. The gating
# core below never names a tool.
# ---------------------------------------------------------------------------

class Tool:
    """One synthesis backend."""

    name = ""          # --tool value, and ci/baselines/synthesis/<name>/
    gate = ""          # ci/testcases/<gate>.yaml, and the build-dir/prefix/stamp name
    flow = ""          # DUT entry point, relative to a configured build tree

    metrics = ()       # recorded, in report order; build_time_s is bookkeeping
    gated = ()         # asserted by default (--gate overrides)
    integral = ()      # recorded as ints; every other metric is a rounded float
    columns = ()       # (header, metric, format, width) for the report table

    # Scheduling weights (seconds) for a build with no recorded build_time_s yet.
    # Only ever used to ORDER the queue, so a rough magnitude is enough.
    weights = {}
    fallback_weight = 1800
    job_label = "tool"   # what --tool-jobs buys, in the schedule banner

    # Log lines that mark a phase transition, in order. Progress is reported
    # against these, so a long build says where it actually is.
    phases = ()
    # The point past which a build can no longer die of a config/RTL typo (see
    # watch()), and -- where the tool prints one -- the line that says it did.
    elab_done = ""
    elab_fail = ""
    elab_label = "elaboration"
    error_re = re.compile(r"^ERROR:")

    def check_env(self):
        """Fail fast if the tool is not on this machine."""

    def env(self, args):
        """The tool environment recorded alongside every baseline."""
        raise NotImplementedError

    def hash_key(self, env):
        """The env fields that make two runs comparable, for config_hash."""
        raise NotImplementedError

    def env_warning(self, build, env):
        """Optional warning when a baseline was recorded under a different tool."""
        return None

    def dut_root(self, build_dir):
        return os.path.join(build_dir, *self.flow.split("/"))

    def work_dir(self, dut_root, prefix, build):
        raise NotImplementedError

    def prepare_flow(self, dut_root, env):
        """Once-per-session setup, before any build starts."""

    def clean(self, dut_root, work, build, benv):
        raise NotImplementedError

    def build_command(self, dut_root, work, build, env, args):
        """(argv, extra environment) that runs this build to completion."""
        raise NotImplementedError

    def extras(self, work):
        """Diagnostics recorded next to the metrics, never gated."""
        return {}

    def incomplete(self, metrics):
        """Whether a cached result predates something this tool now records.

        Reusing it would silently keep the gap forever, so --resume re-runs from
        the checkpoint instead -- minutes of reporting, not hours of synthesis.
        """
        return False

    def describe(self, metrics):
        """One-line result summary for the progress log."""
        return "  ".join("%s=%s" % (c[0], metrics.get(c[1])) for c in self.columns)

    def describe_baseline(self, base):
        """One-line baseline summary for --list."""
        return "  ".join("%s=%s" % (c[0], base.get(c[1])) for c in self.columns)


# --- catalog.mk parsing, shared by both DUT trees ---------------------------

def _catalog(dut_root):
    """The DUT catalog text (dut/catalog.mk is the single DUT list per flow)."""
    try:
        with open(os.path.join(dut_root, "catalog.mk")) as fh:
            return fh.read()
    except OSError:
        return ""


def _make_var(text, name):
    """One `NAME := value` from a catalog, with line continuations folded in."""
    m = re.search(r"^%s\s*:?=\s*((?:.*\\\n)*.*)$" % re.escape(name), text, re.M)
    return m.group(1).replace("\\\n", " ").strip() if m else ""


def dut_names(dut_root):
    """DUT ids declared by dut/catalog.mk."""
    return set(_make_var(_catalog(dut_root), "DUTS").split())


class Xilinx(Tool):
    """Vivado synthesis + place-and-route (hw/syn/xilinx/dut)."""

    name = "xilinx"
    gate = "fpga_gate"
    flow = "hw/syn/xilinx/dut"

    metrics = ("fmax_mhz", "wns_ns", "lut", "lutram", "ff", "bram", "uram",
               "dsp", "build_time_s")
    gated = ("fmax_mhz", "lut")
    integral = ("lut", "lutram", "ff", "bram", "uram", "dsp", "build_time_s")
    columns = (("fmax", "fmax_mhz", "%.0f", 9), ("lut", "lut", "%d", 9),
               ("ff", "ff", "%d", 9), ("lutram", "lutram", "%d", 9),
               ("bram", "bram", "%d", 8), ("uram", "uram", "%d", 6),
               ("dsp", "dsp", "%d", 6))

    weights = {"top": 14400, "core": 7200, "tcu": 3600, "rtu": 3600,
               "vortex": 3600, "cache": 1800, "raster": 1800, "dxa": 1800,
               "tex": 1200, "om": 1200}
    job_label = "Vivado"

    # Vivado prints the ELAB_DONE timing line on the way out of elaboration
    # EITHER WAY, so a clean pass is elab_done with no elab_fail behind it.
    elab_done = "Finished RTL Elaboration"
    elab_fail = "RTL Elaboration failed"
    elab_label = "RTL elaboration"
    phases = (
        ("create_project", "setup"),
        ("Starting synth_design", "elaboration"),
        (elab_done, "synthesis"),
        ("Synth Design complete", "opt"),
        ("Starting Logic Optimization Task", "opt"),
        ("Starting Placer Task", "placement"),
        ("Starting Routing Task", "routing"),
        ("report_timing", "reporting"),
    )

    # Top critical paths, read from the timing.rpt project.tcl already writes.
    # Recorded whether or not timing closed, and never gated: they are the
    # diagnostic that says WHERE the design is tight, so a Fmax regression comes
    # with its location instead of just its magnitude.
    TIMING_RPT = "timing.rpt"
    NUM_PATHS = 10
    # Top high-fanout nets (high_fanout_nets.rpt). Recorded, never gated -- the
    # "why" behind a route-bound critical path.
    FANOUT_RPT = "high_fanout_nets.rpt"
    NUM_FANOUT = 10

    def check_env(self):
        if not os.environ.get("XILINX_VIVADO"):
            sys.exit("ERROR: XILINX_VIVADO is not set -- source the Xilinx env "
                     "first (source ~/dev/xilinx_setup.sh)")

    def env(self, args):
        return {"device": os.environ.get("DEVICE", "xcu55c-fsvh2892-2L-e"),
                "vivado": os.path.basename(os.environ["XILINX_VIVADO"].rstrip("/")),
                "opt_level": args.opt_level, "xlen": args.xlen}

    def hash_key(self, env):
        return [env["device"], str(env["opt_level"]), str(env["xlen"])]

    def env_warning(self, build, env):
        ref = build.get("baseline_env", {}).get("vivado")
        if ref and ref != env["vivado"]:
            return ("baseline was recorded on Vivado %s, running %s -- results "
                    "across tool versions are not comparable" % (ref, env["vivado"]))
        return None

    def work_dir(self, dut_root, prefix, build):
        return os.path.join(dut_root, "%s_%s" % (prefix, build["dut"]))

    def clean(self, dut_root, work, build, benv):
        subprocess.run(["make", "-C", work, "clean"], env=benv,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       check=True)

    def build_command(self, dut_root, work, build, env, args):
        # build.mk plays the role the former dut/<name>/Makefile did: it is
        # copied in as the work tree's Makefile and run there with DUT=<name>.
        os.makedirs(work, exist_ok=True)
        shutil.copy2(os.path.join(dut_root, "build.mk"),
                     os.path.join(work, "Makefile"))
        benv = dict(DUT=build["dut"],
                    CONFIGS=build["configs"],
                    CLK_FREQ_MHZ=str(build["clock_mhz"]),
                    OPT_LEVEL=str(env["opt_level"]),
                    MAX_JOBS=str(args.tool_jobs))
        if build.get("impl_strategy"):
            benv["IMPL_STRATEGY"] = build["impl_strategy"]
        return ["make", "-C", work], benv

    def extras(self, work):
        return {"critical_paths": read_paths(os.path.join(work, self.TIMING_RPT),
                                             self.NUM_PATHS),
                "high_fanout_nets": read_fanout(os.path.join(work, self.FANOUT_RPT),
                                                self.NUM_FANOUT)}

    def incomplete(self, metrics):
        return not (metrics.get("critical_paths")
                    and metrics.get("high_fanout_nets") is not None)

    def describe(self, m):
        return "Fmax=%.1f MHz  LUT=%d" % (m["fmax_mhz"], m["lut"])

    def describe_baseline(self, base):
        return "Fmax=%.0f MHz LUT=%d" % (base["fmax_mhz"], base["lut"])


class Yosys(Tool):
    """Yosys + OpenSTA on an ASIC PDK (hw/syn/yosys/dut).

    No licence and no dedicated machine, so unlike the Vivado gate this one is
    cheap enough to run on a hosted runner. The metrics change with the target:
    LUT/BRAM/DSP have no ASIC meaning, cell area and SRAM area do.
    """

    name = "yosys"
    gate = "asic_gate"
    flow = "hw/syn/yosys/dut"

    metrics = ("fmax_mhz", "wns_ns", "tns_ns", "cell_area_um2", "seq_area_um2",
               "sram_area_um2", "cell_count", "power_mw", "build_time_s")
    gated = ("fmax_mhz", "cell_area_um2")
    integral = ("cell_count", "build_time_s")
    columns = (("fmax", "fmax_mhz", "%.0f", 9),
               ("area_um2", "cell_area_um2", "%.0f", 11),
               ("sram_um2", "sram_area_um2", "%.0f", 10),
               ("cells", "cell_count", "%d", 9),
               ("power_mw", "power_mw", "%.1f", 9))

    weights = {"vortex": 7200, "rtu": 11000, "gfx": 5400, "core": 3600,
               "tcu": 3600, "tensor": 3600, "vm": 1800, "cache": 900,
               "raster": 300, "dxa": 300, "tex": 300, "om": 120}
    fallback_weight = 900
    job_label = "make"

    # sv2v is the parser: a bad define, a missing include or a hierarchy error
    # dies there, before Yosys ever reads a line. run_synth.sh stamps `gen-ys`
    # immediately after source generation and conversion have both succeeded, so
    # that stamp is this flow's "past the typo risk" marker.
    elab_done = "TIME gen-ys"
    elab_label = "source conversion"
    phases = (
        ("gen_sources.sh", "sources"),
        ("sv2v.sh", "sv2v"),
        (elab_done, "synthesis"),
        ("TIME yosys", "sram"),
        ("run_sta.tcl", "timing"),
        ("DONE. Top:", "reporting"),
    )
    error_re = re.compile(r"^(?:ERROR|FATAL|Error)\b|^\S+: \*\*\* ")

    def env(self, args):
        return {"pdk": os.environ.get("PDK", "asap7"),
                "vt": os.environ.get("ASAP7_VT", "rvt"),
                "corner": os.environ.get("CORNER", "tt"),
                "yosys": tool_version("yosys", "-V"),
                "sta": tool_version("sta", "-version"),
                "sv2v": tool_version("sv2v", "--version"),
                "xlen": args.xlen}

    def hash_key(self, env):
        # ASAP7 RVT-TT and LVT-SS are not comparable, so the PDK corner is part
        # of the fingerprint the way the FPGA part number is for Vivado.
        #
        # The tool versions are in the fingerprint too, which they are NOT on the
        # Vivado side: area and Fmax move with the Yosys/ABC release far more
        # than Vivado's do, and often by more than the 5% gate tolerance. A
        # toolchain bump must therefore read as STALE ("re-record"), not as a
        # regression -- the alternative is a red gate that blames the RTL for a
        # tool change. All three ship in one tarball pinned by TOOLCHAIN_REV, so
        # this makes that pin part of the baseline contract.
        return [env["pdk"], env["vt"], env["corner"], str(env["xlen"]),
                env["yosys"], env["sta"], env["sv2v"]]

    def env_warning(self, build, env):
        # Area and Fmax move with the Yosys/ABC version far more than Vivado's
        # do, so a version change is worth saying out loud on every build.
        for key, label in (("yosys", "Yosys"), ("sta", "OpenSTA"),
                           ("sv2v", "sv2v")):
            ref = build.get("baseline_env", {}).get(key)
            if ref and env.get(key) and ref != env[key]:
                # The version is in config_hash, so this build is already going
                # to read STALE. Say WHICH tool moved -- STALE alone cannot.
                return ("baseline was recorded on %s %s, running %s -- results "
                        "across tool versions are not comparable, so this reads "
                        "as STALE until re-recorded" % (label, ref, env[key]))
        return None

    def tops(self, dut_root):
        """DUT -> top module, from dut/catalog.mk. The flow builds into
        <PREFIX>_<top>, so the work directory is not knowable without it."""
        text = _catalog(dut_root)
        return {d: _make_var(text, d + "_TOP") for d in _make_var(text, "DUTS").split()}

    def work_dir(self, dut_root, prefix, build):
        top = self.tops(dut_root).get(build["dut"])
        if not top:
            raise BuildError("no <dut>_TOP for '%s' in %s/catalog.mk"
                             % (build["dut"], dut_root))
        # hw/syn/yosys/Makefile builds in $(PREFIX)_$(TOP_LEVEL_ENTITY), one
        # level above the DUT dispatcher.
        return os.path.join(os.path.dirname(dut_root), "%s_%s" % (prefix, top))

    def prepare_flow(self, dut_root, env):
        """Install the PDK once, before the pool starts.

        The flow's `asap7` target is a prerequisite of every build and is phony,
        so N parallel builds would otherwise race the same download into the
        same directory. Installed once up front, they all hit its marker fast
        path instead.
        """
        subprocess.run(["make", "-C", os.path.dirname(dut_root), "asap7"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       check=True)

    def _make(self, dut_root, build, target):
        return ["make", "-C", dut_root, build["dut"], "TARGET=" + target,
                "PREFIX=%s_%s" % (self.gate, build["id"]),
                "CLOCK_FREQ=%d" % build["clock_mhz"],
                "CONFIGS=" + build["configs"]]

    def clean(self, dut_root, work, build, benv):
        subprocess.run(self._make(dut_root, build, "clean"), env=benv,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       check=True)
        os.makedirs(work, exist_ok=True)

    def build_command(self, dut_root, work, build, env, args):
        os.makedirs(work, exist_ok=True)
        return self._make(dut_root, build, "timing"), {}

    def describe(self, m):
        return "Fmax=%.1f MHz  area=%.0f um^2" % (m.get("fmax_mhz") or 0,
                                                  m.get("cell_area_um2") or 0)

    def describe_baseline(self, base):
        return "Fmax=%.0f MHz area=%.0f um^2" % (base["fmax_mhz"],
                                                 base.get("cell_area_um2") or 0)


TOOLS = {t.name: t for t in (Xilinx(), Yosys())}


def tool_version(name, flag):
    """First line of `<tool> <flag>`, or "" if it cannot be asked.

    Resolved out of $TOOLDIR the way hw/syn/common.mk does, with a PATH fallback:
    the prebuilt toolchain is not on PATH, and a version recorded from some other
    copy of yosys would make the baseline's tool env a lie.
    """
    tooldir = os.environ.get("TOOLDIR") or os.path.expanduser("~/tools")
    binary = os.path.join(tooldir, name, "bin", name)
    if not os.path.exists(binary):
        binary = name
    try:
        out = subprocess.run([binary, flag], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    text = out.stdout.decode(errors="replace").strip()
    if not text:
        return ""
    # Up to the build details: yosys reports its git sha AND the g++ that built
    # it ("Yosys 0.57+157 (git sha1 ..., g++ 11.4 ...)"). Those identify the
    # binary, not the tool, and this string is part of the baseline fingerprint
    # -- rebuilding the same yosys must not invalidate every recorded number.
    return text.splitlines()[0].split("(")[0].strip()


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


def load_catalog(root, tool):
    """Builds keyed by id: the SPEC from ci/testcases/<gate>.yaml, joined to its
    recorded metrics from ci/baselines/synthesis/<tool>/<group>.json.

    Same division as every other check in the catalog: the yaml says what to run
    (and is hand-authored, commented, reviewed), the baseline says what the last
    accepted answer was (and is machine-written). Neither file can be mistaken
    for the other's job.
    """
    with open(spec_path(root, tool.gate)) as fh:
        doc = yaml.safe_load(fh)
    defaults = doc.get("defaults") or {}
    # `builds:`, not `tests:` -- a synthesis build is not a pytest case (no
    # driver, no sim build tree, no xlen expansion, no app). The catalog's
    # `tests:` block holds the ONE case that invokes this script; the build list
    # is ours.
    spec = doc.get("builds")
    if not spec:
        sys.exit("ERROR: no `builds:` block in %s" % spec_path(root, tool.gate))

    baselines = {}
    for group in {e.get("group", "default") for e in spec}:
        path = baseline_path(root, tool.name, group)
        if os.path.exists(path):
            with open(path) as fh:
                baselines[group] = json.load(fh)

    builds = {}
    for entry in spec:
        bid = entry["id"]
        if bid in builds:
            sys.exit("ERROR: duplicate build id '%s' in %s"
                     % (bid, spec_path(root, tool.gate)))
        missing = {"dut", "clock_mhz", "configs"} - set(entry)
        if missing:
            sys.exit("ERROR: %s: missing %s" % (bid, ", ".join(sorted(missing))))
        group = entry.get("group", "default")
        build = dict(entry)
        build["group"] = group
        build["configs"] = entry["configs"] or ""
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


def config_hash(build, env, tool):
    """Fingerprint the synthesis inputs; a change invalidates the stored metrics."""
    key = "|".join([build["dut"], str(build["clock_mhz"]), build["configs"]]
                   + tool.hash_key(env))
    # Appended only when set, so builds without a strategy override keep the
    # hash their baselines were recorded under.
    if build.get("impl_strategy"):
        key += "|" + build["impl_strategy"]
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
    os.makedirs(baseline_dir(root, tool.name), exist_ok=True)
    for group, rs in by_group.items():
        path = baseline_path(root, tool.name, group)
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
                "config_hash": config_hash(b, env, tool),
                "result": dict(r["metrics"], env=dict(env)),
            }
            if b.get("impl_strategy"):
                entries[b["id"]]["impl_strategy"] = b["impl_strategy"]
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


def read_summary(path, tool):
    """Parse the flow's synth_summary.csv into the metric dict.

    A metric the flow could not measure is written as an empty cell rather than
    a zero (a missing SRAM estimate is not "no SRAM"), and stays absent here.
    """
    with open(path) as fh:
        row = next(csv.DictReader(fh))
    out = {}
    for m in tool.metrics:
        value = (row.get(m) or "").strip()
        if not value:
            continue
        out[m] = int(float(value)) if m in tool.integral \
            else round(float(value), 3)
    return out


def read_paths(path, limit):
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
                    if len(paths) >= limit:
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
    if cur and len(paths) < limit:
        paths.append(cur)
    return paths


def read_fanout(path, limit):
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
        if len(nets) >= limit:
            break
    return nets


def threshold(build, metric, args):
    """Gate tolerance for one metric: entry > per-metric global > global."""
    return float((build.get("thresholds") or {}).get(
        metric, args.metric_threshold.get(metric, args.threshold)))


def say(build, tag, msg, t0=None):
    at = "" if t0 is None else " (%dm)" % ((time.monotonic() - t0) // 60)
    print("  [%-5s] %-10s %s%s" % (tag, build["id"], msg, at))


def watch(proc, log, build, tool, timeout, verbose, seen=0):
    """Follow a running build's log: report phases, announce elaboration, collect errors.

    A config mistake -- a bad define, a missing source, a parameter or hierarchy
    error -- dies in the front end, seconds into the run. Passing the tool's
    elab_done marker is therefore the point where a build stops being a typo risk
    and starts being a long run worth waiting for, so it is announced as soon as
    it lands rather than at the end. Vivado prints its marker on the way out of
    elaboration whether it passed or failed, so a clean pass also requires no
    elab_fail and no error behind it.

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
                    if tool.elab_fail and tool.elab_fail in line:
                        failed = True
                    elif tool.elab_done in line and not (failed or errors):
                        reached = True
                        say(build, "elab", "past %s -- into synthesis"
                            % tool.elab_label, t0)
                    for marker, label in tool.phases:
                        if marker in line and label != phase:
                            phase = label
                            if not verbose:   # verbose already streams the line
                                say(build, "phase", label, t0)
                    if tool.error_re.match(line) and len(errors) < 5:
                        errors.append(line.strip())
                seen = fh.tell()
        if rc is not None:
            return rc, reached, errors
        now = time.monotonic()
        if now - t0 > timeout:
            proc.kill()
            raise BuildError("timed out after %ds (%s %s)"
                             % (timeout, tool.elab_label,
                                "passed" if reached else "NOT reached"))
        if not verbose and now - beat >= HEARTBEAT:
            beat = now
            say(build, "...", "%s, still running" % (phase or "starting"), t0)
        time.sleep(POLL_INTERVAL)


def load_stamp(work, chash, stamp_file):
    """This build dir's resume state, if it belongs to the CURRENT config."""
    try:
        with open(os.path.join(work, stamp_file)) as fh:
            stamp = json.load(fh)
    except (OSError, ValueError):
        return None
    # A stamp from a different config means the checkpoints in this tree are for
    # a design we are no longer building -- resuming from them would silently
    # synthesize the OLD config. Treat as absent: rebuild from scratch.
    return stamp if stamp.get("config_hash") == chash else None


def save_stamp(work, chash, status, stamp_file, metrics=None):
    with open(os.path.join(work, stamp_file), "w") as fh:
        json.dump({"config_hash": chash, "status": status, "metrics": metrics},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")


def run_build(build, build_dir, env, args, tool):
    """Synthesize one DUT; return its metrics. Raises BuildError on failure."""
    dut_root = tool.dut_root(build_dir)
    if build["dut"] not in dut_names(dut_root):
        raise BuildError("no DUT target '%s' in %s/catalog.mk"
                         % (build["dut"], dut_root))

    # Unique PREFIX per build id: its own build tree and its own log, so parallel
    # builds (and other synthesis runs on this machine) cannot collide. It is
    # also what keeps the Yosys flow's cached $(BUILD_DIR)/src from serving one
    # DUT's sources to the next.
    prefix = tool.gate + "_" + build["id"]
    work = tool.work_dir(dut_root, prefix, build)
    stamp_file = tool.gate + ".json"
    chash = config_hash(build, env, tool)

    # Resume: a finished build of THIS config is reused as-is; an unfinished one
    # keeps its build tree so the flow picks up from whatever it already
    # produced instead of re-running hours of work. Anything else -- no stamp,
    # or a stamp for a different config -- is a clean rebuild.
    resume = False
    if args.resume:
        stamp = load_stamp(work, chash, stamp_file)
        if stamp and stamp["status"] == "done" and stamp["metrics"]:
            if not tool.incomplete(stamp["metrics"]):
                say(build, "cache", "already built for this config -- reusing")
                return stamp["metrics"]
            say(build, "resum", "built, but its diagnostics predate this gate -- "
                                "re-reporting from checkpoint")
            resume = True
        elif stamp:
            resume = True
            say(build, "resum", "reusing build tree, resuming from checkpoint")

    benv = dict(os.environ)
    argv, extra = tool.build_command(dut_root, work, build, env, args)
    benv.update(extra)
    if resume:
        benv["RESUME"] = "1"   # make clean is a no-op; keeps project_1 + *.dcp

    log = os.path.join(work, "build.log")
    t0 = time.monotonic()
    if not resume:
        tool.clean(dut_root, work, build, benv)
    # Stamp BEFORE building: a run killed mid-flight leaves a resumable marker.
    save_stamp(work, chash, "running", stamp_file)

    if args.verbose:
        say(build, "run", " ".join(argv))
    # On resume the log is appended, so start watching past what is already
    # there -- otherwise the previous attempt's phases replay as if they were
    # this one's (and its old ERROR: lines resurface as this run's failure).
    seen = os.path.getsize(log) if resume and os.path.exists(log) else 0
    with open(log, "a" if resume else "w") as fh:
        proc = subprocess.Popen(argv, env=benv, stdout=fh,
                                stderr=subprocess.STDOUT)
        rc, reached_elab, errors = watch(proc, log, build, tool, args.timeout,
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
        raise BuildError("no synth_summary.csv (the flow did not reach report), "
                         "see %s" % log)
    metrics = read_summary(summary, tool)
    metrics["build_time_s"] = elapsed
    metrics.update(tool.extras(work))
    save_stamp(work, chash, "done", stamp_file, metrics)
    return metrics


def weight(build, tool):
    """Scheduling cost: the recorded build time, else a per-DUT estimate."""
    recorded = (build.get("baseline") or {}).get("build_time_s")
    return recorded or tool.weights.get(build["dut"], tool.fallback_weight)


def run_all(builds, build_dir, env, args, tool):
    """Longest-processing-time-first list scheduling.

    Submitting longest-first to a fixed pool means a long build is always in
    flight from t=0 (it is dispatched first) and the remaining slots churn
    through the short ones -- so the makespan is bounded by the longest build
    rather than by a short build starting last and running past everything.
    """
    order = sorted(builds, key=lambda b: weight(b, tool), reverse=True)
    print("schedule (%d builds, %d parallel, %d %s jobs each):"
          % (len(order), args.jobs, args.tool_jobs, tool.job_label))
    for b in order:
        print("  %-10s dut=%-8s %4d MHz  ~%dm"
              % (b["id"], b["dut"], b["clock_mhz"], weight(b, tool) // 60))
    if args.dry_run:
        return []

    tool.prepare_flow(tool.dut_root(build_dir), env)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_build, b, build_dir, env, args, tool): b
                   for b in order}
        for fut in concurrent.futures.as_completed(futures):
            b = futures[fut]
            try:
                metrics = fut.result()
                say(b, "done", "%s  (%dm)" % (tool.describe(metrics),
                                              metrics["build_time_s"] // 60))
                results.append({"build": b, "metrics": metrics, "error": None})
            except (BuildError, subprocess.CalledProcessError) as e:
                say(b, "FAIL", str(e))
                results.append({"build": b, "metrics": None, "error": str(e)})
    return sorted(results, key=lambda r: r["build"]["id"])


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate(result, env, args, gated, tool):
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
    if build.get("config_hash") != config_hash(build, env, tool):
        return verdict("STALE", ["config changed since the baseline was recorded "
                                 "(re-record with --update-baseline)"])

    reasons = []
    for m in gated:
        cur, ref = result["metrics"].get(m), base.get(m)
        if ref is None:
            reasons.append("%s: missing from baseline" % m)
            continue
        if cur is None:
            reasons.append("%s: not measured by this run" % m)
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

    # Target-frequency gate: Fmax must be within tolerance of the clock the design
    # was BUILT for, independent of the baseline. A baseline recorded below target
    # (a design that never met its clock) must not let a build pass just by matching
    # that number -- the gate answers "does it meet the frequency it targets?", not
    # only "did it get worse than last time?".
    target = build.get("clock_mhz")
    fmax = result["metrics"].get("fmax_mhz")
    if target and fmax is not None:
        tol = threshold(build, "fmax_mhz", args)
        shortfall = (float(target) - fmax) / float(target)
        if shortfall > tol:
            reasons.append("fmax_mhz: %.1f MHz vs %g MHz target (%.1f%% below, "
                           "tolerance %.0f%%) -- BELOW TARGET"
                           % (fmax, target, 100 * shortfall, 100 * tol))

    if not reasons:
        return ("XPASS", ["known issue no longer reproduces -- clear "
                          "`known_issue` from the catalog"]) if known \
            else ("PASS", [])
    return verdict("IMPROVED" if all("improvement" in r for r in reasons)
                   else "FAIL", reasons)


def report(results, env, args, gated, tool, record=False):
    """Print the metric table; return the list of failing verdicts.

    In record mode the table is the review artifact for the baseline update, so
    the per-metric deltas are still shown (that is the diff a human signs off on)
    but they are not verdicts -- recording IS the intent.
    """
    head = "%-10s %-8s" + "".join(" %%%ds" % c[3] for c in tool.columns)
    title = head % (("build", "dut") + tuple(c[0] for c in tool.columns)) \
        + " %6s  %s" % ("time", "verdict")
    print("\n" + title)
    # Slack in the rule: a cell carries its delta ("30113(+0%)") and routinely
    # overruns its column, so the rows are wider than the header.
    print("-" * (len(title) + 10))
    failures = []
    for r in results:
        b = r["build"]
        if record:
            verdict, reasons = ("RECORDED", []) if r["metrics"] \
                else ("BUILD-FAIL", [r["error"]])
        else:
            verdict, reasons = gate(r, env, args, gated, tool)
        r["verdict"] = verdict
        if reasons:   # includes KNOWN-ISSUE/XPASS: reported, but not failing
            failures.append((b["id"], verdict, reasons))
        if r["metrics"] is None:
            print("%-10s %-8s %s" % (b["id"], b["dut"], verdict))
            continue
        m, base = r["metrics"], b.get("baseline") or {}

        def cell(key, fmt):
            cur = m.get(key)
            if cur is None:
                return "-"
            ref = base.get(key)
            if not ref:
                return fmt % cur
            return "%s(%+.0f%%)" % (fmt % cur, 100 * (cur - ref) / ref)

        print(head % ((b["id"], b["dut"])
                      + tuple(cell(c[1], c[2]) for c in tool.columns))
              + " %5dm  %s" % (m["build_time_s"] // 60, verdict))

    for bid, verdict, reasons in failures:
        print("\n%s: %s" % (bid, verdict))
        for reason in reasons:
            print("  - %s" % reason)
    return failures


# ---------------------------------------------------------------------------

def main(argv=None, default_tool="xilinx"):
    tool = TOOLS[default_tool]
    ap = argparse.ArgumentParser(
        description="Synthesis-regression gate (Fmax/area vs golden baseline)")
    ap.add_argument("-b", "--build", action="append", default=[], metavar="ID",
                    help="build id or group to run (repeatable; default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=2, metavar="N",
                    help="max parallel synthesis builds (default: 2)")
    ap.add_argument("--tool-jobs", "--vivado-jobs", type=int, default=0,
                    dest="tool_jobs", metavar="N",
                    help="tool jobs per build (default: cores/jobs, clamped to "
                         "%d..%d). Vivado only." % (MIN_TOOL_JOBS, MAX_TOOL_JOBS))
    ap.add_argument("--tool", default=default_tool, choices=sorted(TOOLS),
                    help="synthesis toolchain; selects the spec "
                         "ci/testcases/<gate>.yaml and the golden file tree "
                         "ci/baselines/synthesis/<tool>/ (default: %s)"
                         % default_tool)
    ap.add_argument("--threshold", type=float, default=TOLERANCE, metavar="F",
                    help="global gate tolerance, fraction (default: %.2f). "
                         "Overridden per-metric by --metric-threshold and "
                         "per-build by the catalog's \"thresholds\"."
                         % TOLERANCE)
    ap.add_argument("--metric-threshold", action="append", default=[],
                    metavar="M=F",
                    help="global tolerance for one metric, e.g. lut=0.10 "
                         "(repeatable); still overridden per-build")
    ap.add_argument("--gate", metavar="M,M",
                    help="metrics to assert on (default: the tool's, %s for %s); "
                         "all metrics are always recorded and reported"
                         % (",".join(tool.gated), default_tool))
    ap.add_argument("--update-baseline", action="store_true",
                    help="record this run's metrics as the new golden baseline. "
                         "Human-reviewed; CI must never pass this.")
    ap.add_argument("--build-dir", metavar="DIR",
                    help="build tree to synthesize in (default: "
                         "<source>/build<xlen>_<gate>, configured on demand)")
    ap.add_argument("--xlen", type=int, default=32, choices=(32, 64),
                    help="ISA width (default: 32)")
    ap.add_argument("--opt-level", type=int, default=3, metavar="N",
                    help="Vivado optimization level, 0=fastest 3=default")
    ap.add_argument("--timeout", type=int, default=8 * 3600, metavar="SEC",
                    help="per-build timeout (default: 8h)")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted session: builds already finished "
                         "for this config are reused, unfinished ones pick up "
                         "from their checkpoint, the rest run. A build whose "
                         "config changed is rebuilt clean.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream each build's tool log to stdout (prefixed by "
                         "build id) instead of phase/heartbeat lines")
    ap.add_argument("--report", metavar="FILE",
                    help="write the full run (metrics + verdicts) as JSON")
    ap.add_argument("--list", action="store_true", help="list builds and exit")
    ap.add_argument("--matrix", action="store_true",
                    help="emit the selected builds as JSON and exit -- the "
                         "fan-out a workflow turns into one job per build")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule without running the tool")
    args = ap.parse_args(argv)
    tool = TOOLS[args.tool]

    # Line-buffer stdout: progress is the point of the elaboration watch, and it
    # is worthless if it sits in a block buffer until the run ends (which is what
    # happens whenever this is piped -- CI, nohup, a log file).
    sys.stdout.reconfigure(line_buffering=True)

    overrides = {}
    for spec in args.metric_threshold:
        metric, _, value = spec.partition("=")
        if metric not in tool.metrics or not value:
            sys.exit("ERROR: --metric-threshold expects <metric>=<fraction>, "
                     "metric one of: %s" % ", ".join(tool.metrics))
        overrides[metric] = float(value)
    args.metric_threshold = overrides

    root = source_root()
    builds = load_catalog(root, tool)

    if args.matrix:
        # Ordered longest-first so the slowest DUT is dispatched into the fan-out
        # first; GitHub starts matrix jobs in order, so this is the same makespan
        # argument run_all() makes for its thread pool.
        print(json.dumps(sorted(
            [{"id": b["id"], "group": b["group"], "dut": b["dut"],
              "clock_mhz": b["clock_mhz"],
              "known_issue": bool(b.get("known_issue"))}
             for b in select(builds, args.build)],
            key=lambda b: (-weight(builds[b["id"]], tool), b["id"]))))
        return 0

    if args.list:
        for b in select(builds, args.build):
            base = b.get("baseline") or {}
            print("%-10s [%s] dut=%-8s %4d MHz  %s%s"
                  % (b["id"], b["group"], b["dut"], b["clock_mhz"],
                     "baseline: " + tool.describe_baseline(base)
                     if base.get("fmax_mhz") else "no baseline",
                     "  KNOWN ISSUE: " + b["known_issue"]
                     if b.get("known_issue") else ""))
            print("    %s" % b["configs"])
        return 0

    tool.check_env()

    if args.jobs < 1:
        sys.exit("ERROR: --jobs must be >= 1")
    if not args.tool_jobs:
        args.tool_jobs = max(MIN_TOOL_JOBS,
                             min(MAX_TOOL_JOBS, os.cpu_count() // args.jobs))

    build_dir = args.build_dir or os.path.join(
        root, "build%d_%s" % (args.xlen, tool.gate))
    env = tool.env(args)

    selected = select(builds, args.build)
    for b in selected:
        warning = tool.env_warning(b, env)
        if warning:
            print("WARNING: %s %s" % (b["id"], warning))

    if not args.dry_run:
        prepare_build_tree(root, build_dir, args.xlen)
    results = run_all(selected, build_dir, env, args, tool)
    if args.dry_run:
        return 0

    gated = tuple(m for m in (args.gate.split(",") if args.gate else tool.gated) if m)
    outcomes = report(results, env, args, gated, tool, args.update_baseline)
    failures = [o for o in outcomes if o[1] not in PASSING]

    ok = [r for r in results if r["metrics"]]
    if args.update_baseline and ok:
        save_baselines(root, tool, ok, env)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"tool": tool.name, "env": env,
                       "tolerance": args.threshold,
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
