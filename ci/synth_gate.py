#!/usr/bin/env python3
"""synth_gate — synthesis-regression gate over a checked-in golden baseline.

Same shape as the perf_gate (ci/perf_baseline.py): a golden baseline per build,
a hard threshold around it, and baselines that only a human updates. The
assertion differs: instead of rtlsim cycles it compares THIS commit's
post-synthesis timing and area against the recorded numbers, so an RTL change
that quietly costs timing closure or area cannot land green.

Four backends, one gate. Everything that is not "how do I drive the tool and
what does it call its numbers" is shared:

  --tool=xilinx       Vivado synthesis + place-and-route   -> ci/fpga_gate.py
  --tool=yosys        Yosys + OpenSTA on an ASIC PDK       -> ci/asic_gate.py
  --tool=beacon       logic levels per RTL unit, no synthesis -> ci/logic_gate.py
  --tool=beacon_asic  Beacon's cell stages on asic_gate's builds (ASAP7)

  Spec      : ci/testcases/<gate>.yaml                  (what to build)
  Baselines : ci/baselines/synthesis/<tool>/<group>.json (last accepted metrics)
  Metrics   : <build>/synth_summary.csv (beacon_summary.csv for Beacon),
              written by the flow itself

Same division as every other check in the catalog: the yaml is hand-authored,
commented and reviewed; the baseline is machine-written and never hand-edited.
A `config_hash` (dut/clock/configs/tool env/xlen) ties the two together, so
metrics recorded under one config can never be compared against another.

See docs/designs/continuous_integration.md 3.5.
"""

import argparse
import concurrent.futures
import csv
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import yaml

# Fmax is held to the build's target clock and area to its fixed budget, both
# from the catalog and both one-sided: Fmax may not fall more than 2% below the
# target, area may not grow more than 5% over the budget, and doing better never
# fails. Neither is measured against the last recorded baseline, so re-recording
# cannot ratchet the limit; raising a budget is a reviewed catalog edit. Any
# other metric gated with --gate is compared with its baseline both ways: the
# improvement side is the ratchet, so a win must be recorded to lock it in.
#
# These are the GLOBAL defaults. They are overridden per-metric on the command
# line (--metric-threshold lut=0.10) and per-build in the catalog ("thresholds":
# {"fmax_mhz": 0.03}) -- most specific wins. Same shape as model_parity's
# `tolerance` (per-case entry > category defaults > DEFAULT_PARITY_TOLERANCE).
TOLERANCE = 0.05
METRIC_TOLERANCE = {"fmax_mhz": 0.02}

# Higher-is-better metrics. Everything else is lower-is-better.
HIGHER_IS_BETTER = ("fmax_mhz",)

# Verdicts that do not fail the run. KNOWN-ISSUE is a tracked expected failure
# (see gate()); XPASS is one that stopped reproducing -- reported loudly, but a
# gate that hard-failed on an unexpected PASS would be the wrong incentive.
# Against a reference (--against-rev), ACCEPTED is a regression the baseline's
# re-record accepted, and NO-REFERENCE a build the reference revision does not
# have at all (a DUT the change adds), judged against its baseline alone. REF-FAIL, a reference that failed to measure the build, fails.
PASSING = ("PASS", "RECORDED", "KNOWN-ISSUE", "XPASS", "ACCEPTED",
           "NO-REFERENCE")

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
    spec = ""          # another gate's ci/testcases/<spec>.yaml, if not the gate's own
    flow = ""          # DUT entry point, relative to a configured build tree

    metrics = ()       # recorded, in report order; build_time_s is bookkeeping
    gated = ()         # asserted by default (--gate overrides)
    area = ""          # the gated area metric, held to the catalog's budget
    budget_key = ""    # the catalog field holding that budget
    # The metric whose presence in a baseline entry means "recorded".
    key_metric = "fmax_mhz"
    # The flow's one-row metric file, in the build's work directory.
    summary = "synth_summary.csv"
    # Default gate tolerance, overridden by --threshold and the catalog.
    tolerance = TOLERANCE
    # Per-metric defaults over `tolerance`, overridden by --metric-threshold
    # and the catalog.
    metric_tolerance = {}
    integral = ()      # recorded as ints; every other metric is a rounded float
    columns = ()       # (header, metric, format, width) for the report table
    # Whether the build's clock_mhz is part of config_hash: it is an input of
    # a synthesis run, but only a gate threshold for an estimate that does not
    # read it, where raising the target must not invalidate the measurements.
    clock_in_hash = True
    # What a register's depth is counted in, for the reasons --against gives.
    depth_unit = "levels"
    # The dut/catalog.mk variable naming a DUT's top module (<dut><top_var>).
    top_var = "_PRJ"

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

    def compare(self, metrics, base):
        """Reasons beyond the per-metric tolerance check, as gate() words them:
        a reason containing "improvement" asks for a re-record, any other
        fails."""
        return []

    def evidence(self, work):
        """Files of a finished build that explain its reasons."""
        return []

    def notes(self, build, metrics, args):
        """Advisory lines printed under the report table; never a verdict."""
        return []

    def top_module(self, dut_root, dut):
        """The DUT's top module, from dut/catalog.mk ("" if not declared)."""
        return _make_var(_catalog(dut_root), dut + self.top_var)

    def diff_levels(self, build, metrics):
        """(watch, fix) thresholds for `beacon.py diff` against a reference."""
        return metrics.get("watch_levels") or 8, metrics.get("fix_levels") or 12

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


# --- Beacon, both targets ----------------------------------------------------

def harness_ties(work):
    """The instance-input bits the DUT top ties to constants, from the Beacon
    run's netlist (ci/beacon.py ties): the count and the ports that carry
    the most. A finished run always has its netlist, so a missing or
    unreadable one is a build error, never a count of zero."""
    run = os.path.join(work, "beacon")
    netlist = os.path.join(run, "netlist.mlir")
    if not os.path.exists(netlist):
        raise BuildError("no %s to count the harness's constant ties in" % netlist)
    import beacon
    try:
        rows = beacon.const_ties(netlist, beacon.netlist_top(run))
    except ValueError as e:
        raise BuildError("counting the harness's constant ties: %s" % e)
    return {"tied_bits": sum(r[0] for r in rows),
            "tied_ports": [{"port": p, "bits": n, "width": w}
                           for n, w, p in rows[:10]]}


def ties_reasons(metrics, base):
    """A harness whose constant ties changed: the synthesis gates fold the
    logic behind a tied input and Beacon does not, so they now measure a
    different design than the one their budgets were set on. Either way it
    needs a reviewed re-record; a baseline without the count is not gated."""
    now, was = metrics.get("tied_bits"), base.get("tied_bits")
    if now is None or was is None or now == was:
        return []
    return ["harness: %d -> %d instance-input bits tied to constants -- "
            "CHANGED: the synthesis gates now fold %s of the logic Beacon "
            "counts; review the fpga_gate/asic_gate budget, then re-record"
            % (was, now, "more" if now > was else "less")]


class Xilinx(Tool):
    """Vivado synthesis + place-and-route (hw/syn/xilinx/dut)."""

    name = "xilinx"
    gate = "fpga_gate"
    flow = "hw/syn/xilinx/dut"

    metrics = ("fmax_mhz", "wns_ns", "lut", "lutram", "ff", "bram", "uram",
               "dsp", "build_time_s")
    # Fmax is held to the build's clock target (gate()'s BELOW TARGET check).
    gated = ("lut",)
    area = "lut"
    budget_key = "lut_budget"
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
    top_var = "_TOP"

    metrics = ("fmax_mhz", "wns_ns", "tns_ns", "cell_area_um2", "seq_area_um2",
               "sram_area_um2", "cell_count", "power_mw", "build_time_s")
    # Fmax is held to the build's clock target (gate()'s BELOW TARGET check),
    # not to its baseline. ABC maps against that target (-D) and spends any
    # slack beyond it recovering area, so above the target Fmax moves with
    # whatever else the design contains: #424 left the RTU's worst path in an
    # unchanged FMA stage yet moved it 957 -> 903 MHz.
    gated = ("cell_area_um2",)
    area = "cell_area_um2"
    budget_key = "area_budget_um2"
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

    # gen_sources.sh fails on a bad define or a missing include while it writes
    # the sources; run_synth.sh stamps `gen-ys` once they are written. The design
    # is elaborated later, by read_slang inside Yosys, so a hierarchy error still
    # surfaces during the synthesis phase.
    elab_done = "TIME gen-ys"
    elab_label = "source generation"
    phases = (
        ("gen_sources.sh", "sources"),
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
        # tool change. Both ship in one tarball pinned by TOOLCHAIN_REV, so
        # this makes that pin part of the baseline contract.
        return [env["pdk"], env["vt"], env["corner"], str(env["xlen"]),
                env["yosys"], env["sta"]]

    def env_warning(self, build, env):
        # Area and Fmax move with the Yosys/ABC version far more than Vivado's
        # do, so a version change is worth saying out loud on every build.
        for key, label in (("yosys", "Yosys"), ("sta", "OpenSTA")):
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


class Beacon(Tool):
    """Beacon: logic levels per RTL unit and the design's area
    (hw/syn/xilinx/dut, target `beacon`).

    foldsim-estimate-timing ranks every register-to-register path of the CIRCT
    netlist the Xilinx flow elaborates by its logic levels, counted as Vivado
    counts them on DEVICE's family; foldsim-estimate-area counts its LUTs,
    flip-flops, distributed and block RAM, UltraRAM and DSPs, placing each
    memory as Vivado infers it. It needs no Vivado and no licence and takes
    seconds per DUT, so it can run on every change where the Fmax gate runs
    nightly.

    Depth is deterministic -- the same RTL and tools give the same levels -- so
    the gate has no tolerance: a unit that gets deeper fails, by name, until it
    is fixed or a human re-records it. Area is as deterministic, but a feature
    adds some by design, so a resource of the design fails past its baseline
    by more than fpga_gate's LUT tolerance (metric_tolerance), and a memory
    that moves between distributed and block RAM shows as both columns
    moving. What it gates is depth and cells, not Fmax or placement: Beacon
    does not see wires (docs/designs/continuous_integration.md 3.5).
    """

    name = "beacon"
    gate = "logic_gate"
    flow = "hw/syn/xilinx/dut"

    area_metrics = ("lut", "ff", "lutram", "bram", "uram", "dsp")
    metrics = ("max_levels", "deep_endpoints", "watch_endpoints",
               "endpoints") + area_metrics + ("pred_fmax_mhz", "risk_endpoints",
                                              "build_time_s")
    gated = ("max_levels",) + area_metrics
    integral = ("max_levels", "deep_endpoints", "watch_endpoints",
                "endpoints", "lut", "ff", "lutram", "uram", "dsp",
                "risk_endpoints", "build_time_s")
    key_metric = "max_levels"
    summary = "beacon_summary.csv"
    tolerance = 0.0
    metric_tolerance = dict.fromkeys(area_metrics, 0.05)
    columns = (("max_lv", "max_levels", "%d", 7),
               ("deep", "deep_endpoints", "%d", 9),
               ("watch", "watch_endpoints", "%d", 9),
               ("ends", "endpoints", "%d", 9),
               ("lut", "lut", "%d", 9), ("ff", "ff", "%d", 9),
               ("lutram", "lutram", "%d", 9), ("bram", "bram", "%g", 8),
               ("uram", "uram", "%d", 6), ("dsp", "dsp", "%d", 6),
               ("p_fmax", "pred_fmax_mhz", "%.1f", 8),
               ("risk", "risk_endpoints", "%d", 6),
               ("tied", "tied_bits", "%d", 6))

    # Elaboration and mapping of the widest DUTs; the level count itself is
    # seconds.
    weights = {"top": 600, "vortex": 600, "gfx": 600, "core": 300,
               "tensor": 300, "tcu": 300}
    fallback_weight = 120
    job_label = "make"

    # The netlist exists once foldsim-opt starts mapping it; a typo in a define
    # or an include dies in gen_sources.sh or circt_elaborate.sh before that.
    elab_done = "foldsim-beacon-map"
    elab_label = "elaboration"
    phases = (
        ("gen_sources.sh", "sources"),
        ("circt_elaborate.sh", "elaboration"),
        (elab_done, "mapping"),
        ("foldsim-estimate-timing", "levels"),
        ("foldsim-estimate-area", "area"),
        ("beacon_summary.py", "reporting"),
    )
    error_re = re.compile(r"^(?:ERROR|FATAL|Error)\b|: error:|^\S+: \*\*\* ")

    UNITS = "beacon_units.json"
    PATHS = "beacon_paths.txt"
    AREA = "beacon_area.txt"
    RISK = "beacon_risk.txt"
    # The learned calibration per device family (docs/designs/beacon.md 7.1).
    CALIBRATIONS = {"xcu55c": "beacon_calibration_u55c.json"}
    RISK_ROWS = 40

    def env(self, args):
        return {"circt": tool_version("circt-verilog", "--version",
                                      package="circt", line_prefix="CIRCT"),
                "foldsim": tool_version("foldsim-opt", "--version",
                                        package="foldsim",
                                        line_prefix="foldsim"),
                "device": os.environ.get("DEVICE", "xcu55c-fsvh2892-2L-e"),
                "xlen": args.xlen}

    def hash_key(self, env):
        # The levels are the model's: a new foldsim or CIRCT can map the same
        # RTL to other levels, and another device's family counts them
        # differently; either must read as STALE, not as a regression.
        return [env["circt"], env["foldsim"], env["device"], str(env["xlen"])]

    def env_warning(self, build, env):
        for key, label in (("foldsim", "foldsim"), ("circt", "CIRCT")):
            ref = build.get("baseline_env", {}).get(key)
            if ref and env.get(key) and ref != env[key]:
                return ("baseline was recorded on %s %s, running %s -- this "
                        "reads as STALE until re-recorded" % (label, ref, env[key]))
        return None

    def work_dir(self, dut_root, prefix, build):
        return os.path.join(dut_root, "%s_%s" % (prefix, build["dut"]))

    def clean(self, dut_root, work, build, benv):
        subprocess.run(["make", "-C", work, "clean"], env=benv,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       check=True)

    def build_command(self, dut_root, work, build, env, args):
        os.makedirs(work, exist_ok=True)
        shutil.copy2(os.path.join(dut_root, "build.mk"),
                     os.path.join(work, "Makefile"))
        return ["make", "-C", work, "beacon"], dict(
            DUT=build["dut"], CONFIGS=build["configs"],
            BEACON_INTERFACES=build.get("interfaces", ""))

    def extras(self, work):
        try:
            with open(os.path.join(work, self.UNITS)) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return {}
        out = {"units": doc.get("units", {}),
               "interfaces": doc.get("interfaces", {}),
               "deepest_paths": doc.get("deepest_paths", []),
               "fix_levels": doc.get("fix_levels"),
               "watch_levels": doc.get("watch_levels")}
        out.update(self.risk(work))
        out.update(harness_ties(work))
        return out

    def risk(self, work):
        """The learned calibration's Fmax for the design, the slowest register's,
        and its registers likely to fail, written to beacon_risk.txt. Levels
        miss what wires decide (a wide enable driven across instances); the
        calibration reads width, fan-out and span as well, and ranks a design's
        registers far better than levels do. Advisory: its Fmax is not gated."""
        device = os.environ.get("DEVICE", "xcu55c-fsvh2892-2L-e")
        cal = next((f for d, f in self.CALIBRATIONS.items()
                    if device.startswith(d)), None)
        run = os.path.join(work, "beacon")
        if cal is None or not os.path.isdir(run):
            return {}
        import beacon
        with open(os.path.join(source_root(), "hw", "scripts", cal)) as fh:
            model = json.load(fh)
        rows, pathless = beacon.score_run(run, model)
        if not rows:
            return {}
        period = model.get("period_ns", beacon.CAL_PERIOD)
        rows.sort(key=lambda r: r[1])
        fmax = 1000.0 / (period - rows[0][1])
        risky = sum(1 for r in rows if r[2] >= 0.5)
        with open(os.path.join(work, self.RISK), "w") as fh:
            fh.write("predicted Fmax %.1f MHz (slowest register, advisory); %d of %d "
                     "registers P(fail) >= 0.5 at %.3f ns; model %s\n"
                     % (fmax, risky, len(rows), period, cal))
            if pathless:
                fh.write("warning: registers at %d levels or more lack a path; "
                         "the scores of those are partial\n"
                         % model["path_levels"])
            fh.write("\n%8s %7s %7s %4s  register\n"
                     % ("fmax", "slack", "P(fail)", "lv"))
            for p_near, slack, p_fail, lv, reg in rows[:self.RISK_ROWS]:
                fh.write("%8.1f %+7.3f %7.2f %4d  %s\n"
                         % (1000.0 / (period - slack), slack, p_fail, lv, reg))
        return {"pred_fmax_mhz": round(fmax, 1), "risk_endpoints": risky}

    def incomplete(self, metrics):
        return metrics.get("units") is None or "tied_bits" not in metrics

    def notes(self, build, metrics, args):
        # The calibrated Fmax against the clock is a pointer, not a verdict:
        # held to it, it failed known-good builds as often as bad ones.
        pfmax, clock = metrics.get("pred_fmax_mhz"), build.get("clock_mhz")
        if pfmax is None or not clock or pfmax >= clock:
            return []
        return ["p_fmax %.1f MHz is %.1f%% below the %g MHz clock; slowest "
                "registers in %s (advisory, not gated)"
                % (pfmax, 100.0 * (clock - pfmax) / clock, clock, self.RISK)]

    def compare(self, metrics, base):
        """Per unit: deeper than its baseline fails; a new unit at the fix
        threshold fails; shallower, or gone from the watch list, is an
        improvement to lock in. Per interface output: more bits reached
        combinationally from outside its instance fails, fewer is an
        improvement. A harness whose constant ties changed fails."""
        reasons = ties_reasons(metrics, base)
        cur, ref = metrics.get("units"), base.get("units")
        if cur is None or ref is None:
            return reasons
        fix = metrics.get("fix_levels") or 0
        for unit in sorted(set(cur) | set(ref)):
            now, was = cur.get(unit), ref.get(unit)
            if was is None:
                if now >= fix:
                    reasons.append("unit %s: new at %d levels (fix threshold "
                                   "%d) -- REGRESSION" % (unit, now, fix))
            elif now is None:
                reasons.append("unit %s: %d levels -> below the watch threshold "
                               "-- improvement, update the baseline to lock it "
                               "in" % (unit, was))
            elif now != was:
                reasons.append("unit %s: %d -> %d levels -- %s"
                               % (unit, was, now,
                                  "REGRESSION" if now > was else
                                  "improvement, update the baseline to lock "
                                  "it in"))
        cur, ref = metrics.get("interfaces") or {}, base.get("interfaces") or {}
        for port in sorted(set(cur) | set(ref)):
            now, was = cur.get(port, 0), ref.get(port, 0)
            if now != was:
                reasons.append("interface %s: %d -> %d bits reached "
                               "combinationally from outside -- %s"
                               % (port, was, now,
                                  "REGRESSION, register it" if now > was else
                                  "improvement, update the baseline to lock "
                                  "it in"))
        return reasons

    def evidence(self, work):
        # Every unit at or above the watch threshold, its paths op by op: a
        # failing unit is a heading there; the area by unit and each memory's
        # placement.
        return [p for p in (os.path.join(work, self.PATHS),
                            os.path.join(work, self.AREA),
                            os.path.join(work, self.RISK))
                if os.path.exists(p)]

    def describe(self, m):
        area = (", %d LUTs, %g BRAM, %d DSP" % (m["lut"], m["bram"], m["dsp"])
                if m.get("lut") is not None else "")
        return "max %d levels, %d end points >= %s%s" % (
            m["max_levels"], m["deep_endpoints"], m.get("fix_levels"), area)

    def describe_baseline(self, base):
        return "max %d levels" % base["max_levels"]


class BeaconAsic(Tool):
    """Beacon on the ASIC flow (hw/syn/yosys/dut, target `beacon`,
    target=$(PDK)): asic_gate's builds, DUT for DUT and config for config, in
    minutes instead of hours.

    foldsim-estimate-timing counts cell stages (docs/designs/beacon.md 4.3)
    and converts the deepest path to a period with the library's delays;
    foldsim-estimate-area counts cells, flip-flops and SRAM macros, a memory
    without a macro as flip-flops. Flip-flops and cells are held to the
    baseline (5 %), so a memory that leaves its SRAM macro, or one Yosys must
    build from flip-flops, fails here before the nightly synthesis. The
    predicted Fmax is held to the build's clock_mhz with a wide band (see
    FMAX_BAND): a backstop for a path far too deep to close, not a timing
    verdict. The sensitive timing check is register by register against a
    reference (--against-rev), where a path that got deeper fails by name.
    """

    name = "beacon_asic"
    gate = "logic_gate_asic"
    spec = "asic_gate"
    flow = "hw/syn/yosys/dut"
    top_var = "_TOP"

    metrics = ("fmax_mhz", "period_ps", "max_stages", "cells", "ff", "macros",
               "endpoints", "stage_ps", "overhead_ps", "build_time_s")
    gated = ("ff", "cells")
    integral = ("period_ps", "max_stages", "cells", "ff", "macros",
                "endpoints", "stage_ps", "overhead_ps", "build_time_s")
    key_metric = "period_ps"
    summary = "beacon_asic_summary.csv"
    # The predicted Fmax fails only more than FMAX_BAND below clock_mhz.
    # Beacon's period runs 0.59-1.05x OpenSTA's on the 14 DUTs (it prices
    # vm's L2 TLB MRU update at 46 stages where STA closes it at 1266 MHz),
    # so a tight band fails designs that close. Over every validated (design,
    # target) pair, the lowest prediction of a design STA closes is 0.749 of
    # its target (vm and the fixed top, 749 MHz at 1 GHz), and the highest of
    # one that misses badly is 0.615 (core and top before their FPU fixes,
    # 492 MHz at 800). 0.32 splits that gap evenly in ratio (about 10 % margin
    # each side) and holds at both the 800 MHz and the 1 GHz target.
    FMAX_BAND = 0.32
    metric_tolerance = {"fmax_mhz": FMAX_BAND}
    # Beacon reads no clock: the target only sets the band and the stage
    # thresholds, so a new target must not invalidate the measurements.
    clock_in_hash = False
    depth_unit = "stages"
    columns = (("fmax", "fmax_mhz", "%.0f", 7),
               ("stages", "max_stages", "%d", 7),
               ("cells", "cells", "%d", 9),
               ("ff", "ff", "%d", 8),
               ("macros", "macros", "%d", 7),
               ("tied", "tied_bits", "%d", 6))
    weights = {"top": 600, "gfx": 600, "core": 300, "tensor": 300, "vm": 300}
    fallback_weight = 120
    job_label = "make"
    elab_done = "foldsim-beacon-map"
    phases = (
        ("gen_sources.sh", "sources"),
        ("circt_elaborate.sh", "elaboration"),
        (elab_done, "mapping"),
        ("foldsim-estimate-timing", "stages"),
        ("foldsim-estimate-area", "area"),
        ("asic-summary", "reporting"),
    )
    error_re = re.compile(r"^(?:ERROR|FATAL|Error)\b|: error:|^\S+: \*\*\* ")

    REPORT = os.path.join("beacon", "report.txt")
    AREA = os.path.join("beacon", "area.txt")

    def env(self, args):
        return {"circt": tool_version("circt-verilog", "--version",
                                      package="circt", line_prefix="CIRCT"),
                "foldsim": tool_version("foldsim-opt", "--version",
                                        package="foldsim",
                                        line_prefix="foldsim"),
                "pdk": os.environ.get("PDK", "asap7"),
                "xlen": args.xlen}

    def hash_key(self, env):
        return [env["circt"], env["foldsim"], env["pdk"], str(env["xlen"])]

    def env_warning(self, build, env):
        return Beacon.env_warning(self, build, env)

    tops = Yosys.tops
    work_dir = Yosys.work_dir

    def diff_levels(self, build, metrics):
        """(watch, fix) stages for `beacon.py diff`: the most that fit 90 %
        and 100 % of the target period, plus one, with the run's own delays
        (period = overhead + stages x stage)."""
        period = 1e6 / float(build["clock_mhz"])
        stage = metrics.get("stage_ps") or 24
        over = metrics.get("overhead_ps") or 231
        return (int((0.9 * period - over) // stage) + 1,
                int((period - over) // stage) + 1)

    def _make(self, dut_root, build, target):
        return ["make", "-C", dut_root, build["dut"], "TARGET=" + target,
                "PREFIX=%s_%s" % (self.gate, build["id"]),
                "CONFIGS=" + build["configs"]]

    def clean(self, dut_root, work, build, benv):
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)

    def build_command(self, dut_root, work, build, env, args):
        os.makedirs(work, exist_ok=True)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "beacon.py")
        make = " ".join(shlex.quote(a) for a in self._make(dut_root, build, "beacon"))
        return ["sh", "-c", "%s && echo asic-summary && %s %s asic-summary %s %s"
                % (make, shlex.quote(sys.executable), shlex.quote(script),
                   shlex.quote(os.path.join(work, "beacon")),
                   shlex.quote(os.path.join(work, self.summary)))], {}

    def extras(self, work):
        return harness_ties(work)

    def incomplete(self, metrics):
        return "tied_bits" not in metrics

    def compare(self, metrics, base):
        return ties_reasons(metrics, base)

    def notes(self, build, metrics, args):
        # Below the target but inside the band: worth a look, not a verdict.
        # Beyond the band, gate() already failed it.
        fmax, clock = metrics.get("fmax_mhz"), build.get("clock_mhz")
        if fmax is None or not clock or fmax >= clock:
            return []
        band = threshold(build, "fmax_mhz", args)
        shortfall = (clock - fmax) / float(clock)
        if shortfall > band:
            return []
        return ["predicted Fmax %.0f MHz is %.1f%% below the %g MHz target, "
                "inside the %.0f%% band; deepest path in %s (advisory)"
                % (fmax, 100 * shortfall, clock, 100 * band, self.REPORT)]

    def evidence(self, work):
        return [p for p in (os.path.join(work, self.REPORT),
                            os.path.join(work, self.AREA))
                if os.path.exists(p)]

    def describe(self, m):
        return "Fmax=%.0f MHz (%d stages)  %d cells, %d FFs" % (
            m.get("fmax_mhz") or 0, m.get("max_stages") or 0,
            m.get("cells") or 0, m.get("ff") or 0)

    def describe_baseline(self, base):
        return "Fmax=%.0f MHz %d FFs" % (base.get("fmax_mhz") or 0, base.get("ff") or 0)


TOOLS = {t.name: t for t in (Xilinx(), Yosys(), Beacon(), BeaconAsic())}


def tool_version(name, flag, package=None, line_prefix=None):
    """First line of `<tool> <flag>`, or "" if it cannot be asked.

    Resolved out of $TOOLDIR the way hw/syn/common.mk does, with a PATH fallback:
    the prebuilt toolchain is not on PATH, and a version recorded from some other
    copy of yosys would make the baseline's tool env a lie.
    """
    tooldir = os.environ.get("TOOLDIR") or os.path.expanduser("~/tools")
    package = package or name
    # <PACKAGE>_PATH overrides the toolchain copy, as in hw/syn/common.mk.
    root = os.environ.get(package.upper() + "_PATH") or \
        os.path.join(tooldir, package)
    binary = os.path.join(root, "bin", name)
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
    # A tool that prints several components' versions names the one that
    # matters on a line of its own (circt-verilog: "CIRCT <sha>").
    if line_prefix:
        lines = [l.strip() for l in text.splitlines()
                 if l.strip().startswith(line_prefix)]
        return lines[0] if lines else ""
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


def baseline_dir(root, tool, override=None):
    """ci/baselines/synthesis/<tool>, or <override>/<tool> to read and write
    proposed baselines outside the tree."""
    return os.path.join(override or os.path.join(root, "ci", "baselines",
                                                 "synthesis"), tool)


def baseline_path(root, tool, group, override=None):
    return os.path.join(baseline_dir(root, tool, override), group + ".json")


def load_catalog(root, tool, xlen=32, baselines_from=None):
    """Builds keyed by id: the SPEC from ci/testcases/<gate>.yaml, joined to its
    recorded metrics from ci/baselines/synthesis/<tool>/<group>.json.

    Same division as every other check in the catalog: the yaml says what to run
    (and is hand-authored, commented, reviewed), the baseline says what the last
    accepted answer was (and is machine-written). Neither file can be mistaken
    for the other's job.

    A tool that measures another gate's builds (`spec`) takes their designs
    and targets, not their verdict policy: that gate's `known_issue` and
    `thresholds` describe its own tool's numbers.
    """
    with open(spec_path(root, tool.spec or tool.gate)) as fh:
        doc = yaml.safe_load(fh)
    defaults = doc.get("defaults") or {}
    # `builds:`, not `tests:` -- a synthesis build is not a pytest case (no
    # driver, no sim build tree, no xlen expansion, no app). The catalog's
    # `tests:` block holds the ONE case that invokes this script; the build list
    # is ours.
    spec = doc.get("builds")
    if not spec:
        sys.exit("ERROR: no `builds:` block in %s" % spec_path(root, tool.spec or tool.gate))

    baselines = {}
    for group in {e.get("group", "default") for e in spec}:
        path = baseline_path(root, tool.name, group, baselines_from)
        if os.path.exists(path):
            with open(path) as fh:
                baselines[group] = json.load(fh)

    builds = {}
    for entry in spec:
        bid = entry["id"]
        if bid in builds:
            sys.exit("ERROR: duplicate build id '%s' in %s"
                     % (bid, spec_path(root, tool.spec or tool.gate)))
        required = {"dut", "clock_mhz", "configs"} | ({tool.budget_key} if tool.budget_key else set())
        missing = required - set(entry)
        if missing:
            sys.exit("ERROR: %s: missing %s" % (bid, ", ".join(sorted(missing))))
        group = entry.get("group", "default")
        build = dict(entry)
        build["group"] = group
        build["configs"] = entry["configs"] or ""
        # A build may pin its own ISA width and hardware profile
        # (profiles/<name>.json, folded under its configs by config.mk); a
        # profile that models a 64-bit machine needs xlen: 64.
        build["xlen"] = entry.get("xlen", xlen)
        build["profile"] = entry.get("profile", "")
        build["opt_in"] = bool(entry.get("opt_in", False))
        borrowed = bool(tool.spec)
        build["known_issue"] = "" if borrowed else entry.get(
            "known_issue", defaults.get("known_issue", ""))
        build["thresholds"] = {} if borrowed else entry.get(
            "thresholds", defaults.get("thresholds", {}))
        recorded = baselines.get(group, {}).get(bid, {})
        result = recorded.get("result") or {}
        # An all-null skeleton means "never recorded" -- the file ships with the
        # shape visible, not with numbers.
        build["baseline"] = result if result.get(tool.key_metric) is not None \
            else None
        build["baseline_env"] = result.get("env") or {}
        build["config_hash"] = recorded.get("config_hash")
        # Regressions against a reference that the last re-record accepted
        # (--update-baseline with --against-rev); see compare_against().
        build["accepted"] = recorded.get("accepted") or {}
        builds[bid] = build
    return builds


def select(builds, selectors):
    """Resolve --build selectors: an exact build id wins over a group of the
    same name; otherwise a group selects all its builds. With no selector, every
    build but the opt-in ones, which run only when named."""
    if not selectors:
        return [builds[b] for b in sorted(builds) if not builds[b]["opt_in"]]
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
    clock = [str(build["clock_mhz"])] if tool.clock_in_hash else []
    key = "|".join([build["dut"]] + clock + [build["configs"]]
                   + tool.hash_key(dict(env, xlen=build["xlen"])))
    # Appended only when set, so builds without a strategy override or a
    # profile keep the hash their baselines were recorded under.
    if build.get("impl_strategy"):
        key += "|" + build["impl_strategy"]
    if build.get("profile"):
        key += "|profile=" + build["profile"]
    if build.get("interfaces"):
        key += "|interfaces=" + build["interfaces"]
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def save_baselines(root, tool, results, env, override=None):
    """Write measured metrics to the golden baselines (never done by CI).

    Each entry records the GPU configuration it represents -- dut, target clock,
    and configs -- so a baseline is meaningful on its own: a set of metrics with
    no identifiable config is worthless. The yaml `builds:` block stays the
    source of truth for what to build; config_hash fingerprints that spec and
    invalidates the metrics if it drifts, so the recorded config cannot be
    hand-edited to disagree with the numbers next to it without failing the gate.

    Recorded against a revision (--against-rev), an entry also lists the
    regressions against it as `accepted`: what the re-record decided, in the
    diff a reviewer reads, and what compare_against() lets pass.
    """
    by_group = {}
    for r in results:
        by_group.setdefault(r["build"]["group"], []).append(r)
    os.makedirs(baseline_dir(root, tool.name, override), exist_ok=True)
    for group, rs in by_group.items():
        path = baseline_path(root, tool.name, group, override)
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
                "result": dict(r["metrics"], env=dict(env, xlen=b["xlen"])),
            }
            if b.get("impl_strategy"):
                entries[b["id"]]["impl_strategy"] = b["impl_strategy"]
            if b.get("profile"):
                entries[b["id"]]["profile"] = b["profile"]
            against = r.get("against") or {}
            if against.get("rev") and against.get("regressions"):
                entries[b["id"]]["accepted"] = {
                    "against": against["rev"],
                    "regressions": against["regressions"]}
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


def kill_tree(proc, grace=10):
    """Terminate a build and everything it spawned.

    proc is the outer make; the expensive processes are its descendants
    (run_synth.sh -> yosys -> yosys-abc, or Vivado). Popen starts each build in
    its own session, so one signal to the process group reaches all of them --
    signalling proc alone leaves the toolchain running with nothing collecting
    its output, which is how a timed-out gfx build held a core and 38 GB for 16
    hours after the gate had moved on.

    SIGTERM first so a tool can clean up its temp dirs, then SIGKILL
    unconditionally: proc exiting says nothing about its descendants, and yosys
    hands work to a yosys-abc that can outlive the shell above it.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:                     # already reaped
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return                          # group already gone
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Unconditional, not "only if proc survived": the group can still hold a
    # descendant that trapped SIGTERM long after the outer make has exited.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


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
            kill_tree(proc)
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


def build_work(build, build_dir, tool):
    """A build's work directory.

    Unique PREFIX per build id: its own build tree and its own log, so parallel
    builds (and other synthesis runs on this machine) cannot collide. It is
    also what keeps the Yosys flow's cached $(BUILD_DIR)/src from serving one
    DUT's sources to the next.
    """
    return tool.work_dir(tool.dut_root(build_dir), tool.gate + "_" + build["id"],
                         build)


def run_build(build, build_dir, env, args, tool):
    """Synthesize one DUT; return its metrics. Raises BuildError on failure."""
    dut_root = tool.dut_root(build_dir)
    if build["dut"] not in dut_names(dut_root):
        raise BuildError("no DUT target '%s' in %s/catalog.mk"
                         % (build["dut"], dut_root))

    work = build_work(build, build_dir, tool)
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
            # The diagnostics are read from the finished build's own files,
            # so they can usually be brought up to date without the tool.
            try:
                metrics = dict(stamp["metrics"], **tool.extras(work))
            except BuildError:
                metrics = stamp["metrics"]
            if not tool.incomplete(metrics):
                save_stamp(work, chash, "done", stamp_file, metrics)
                say(build, "cache", "already built for this config -- reusing, "
                                    "diagnostics re-read")
                return metrics
            say(build, "resum", "built, but its diagnostics predate this gate -- "
                                "re-reporting from checkpoint")
            resume = True
        elif stamp:
            resume = True
            say(build, "resum", "reusing build tree, resuming from checkpoint")

    benv = dict(os.environ)
    argv, extra = tool.build_command(dut_root, work, build, env, args)
    benv.update(extra)
    # Set even when empty, so a PROFILE in the caller's environment cannot
    # leak into a build that names none.
    benv["PROFILE"] = build["profile"]
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
        # start_new_session puts the build in its own process group, so a
        # timeout can take down the whole toolchain subtree. proc is only the
        # outer make; the work happens in run_synth.sh -> yosys -> yosys-abc
        # below it, and signalling make alone leaves those orphaned. A gfx
        # build that blew the 8h timeout kept a core and 38 GB busy for a
        # further 16 hours after the gate had already reported it.
        proc = subprocess.Popen(argv, env=benv, stdout=fh,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        rc, reached_elab, errors = watch(proc, log, build, tool, args.timeout,
                                         args.verbose, seen)
    elapsed = int(time.monotonic() - t0)

    if rc != 0:
        raise BuildError("%s (rc=%d) after %dm%s\n      see %s"
                         % ("FAILED BEFORE SYNTHESIS -- config/RTL error"
                            if not reached_elab else "build failed",
                            rc, elapsed // 60,
                            "".join("\n      | " + e for e in errors), log))

    summary = os.path.join(work, tool.summary)
    if not os.path.exists(summary):
        raise BuildError("no %s (the flow did not reach report), see %s"
                         % (tool.summary, log))
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
            work = build_work(b, build_dir, tool)
            try:
                metrics = fut.result()
                say(b, "done", "%s  (%dm)" % (tool.describe(metrics),
                                              metrics["build_time_s"] // 60))
                results.append({"build": b, "metrics": metrics, "error": None,
                                "work": work})
            except (BuildError, subprocess.CalledProcessError) as e:
                say(b, "FAIL", str(e))
                results.append({"build": b, "metrics": None, "error": str(e),
                                "work": work})
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

    A BUILD-FAIL is NOT covered by that contract. `known_issue` annotates what a
    DUT's numbers do -- a PPA regression, a metric this RTL cannot yet hit -- and
    a build that never produced numbers has not reached the thing being excused.
    Treating the two alike lets an infrastructure outage read as green: when a
    missing TOOLDIR broke every asic_gate build at once, the six DUTs carrying a
    `known_issue` reported KNOWN-ISSUE and the run looked like five ordinary
    regressions instead of a gate that could not build anything.
    """
    build = result["build"]
    known = build.get("known_issue")

    def verdict(name, reasons):
        if not known:
            return name, reasons
        return "KNOWN-ISSUE", reasons + ["known issue: " + known]

    # Deliberately not routed through verdict(): see the docstring.
    if result["error"]:
        return "BUILD-FAIL", [result["error"]]

    reasons = []
    vs_base = [m for m in gated if m != tool.area]

    if tool.area in gated:
        budget = float(build[tool.budget_key])
        cur = result["metrics"].get(tool.area)
        if cur is None:
            reasons.append("%s: not measured by this run" % tool.area)
        else:
            tol = threshold(build, tool.area, args)
            excess = (cur - budget) / budget
            if excess > tol:
                reasons.append("%s: %g vs %g budget (%+.1f%%, tolerance %.0f%%) "
                               "-- OVER BUDGET"
                               % (tool.area, cur, budget, 100 * excess, 100 * tol))

    base = build.get("baseline") or {}
    if vs_base and not any(base.get(m) is not None for m in vs_base):
        return verdict("NO-BASELINE",
                       ["no recorded baseline (run --update-baseline)"])
    if vs_base and build.get("config_hash") != config_hash(build, env, tool):
        return verdict("STALE", ["config changed since the baseline was recorded "
                                 "(re-record with --update-baseline)"])

    for m in vs_base:
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

    reasons += tool.compare(result["metrics"], base)

    # Target-frequency gate: Fmax must be within tolerance of the clock the design
    # was BUILT for, never of the last recording. One-sided: a faster build passes,
    # since placement noise alone moves Fmax by more than the tolerance.
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


def report_entry(r, env, tool):
    """One build of the `--report` JSON: a superset of its baseline entry, next
    to the baseline the verdict was judged against."""
    b = r["build"]
    entry = {"id": b["id"],
             "group": b["group"],
             "dut": b["dut"],
             "clock_mhz": b["clock_mhz"],
             "configs": b["configs"],
             "config_hash": config_hash(b, env, tool),
             "thresholds": b.get("thresholds") or {},
             "known_issue": b.get("known_issue", ""),
             "metrics": r["metrics"],
             "baseline": b.get("baseline"),
             "baseline_config_hash": b.get("config_hash"),
             "error": r["error"],
             "verdict": r["verdict"],
             "reasons": r["reasons"]}
    if b.get("impl_strategy"):
        entry["impl_strategy"] = b["impl_strategy"]
    entry["xlen"] = b["xlen"]
    if b.get("profile"):
        entry["profile"] = b["profile"]
    if r.get("against"):
        entry["against"] = r["against"]
    return entry


def report(results, env, args, gated, tool, mode="gate"):
    """Print the metric table; return the list of failing verdicts.

    In record mode the table is the review artifact for the baseline update, so
    the per-metric deltas are still shown (that is the diff a human signs off on)
    but they are not verdicts -- recording IS the intent. In measure mode (a
    reference run for --against-rev) nothing is judged at all.
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
        if mode != "gate":
            verdict, reasons = (("RECORDED" if mode == "record" else "MEASURED",
                                 []) if r["metrics"]
                                else ("BUILD-FAIL", [r["error"]]))
        else:
            verdict, reasons = gate(r, env, args, gated, tool)
        r["verdict"] = verdict
        r["reasons"] = reasons
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

    works = {r["build"]["id"]: r.get("work") for r in results}
    for bid, verdict, reasons in failures:
        print("\n%s: %s" % (bid, verdict))
        for reason in reasons:
            print("  - %s" % reason)
        if works.get(bid) and verdict not in ("BUILD-FAIL", "STALE",
                                              "NO-BASELINE"):
            for path in tool.evidence(works[bid]):
                print("  see %s" % path)
    notes = [(r["build"]["id"], n) for r in results if r["metrics"]
             for n in tool.notes(r["build"], r["metrics"], args)]
    if notes:
        print("\nadvisory (not gated):")
        for bid, note in notes:
            print("  %-10s %s" % (bid, note))
    return failures




# ---------------------------------------------------------------------------
# reference runs (--against, --against-rev)
# ---------------------------------------------------------------------------

# Where --against-rev's reference takes REV's files: the RTL, the harnesses,
# the RTL the flows add (DPI headers, the ASIC RAM stubs) and the
# configuration; submodules are taken at the commit REV pins. Every other
# file is this checkout's, so both sides are measured by the same flow and
# the same tools.
REF_PATHS = ("hw/rtl", "hw/unittest", "hw/dpi", "hw/syn/libs/no_mem",
             "VX_config.toml", "VX_types.toml")
# Tracked paths no Beacon run reads: the reference stays valid across them.
FLOW_EXCLUDE = ("docs", "sim", "tests", "**/*.md")
# The default of --against-rev: how CI passes a pull request's merge base.
AGAINST_REV_ENV = "BEACON_AGAINST_REV"
# The default of --ref-cache.
REF_CACHE_ENV = "BEACON_REF_CACHE"
# Reference trees kept in the cache, the one in use among them.
REF_KEEP = 4
# Regressions listed per build against a reference; beacon_diff.txt has all.
REASONS_SHOWN = 10


class RefError(Exception):
    """The reference cannot be staged; the run is not gated (exit 2)."""


def git(root, *argv):
    return subprocess.check_output(["git", "-C", root] + list(argv))


def submodules(root):
    """Paths of this checkout's submodules (gitlinks)."""
    return [line.split(b"\t", 1)[1].decode()
            for line in git(root, "ls-files", "-s", "-z").split(b"\0")
            if line.startswith(b"160000 ")]


def paths_at(root, rev, paths):
    """The ones of `paths` that exist at REV."""
    return [p for p in paths if git(root, "ls-tree", "--name-only", rev, "--", p)]


def flow_fingerprint(root):
    """This checkout's flow, as staged and as edited: every tracked file but
    REF_PATHS, the submodules and FLOW_EXCLUDE. A reference measured under
    another flow is rebuilt; under this one it is reused whatever the RTL
    here does, which is what an RTL iteration loop wants."""
    spec = ["--", "."] + [":(exclude,glob)%s" % p if "*" in p else ":(exclude)%s" % p
                          for p in REF_PATHS + tuple(submodules(root)) + FLOW_EXCLUDE]
    h = hashlib.sha256()
    h.update(git(root, "ls-files", "-s", *spec))
    h.update(git(root, "diff", "--binary", *spec))
    return h.hexdigest()[:16]


def _extract(argv, dst):
    """`argv` (a git archive) unpacked into dst; False if it failed."""
    os.makedirs(dst, exist_ok=True)
    archive = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tar = subprocess.run(["tar", "-C", dst, "-xf", "-"], stdin=archive.stdout,
                         stderr=subprocess.DEVNULL)
    archive.stdout.close()
    return archive.wait() == 0 and tar.returncode == 0


def stage_reference(root, rev, src):
    """A source tree of this checkout's tracked files with REV's REF_PATHS and
    REV's submodule commits: REV's design under this checkout's flow."""
    shutil.rmtree(src, ignore_errors=True)
    os.makedirs(src)
    subs = submodules(root)
    skip = tuple(p + "/" for p in REF_PATHS)
    for line in git(root, "ls-files", "-s", "-z").split(b"\0"):
        if not line:
            continue
        path = line.decode().split("\t", 1)[1]
        if path in REF_PATHS or path.startswith(skip) or path in subs:
            continue
        dst = os.path.join(src, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(os.path.join(root, path)):
            shutil.copy2(os.path.join(root, path), dst)
    paths = paths_at(root, rev, REF_PATHS)
    if not paths:
        raise RefError("%s has none of %s: not a revision of this design"
                       % (rev, ", ".join(REF_PATHS)))
    if not _extract(["git", "-C", root, "archive", rev] + paths, src):
        raise RefError("git archive %s %s failed" % (rev, " ".join(paths)))
    for path in subs:
        dst = os.path.join(src, path)
        pinned = git(root, "ls-tree", rev, "--", path).split()
        if len(pinned) < 3 or pinned[1] != b"commit":
            # REV does not pin it: the flow finds this checkout's copy there.
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.path.join(root, path), dst)
            continue
        stage_submodule(os.path.join(root, path), pinned[2].decode(), dst, rev)
    open(os.path.join(src, ".ready"), "w").close()


def stage_submodule(repo, sha, dst, rev):
    """The submodule checked out at `repo`, at commit `sha`, into dst, with
    its own submodules at the commits it pins. One not checked out here is
    left empty, as it is in this checkout."""
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(os.path.join(repo, ".git")):
        return
    if not _extract(["git", "-C", repo, "archive", sha], dst):
        raise RefError("%s does not have %s, the commit %s pins (git -C %s fetch)"
                       % (repo, sha[:12], rev, repo))
    for line in git(repo, "ls-tree", "-r", "-z", sha).split(b"\0"):
        meta, _, path = line.decode().partition("\t")
        if meta.split()[:2] == ["160000", "commit"]:
            stage_submodule(os.path.join(repo, path), meta.split()[2],
                            os.path.join(dst, path), rev)


def defines_module(root, module, rev=None):
    """Whether REF_PATHS define `module`, at REV or in this checkout."""
    pattern = r"^[[:space:]]*module[[:space:]]+%s([^A-Za-z0-9_$]|$)" % re.escape(module)
    where = ([rev, "--"] + paths_at(root, rev, REF_PATHS) if rev
             else ["--"] + list(REF_PATHS))
    rc = subprocess.run(["git", "-C", root, "grep", "-q", "-E", pattern] + where,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc not in (0, 1):
        raise RefError("git grep %s failed (exit %d)" % (rev or "", rc))
    return rc == 0


def absent_at(root, rev, tool, dut):
    """Whether REV has no such DUT at all: neither the DUT in its catalog nor
    its top module. Conclusive only for a top this checkout's REF_PATHS
    define; a top generated or kept elsewhere, or one the catalog names by
    configuration, counts as present. A DUT REV has under another top (a
    renamed harness) is present, so its failed reference reads REF-FAIL."""
    catalog = os.path.join(root, *tool.flow.split("/"))
    top = tool.top_module(catalog, dut)
    if not top or "$" in top or not defines_module(root, top):
        return False
    try:
        text = git(root, "show", "%s:%s/catalog.mk" % (rev, tool.flow)).decode()
    except subprocess.CalledProcessError:
        return False
    if dut in _make_var(text, "DUTS").split():
        return False
    return not defines_module(root, top, rev)


def prune_references(cache, keep):
    """Remove the cache's reference trees beyond the REF_KEEP most recently
    used; `keep` is always kept, and one another run holds is skipped."""
    trees = []
    for d in os.listdir(cache):
        try:
            if re.fullmatch(r"[0-9a-f]{12}", d) and d != keep:
                trees.append((os.path.getmtime(os.path.join(cache, d)), d))
        except OSError:
            continue    # removed by another run since the listing
    for _, d in sorted(trees, reverse=True)[REF_KEEP - 1:]:
        with open(os.path.join(cache, d + ".lock"), "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            shutil.rmtree(os.path.join(cache, d), ignore_errors=True)


def reference_tree(root, rev, args, tool, builds, xlen):
    """Measure `builds` on REV's design under this checkout's flow; return
    (the build tree to compare against, REV's SHA, the ids of the builds
    REV does not have (absent_at), the reference run's exit code, the held
    lock).

    The reference lives in <cache>/<sha>/: `src/` is the staged source tree
    and `build<xlen>_<gate>/` its builds, resumable like any gate session, so
    a second run against the same REV reuses every finished build. CI keeps
    the build trees in its cache keyed by the merge base's SHA. A change of
    this checkout's flow (flow_fingerprint) discards both. <sha>.lock is held
    exclusively while the tree is checked, staged and built, and shared
    while it is read, so runs on one cache (the FPGA and the ASIC flow, or
    worktrees sharing BEACON_REF_CACHE) never remove each other's files."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--verify", "%s^{commit}" % rev],
            stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        raise RefError("--against-rev %s is not a commit in %s (fetch it first)"
                       % (rev, root))
    cache = os.path.abspath(args.ref_cache or os.environ.get(REF_CACHE_ENV) or
                            os.path.join(root, ".beacon_ref"))
    os.makedirs(cache, exist_ok=True)
    ref = os.path.join(cache, sha[:12])
    lock = open(ref + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    src = os.path.join(ref, "src")
    build_dir = os.path.join(ref, "build%d_%s" % (xlen, tool.gate))
    stamp = os.path.join(ref, "flow")
    flow = flow_fingerprint(root)
    try:
        with open(stamp) as fh:
            same = fh.read().strip() == flow
    except OSError:
        same = False
    if not same:
        shutil.rmtree(ref, ignore_errors=True)
    os.makedirs(ref, exist_ok=True)
    os.utime(ref)
    prune_references(cache, sha[:12])
    if not os.path.exists(os.path.join(src, ".ready")):
        print("reference: staging %s (%s) in %s" % (rev, sha[:12], src), flush=True)
        stage_reference(root, sha, src)
    with open(stamp, "w") as fh:
        fh.write(flow + "\n")

    absent = {b["id"] for b in builds if absent_at(root, sha, tool, b["dut"])}
    for bid in sorted(absent):
        print("reference: %s has no DUT for %s" % (sha[:12], bid), flush=True)
    present = [b["id"] for b in builds if b["id"] not in absent]
    rc = 0
    if present:
        argv = [sys.executable, os.path.join(src, "ci", "logic_gate.py"),
                "--tool", tool.name, "--build-dir", build_dir, "--resume",
                "--measure-only", "--xlen", str(xlen), "-j", str(args.jobs),
                "--timeout", str(args.timeout)]
        if args.tool_jobs:
            argv += ["--tool-jobs", str(args.tool_jobs)]
        if args.configs:
            argv.append("--configs=" + args.configs)
        for bid in present:
            argv += ["-b", bid]
        env = dict(os.environ)
        env.pop(AGAINST_REV_ENV, None)
        print("reference: %s (%s) in %s" % (rev, sha[:12], build_dir), flush=True)
        rc = subprocess.run(argv, env=env).returncode
        if rc:
            print("ERROR: the reference run failed (exit %d)" % rc, flush=True)
    fcntl.flock(lock, fcntl.LOCK_SH)
    return build_dir, sha, absent, rc, lock


def live_acceptance(build, root, rev, tool):
    """The baseline's accepted regressions, unless the reference already has
    them: a re-record's acceptance holds until it lands, and once the
    reference revision's baseline carries the same entry the change it
    accepted is part of the reference and the acceptance has done its job."""
    accepted = build.get("accepted") or {}
    if not accepted or not rev:
        return accepted
    path = "ci/baselines/synthesis/%s/%s.json" % (tool.name, build["group"])
    try:
        landed = json.loads(git(root, "show", "%s:%s" % (rev, path)))
    except (subprocess.CalledProcessError, ValueError):
        return accepted
    return {} if (landed.get(build["id"]) or {}).get("accepted") == accepted \
        else accepted


def accepted_by(accepted, reg):
    """Whether the baseline's last re-record accepted this regression: the
    same register, resource or harness, and no worse than then."""
    for a in (accepted or {}).get("regressions", ()):
        if a.get("kind") == reg["kind"] and a.get("name") == reg["name"]:
            if reg["kind"] == "ties":
                return a.get("after") == reg["after"]
            return reg["after"] <= a.get("after", float("-inf"))
    return False


def regression_reason(reg, tool, fix):
    kind, name, unit = reg["kind"], reg["name"], tool.depth_unit
    if kind == "deeper":
        return ("%s: %d -> %d %s -- DEEPER than the reference"
                % (name, reg["before"], reg["after"], unit))
    if kind == "new":
        return ("%s: new at %d %s (fix threshold %d) -- DEEPER than the "
                "reference" % (name, reg["after"], unit, fix))
    if kind == "area":
        return ("%s: %g -> %g (%+.1f%%) -- AREA grown past the tolerance "
                "against the reference"
                % (name, reg["before"], reg["after"],
                   100.0 * (reg["after"] - reg["before"]) / reg["before"]
                   if reg["before"] else 100.0))
    return ties_reasons({"tied_bits": reg["after"]},
                        {"tied_bits": reg["before"]})[0]


def compare_against(results, ref_dir, tool, env, root, rev=None, absent=()):
    """Each finished build's Beacon run against the same build in ref_dir
    (ci/beacon.py diff): register by register, area by resource, and the
    harness's constant ties. Sets each result's `against`: its verdict,
    the regressions, and the reasons.

    A build the reference revision does not have (a DUT the change adds,
    absent_at) reads NO-REFERENCE and is judged against its baseline alone. Any
    other build without a finished reference of the same config is REF-FAIL:
    a reference that timed out or crashed must not pass for one that agrees.

    A regression the baseline's re-record accepted (--update-baseline with
    --against-rev, stored in the baseline entry) reads ACCEPTED until that
    acceptance reaches the reference (live_acceptance). One the re-record
    did not see fails."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beacon.py")
    label = rev[:12] if rev else ref_dir
    rows = []
    for r in results:
        b, work = r["build"], r.get("work")
        if not r["metrics"] or not work:
            continue
        against = {"reference": label, "rev": rev, "regressions": []}
        r["against"] = against
        if b["id"] in absent:
            against.update(verdict="NO-REFERENCE", reasons=[
                "%s has no %s DUT (neither in its catalog nor its top module): "
                "judged against its baseline only" % (label, b["dut"])])
            rows.append((b["id"], "NO-REFERENCE", "-"))
            continue
        ref_work = build_work(b, ref_dir, tool)
        ref = os.path.join(ref_work, "beacon")
        try:
            with open(os.path.join(ref_work, tool.gate + ".json")) as fh:
                stamp = json.load(fh)
        except (OSError, ValueError):
            stamp = {}
        problem = (None if stamp.get("status") == "done" and stamp.get("metrics")
                   and os.path.exists(os.path.join(ref, "endpoints.csv"))
                   else "no finished reference build in %s (see its build.log)"
                   % ref_work)
        if not problem and stamp.get("config_hash") != config_hash(b, env, tool):
            problem = ("the reference in %s was measured under another "
                       "configuration (config hash %s, this build %s)"
                       % (ref_work, stamp.get("config_hash"),
                          config_hash(b, env, tool)))
        if problem:
            against.update(verdict="REF-FAIL", reasons=[problem])
            rows.append((b["id"], "REF-FAIL", "-"))
            continue
        ref_metrics = stamp["metrics"]
        out = os.path.join(work, "beacon_diff.txt")
        doc_path = os.path.join(work, "beacon_diff.json")
        if os.path.exists(doc_path):
            os.remove(doc_path)
        watch, fix = tool.diff_levels(b, r["metrics"])
        with open(out, "w") as fh:
            rc = subprocess.run([sys.executable, script, "diff", ref,
                                 os.path.join(work, "beacon"),
                                 "--min-levels", str(watch),
                                 "--fix-levels", str(fix),
                                 "--json", doc_path],
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        against["diff"] = out
        try:
            with open(doc_path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = None
        if rc not in (0, 1, 3) or doc is None:
            against.update(verdict="BUILD-FAIL",
                           reasons=["beacon.py diff failed (exit %d): %s" % (rc, out)])
            rows.append((b["id"], "BUILD-FAIL", "-"))
            continue
        regs = [dict(kind="deeper", name=d["register"], before=d["before"],
                     after=d["after"]) for d in doc["deeper"]]
        regs += [dict(kind="new", name=d["register"], before=None,
                      after=d["after"]) for d in doc["new"]]
        regs += [dict(kind="area", name=d["resource"], before=d["before"],
                      after=d["after"]) for d in doc["area"]]
        was, now = ref_metrics.get("tied_bits"), r["metrics"].get("tied_bits")
        if was is not None and now is not None and was != now:
            regs.append(dict(kind="ties", name="tied_bits", before=was, after=now))
        accepted = live_acceptance(b, root, rev, tool)
        open_regs = [g for g in regs if not accepted_by(accepted, g)]
        kinds = {g["kind"] for g in open_regs}
        verdict = ("DEEPER" if kinds & {"deeper", "new"} else
                   "AREA" if "area" in kinds else
                   "HARNESS" if "ties" in kinds else
                   "ACCEPTED" if regs else "PASS")
        # Generate copies of one register repeat its line; the diff has them all.
        shown = open_regs[:REASONS_SHOWN]
        reasons = [regression_reason(g, tool, fix) for g in shown]
        covered = [g for g in regs if g not in open_regs]
        reasons += ["accepted by the baseline's re-record against %s: %s"
                    % (accepted.get("against", "?")[:12],
                       regression_reason(g, tool, fix))
                    for g in covered[:max(0, REASONS_SHOWN - len(shown))]]
        hidden = len(regs) - len(reasons)
        if hidden > 0:
            reasons.append("... and %d more" % hidden)
        if reasons:
            reasons.append("see %s" % out)
        against.update(verdict=verdict, regressions=regs, reasons=reasons,
                       headline=doc.get("headline", ""))
        rows.append((b["id"], verdict, doc.get("headline", "")))

    if rows:
        print("\nagainst %s (%s):" % (label, ref_dir))
        for bid, verdict, head in rows:
            print("  %-10s %-12s %s" % (bid, verdict, head))
        for r in results:
            a = r.get("against") or {}
            if a.get("verdict") not in (None, "PASS"):
                print("\n%s against %s: %s" % (r["build"]["id"], label, a["verdict"]))
                for reason in a["reasons"]:
                    print("  - %s" % reason)


def trim_reference_cache(cache):
    """Shrink a reference cache to what a later comparison reads (stamps, end
    points, instances, area) before CI saves it. A build that did not finish
    is removed whole, so the next run under the same key rebuilds it clean
    instead of resuming from a tree whose generated sources are gone."""
    for ref in glob.glob(os.path.join(cache, "[0-9a-f]" * 12)):
        shutil.rmtree(os.path.join(ref, "src"), ignore_errors=True)
        works = glob.glob(os.path.join(ref, "build*", "hw", "syn", "*", "logic_gate*_*")) \
            + glob.glob(os.path.join(ref, "build*", "hw", "syn", "*", "dut", "logic_gate*_*"))
        for work in [w for w in works if os.path.isdir(w)]:
            stamps = glob.glob(os.path.join(work, "logic_gate*.json"))
            try:
                with open(stamps[0]) as fh:
                    done = json.load(fh).get("status") == "done"
            except (IndexError, OSError, ValueError):
                done = False
            if not done:
                shutil.rmtree(work)
                continue
            for top, dirs, files in os.walk(work):
                for d in [d for d in dirs if d == "src"]:
                    shutil.rmtree(os.path.join(top, d))
                    dirs.remove(d)
                for f in files:
                    if f.endswith((".mlir", ".mlirbc")) or \
                            f in ("paths.jsonl", "sources.txt", "sources.f"):
                        os.remove(os.path.join(top, f))


def reference_cache_done(cache):
    """The finished reference builds in a cache, one line each: the stamp's
    path and the config hash it was measured under."""
    done = []
    for stamp in glob.glob(os.path.join(cache, "*", "build*", "hw", "syn", "**",
                                        "logic_gate*.json"), recursive=True):
        try:
            with open(stamp) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if doc.get("status") == "done":
            done.append("%s %s" % (os.path.relpath(stamp, cache), doc.get("config_hash")))
    return sorted(done)


def reference_cache_id(done):
    """A digest of reference_cache_done()'s lines: CI saves a cache under it
    once the set has grown, so a later push can save the builds an earlier
    one did not finish, under a new key."""
    return hashlib.sha256("\n".join(done).encode()).hexdigest()[:16]


def merge_against(results):
    """Fold each build's verdict against the reference into its verdict: a
    failing one replaces a passing baseline verdict, and every reason is
    kept."""
    for r in results:
        a = r.get("against")
        if not a or a["verdict"] == "PASS":
            continue
        if a["verdict"] not in PASSING and r["verdict"] in PASSING:
            r["verdict"] = a["verdict"]
        r["reasons"] = r["reasons"] + ["against %s: %s" % (a["reference"], x)
                                       for x in a["reasons"]]


# ---------------------------------------------------------------------------

def main(argv=None, default_tool="xilinx"):
    tool = TOOLS[default_tool]
    ap = argparse.ArgumentParser(
        description="Synthesis-regression gate (Fmax vs target clock, area vs budget)")
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
    ap.add_argument("--threshold", type=float, default=None, metavar="F",
                    help="global gate tolerance, fraction (default: the "
                         "tool's, %.2f for synthesis, 0 for Beacon). "
                         "Overridden per-metric by --metric-threshold and "
                         "per-build by the catalog's \"thresholds\"."
                         % TOLERANCE)
    ap.add_argument("--metric-threshold", action="append", default=[],
                    metavar="M=F",
                    help="global tolerance for one metric, e.g. lut=0.10 "
                         "(repeatable; default: %s); still overridden "
                         "per-build" % ", ".join("%s=%g" % kv for kv in
                                                 METRIC_TOLERANCE.items()))
    ap.add_argument("--gate", metavar="M,M",
                    help="metrics to assert on (default: the tool's, %s for %s); "
                         "all metrics are always recorded and reported"
                         % (",".join(tool.gated), default_tool))
    ap.add_argument("--update-baseline", action="store_true",
                    help="record this run's metrics as the new golden baseline. "
                         "Human-reviewed; CI must never pass this. With "
                         "--against-rev, the regressions against that revision "
                         "are recorded as accepted")
    ap.add_argument("--baseline-dir", metavar="DIR",
                    help="read and write the baselines under DIR/<tool>/ "
                         "instead of ci/baselines/synthesis/<tool>/, e.g. to "
                         "propose baselines for review outside the tree")
    ap.add_argument("--build-dir", metavar="DIR",
                    help="build tree to synthesize in (default: "
                         "<source>/build<xlen>_<gate>, configured on demand)")
    ap.add_argument("--against", metavar="DIR",
                    help="Beacon only: the build tree of a reference run, "
                         "typically the parent revision's; each build's end "
                         "points are compared register by register, its "
                         "area by resource and unit (ci/beacon.py diff), and "
                         "its harness's constant ties, with the same build "
                         "there; a register that got deeper, area grown past "
                         "the tolerance, or changed ties fail the run")
    ap.add_argument("--against-rev", metavar="REV",
                    default=os.environ.get(AGAINST_REV_ENV) or None,
                    help="Beacon only: as --against, with the reference built "
                         "here first from REV's RTL (%s) under this "
                         "checkout's flow, so a change is judged against its "
                         "merge base and not only against a baseline it may "
                         "have re-recorded itself (default: $%s)"
                         % (", ".join(REF_PATHS), AGAINST_REV_ENV))
    ap.add_argument("--ref-cache", metavar="DIR",
                    help="where --against-rev keeps its reference trees, one "
                         "per revision (default: $%s, else <source>/.beacon_ref)"
                         % REF_CACHE_ENV)
    ap.add_argument("--measure-only", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--configs", metavar="FLAGS", default="",
                    help="extra -D flags appended to every selected build's "
                         "configs, for a non-default configuration: run the "
                         "reference and the change (--against) with the same "
                         "flags. The catalog baselines were recorded without "
                         "them, so they are not gated, and none is recorded. "
                         "Pass as --configs=\"-D...\" (the value starts with -)")
    ap.add_argument("--xlen", type=int, default=32, choices=(32, 64),
                    help="ISA width of the builds that do not pin their own "
                         "(default: 32)")
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
    if args.threshold is None:
        args.threshold = tool.tolerance

    # Line-buffer stdout: progress is the point of the elaboration watch, and it
    # is worthless if it sits in a block buffer until the run ends (which is what
    # happens whenever this is piped -- CI, nohup, a log file).
    sys.stdout.reconfigure(line_buffering=True)

    overrides = dict(METRIC_TOLERANCE, **tool.metric_tolerance)
    for spec in args.metric_threshold:
        metric, _, value = spec.partition("=")
        if metric not in tool.metrics or not value:
            sys.exit("ERROR: --metric-threshold expects <metric>=<fraction>, "
                     "metric one of: %s" % ", ".join(tool.metrics))
        overrides[metric] = float(value)
    args.metric_threshold = overrides

    root = source_root()
    builds = load_catalog(root, tool, args.xlen, args.baseline_dir)

    if args.matrix:
        # Ordered longest-first so the slowest DUT is dispatched into the fan-out
        # first; GitHub starts matrix jobs in order, so this is the same makespan
        # argument run_all() makes for its thread pool.
        print(json.dumps(sorted(
            [{"id": b["id"], "group": b["group"], "dut": b["dut"],
              "clock_mhz": b["clock_mhz"], "xlen": b["xlen"],
              "known_issue": bool(b.get("known_issue"))}
             for b in select(builds, args.build)],
            key=lambda b: (-weight(builds[b["id"]], tool), b["id"]))))
        return 0

    if args.list:
        for b in select(builds, args.build):
            base = b.get("baseline") or {}
            print("%-10s [%s] dut=%-8s %4d MHz  xlen %d%s%s  %s%s"
                  % (b["id"], b["group"], b["dut"], b["clock_mhz"], b["xlen"],
                     "  profile " + b["profile"] if b["profile"] else "",
                     "  (opt-in)" if b["opt_in"] else "",
                     "baseline: " + tool.describe_baseline(base)
                     if base.get(tool.key_metric) is not None
                     else "no baseline",
                     "  KNOWN ISSUE: " + b["known_issue"]
                     if b.get("known_issue") else ""))
            print("    %s" % b["configs"])
        return 0

    tool.check_env()
    if args.measure_only:
        args.against = args.against_rev = None
    if (args.against or args.against_rev) and not tool.name.startswith("beacon"):
        sys.exit("ERROR: --against compares Beacon runs; use it with --tool=beacon "
                 "or --tool=beacon_asic")
    if args.configs and args.update_baseline:
        sys.exit("ERROR: --configs cannot record a baseline: the catalog "
                 "baselines are the catalog configurations")

    if args.jobs < 1:
        sys.exit("ERROR: --jobs must be >= 1")
    if not args.tool_jobs:
        args.tool_jobs = max(MIN_TOOL_JOBS,
                             min(MAX_TOOL_JOBS, os.cpu_count() // args.jobs))

    env = tool.env(args)

    selected = select(builds, args.build)
    if args.configs:
        for b in selected:
            b["configs"] = (b["configs"] + " " + args.configs).strip()
            b["baseline"] = {}
    for b in selected:
        warning = tool.env_warning(b, env)
        if warning:
            print("WARNING: %s %s" % (b["id"], warning))

    # One configured build tree per ISA width: a build that pins its own xlen
    # runs in that width's tree.
    xlens = sorted({b["xlen"] for b in selected})
    if len(xlens) > 1 and (args.build_dir or args.against or args.against_rev):
        sys.exit("ERROR: --build-dir/--against name one build tree, and the "
                 "selected builds span xlen %s; select one width"
                 % "/".join(map(str, xlens)))
    if args.against and not os.path.isdir(args.against):
        print("ERROR: --against %s is not a build tree" % args.against)
        return 2
    # ref_lock stays held (shared) until the run ends, while the reference is read.
    against_sha, absent, ref_rc, ref_lock = None, set(), 0, None
    if args.against_rev and not args.dry_run:
        try:
            args.against, against_sha, absent, ref_rc, ref_lock = reference_tree(
                root, args.against_rev, args, tool, selected, xlens[0])
        except (RefError, subprocess.CalledProcessError) as e:
            print("ERROR: reference %s: %s" % (args.against_rev, e))
            return 2
    results = []
    for xlen in xlens:
        build_dir = args.build_dir or os.path.join(
            root, "build%d_%s" % (xlen, tool.gate))
        if not args.dry_run:
            prepare_build_tree(root, build_dir, xlen)
        results += run_all([b for b in selected if b["xlen"] == xlen],
                           build_dir, env, args, tool)
    if args.dry_run:
        return 0
    results.sort(key=lambda r: r["build"]["id"])

    gated = tuple(m for m in (args.gate.split(",") if args.gate else tool.gated) if m)
    mode = ("measure" if args.measure_only else
            "record" if args.update_baseline else "gate")
    report(results, env, args, gated, tool, mode)
    if args.against:
        compare_against(results, args.against, tool, env, root, against_sha,
                        absent)
        if mode == "gate":
            merge_against(results)

    ok = [r for r in results if r["metrics"]]
    if args.update_baseline and ok:
        save_baselines(root, tool, ok, env, args.baseline_dir)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"tool": tool.name, "env": env,
                       "tolerance": args.threshold,
                       "metric_thresholds": args.metric_threshold,
                       "gated": list(gated),
                       "against": against_sha or args.against,
                       "builds": [report_entry(r, env, tool)
                                  for r in results]}, fh, indent=2)
            fh.write("\n")
        print("\nwrote %s" % args.report)

    if mode != "gate":
        # Recording IS the intent; a delta is what we just wrote down. A build
        # that never produced metrics is still a failure, `known_issue` or not:
        # there is nothing to record, so excusing it would end the run claiming
        # a baseline it did not write. Same reasoning as the BUILD-FAIL verdict
        # in gate().
        broken = [r for r in results if not r["metrics"]]
        return 2 if broken or ref_rc else 0

    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    failures = [r for r in results if r["verdict"] not in PASSING]
    print("\n%s: %d builds%s%s -- %s"
          % (tool.gate, len(results),
             "".join(", %d %s" % (n, v) for v, n in sorted(counts.items())),
             ", the reference run failed (exit %d)" % ref_rc if ref_rc else "",
             "FAILED" if failures or ref_rc else "passed"))
    # A build error on either side means the change was not gated.
    if ref_rc or any(r["verdict"] in ("BUILD-FAIL", "REF-FAIL") for r in failures):
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
