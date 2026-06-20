#!/usr/bin/env python3
"""Vortex test cases — the model and the CLI.

Loads ci/testcases/*.yaml into concrete test cases, and exposes the planner CLI
(`lint` / `matrix` / `select`) the CI workflow uses. Pure logic with no pytest
dependency (only PyYAML), so the lightweight ci.yml plan job and the pytest
harness (ci/test_runner.py) both build on it. See docs/designs/continuous_integration.md.

A *case* (class Spec) is one declarative test. A testcases file lists entries per
category; an entry with `drivers: [...]` expands to one case per driver. `xlen` is
an outer dimension — cases are filtered against the ambient build tree's XLEN,
never expanded here (build32/ and build64/ are separate trees).

  testcase.py lint
      validate every case; exit non-zero on any error
  testcase.py matrix [--drivers=simx,rtlsim] [--tier=fast,smoke] [--xlen=32] [--changed-from=REF]
      JSON (category x driver x xlen) cells for the GitHub matrix
  testcase.py select --changed-from=REF
      categories whose touches[] intersect the diff (path-scaling)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

TESTCASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcases")

# Execution driver name (matches blackbox.sh --driver= and `make run-<d>`) ->
# user-facing marker / slice name (what --drivers="...,xrtsim,opaesim" selects).
_DRIVER_TO_MARKER = {"xrt": "xrtsim", "opae": "opaesim"}
# Execution driver name -> its sim source directory under sim/.
_DRIVER_TO_SIMDIR = {"simx": "simx", "rtlsim": "rtlsim", "xrt": "xrtsim", "opae": "opaesim"}
VALID_DRIVERS = set(_DRIVER_TO_SIMDIR)
VALID_VIA = {"blackbox", "make-run", "script"}


def driver_marker(driver):
    """User-facing marker name for an execution driver (xrt->xrtsim, ...)."""
    return _DRIVER_TO_MARKER.get(driver, driver)


class Spec:
    """One concrete test case: a single (category, driver, configs, shape) point."""

    def __init__(self, category, entry, driver, defaults):
        self.category = category
        self.driver = driver
        self.via = entry.get("via", "blackbox")
        self.app = entry.get("app", "")
        self.args = entry.get("args", "")
        # Verbatim extra blackbox flags the structured fields don't model
        # (e.g. --debug=3 --perf=6 --scope --nohup --log=...).
        self.flags = entry.get("flags", "")
        self.shape = dict(entry.get("shape", {}))
        self.tier = entry.get("tier", defaults.get("tier", "fast"))
        self.needs = list(entry.get("needs", defaults.get("needs", [])))
        self.touches = list(entry.get("touches", defaults.get("touches", [])))
        self.xlens = [int(x) for x in entry.get("xlen", defaults.get("xlen", [32, 64]))]
        self.configs = _merge_configs(defaults.get("configs", ""), entry)
        # make-run / script fields
        self.dir = entry.get("dir", "")
        self.target = entry.get("target", "")
        self.run = entry.get("run", "")
        self.vars = dict(entry.get("vars", {}))
        # Stable, unique id: <category>:<authored-id>:<marker-driver>
        self.id = "{}:{}:{}".format(category, entry["id"], self.marker_driver)

    @property
    def marker_driver(self):
        return driver_marker(self.driver) if self.driver else None

    @property
    def needs_sim(self):
        """Whether a sim build is required (driverless via:script cases self-build)."""
        return self.via != "script" and self.driver is not None

    @property
    def sim_dir(self):
        return "sim/" + _DRIVER_TO_SIMDIR[self.driver]

    def build_key(self):
        """What determines a *sim build*. `via` is deliberately excluded so a
        make-run case and a blackbox case with the same (driver, configs) share
        one sim build. xlen is implicit in the ambient tree.
        """
        return (self.driver, self.configs)

    def markers(self):
        """pytest marker names for `-m` selection (one per value)."""
        m = [self.category, self.tier]
        if self.marker_driver:
            m.append(self.marker_driver)
        m += ["needs_{}".format(n) for n in self.needs]
        return m

    def applies_to_xlen(self, xlen):
        return int(xlen) in self.xlens

    def build_command(self, xlen):
        """argv + env to build this case's sim once (shared across build_key)."""
        env = {"CONFIGS": _subst(self.configs, xlen)} if self.configs else {}
        return ["make", "-C", self.sim_dir], env

    def run_command(self, xlen):
        """argv + env to execute this case at the given ambient xlen."""
        env = {"CONFIGS": _subst(self.configs, xlen)} if self.configs else {}
        if self.via == "blackbox":
            argv = ["./ci/blackbox.sh", "--driver=" + self.driver, "--app=" + self.app]
            argv += _shape_flags(self.shape)
            if self.args:
                argv.append("--args=" + self.args)
            if self.flags:
                argv += self.flags.split()
            return argv, env
        if self.via == "make-run":
            target = self.target.format(driver=self.driver, xlen=xlen)
            argv = ["make", "-C", self.dir, target]
            argv += ["{}={}".format(k, v) for k, v in self.vars.items()]
            return argv, env
        if self.via == "script":
            # run through a shell so multi-step `cmd1 && cmd2` scripts work.
            return ["bash", "-c", _subst(self.run, xlen)], env
        raise ValueError("unknown via: {!r}".format(self.via))


def _subst(text, xlen):
    """Resolve the ambient-xlen placeholders in a config/run string:
    {xlen} -> 32/64, {xsize} -> XLEN/8 (the legacy $XLEN / $XSIZE)."""
    return text.replace("{xlen}", str(xlen)).replace("{xsize}", str(int(xlen) // 8))


def _merge_configs(default, entry):
    """`configs` overrides the default; `configs+` appends to it."""
    if "configs" in entry:
        return entry["configs"]
    if "configs+" in entry:
        return (default + " " + entry["configs+"]).strip()
    return default


def _shape_flags(shape):
    flags = []
    for knob in ("clusters", "cores", "warps", "threads"):
        if shape.get(knob):
            flags.append("--{}={}".format(knob, shape[knob]))
    for boolean in ("l2cache", "l3cache", "scope"):
        if shape.get(boolean):
            flags.append("--" + boolean)
    return flags


def load_category(path):
    """Expand one testcases YAML file into concrete cases."""
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    category = doc["category"]
    defaults = doc.get("defaults", {})
    cases = []
    for entry in doc.get("tests", []):
        # via:script cases may be driverless (host/synthesis); everything else
        # has a driver or a drivers list.
        drivers = entry.get("drivers") or ([entry["driver"]] if "driver" in entry else [None])
        for driver in drivers:
            cases.append(Spec(category, entry, driver, defaults))
    return cases


def load_all(testcases_dir=TESTCASES_DIR):
    """Load every ci/testcases/*.yaml into a flat list of concrete cases."""
    cases = []
    for path in sorted(glob.glob(os.path.join(testcases_dir, "*.yaml"))):
        cases.extend(load_category(path))
    return cases


def execute(argv, env_extra=None, cwd=None):
    """Run argv with CONFIGS et al. merged into the environment; return exit code."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(argv, env=env, cwd=cwd).returncode


# --------------------------------------------------------------------------- #
# Planner CLI — lint / matrix / select. Reads the same data the harness runs,
# but needs no pytest or build env, so the ci.yml plan job can call it.
# --------------------------------------------------------------------------- #

def _changed_files(ref):
    out = subprocess.run(["git", "diff", "--name-only", ref + "...HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:  # fall back to a two-dot diff if no merge base
        out = subprocess.run(["git", "diff", "--name-only", ref],
                             capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def _touched(case, changed):
    return any(f.startswith(prefix) for prefix in case.touches for f in changed)


def _filter(cases, args):
    drivers = set(args.drivers.split(",")) if getattr(args, "drivers", None) else None
    tiers = set(args.tier.split(",")) if getattr(args, "tier", None) else None
    changed = _changed_files(args.changed_from) if getattr(args, "changed_from", None) else None
    out = []
    for c in cases:
        if drivers and c.marker_driver not in drivers:
            continue
        if tiers and c.tier not in tiers:
            continue
        if changed is not None and not _touched(c, changed):
            continue
        out.append(c)
    return out


def cmd_matrix(args):
    # One GitHub matrix cell per (category, driver, xlen): the build tree is
    # per-xlen, so xlen is flattened out (not a per-cell list).
    xfilter = {int(x) for x in args.xlen.split(",")} if getattr(args, "xlen", None) else None
    cells = {}
    for c in _filter(load_all(), args):
        drv = c.marker_driver or "host"
        for xlen in c.xlens:
            if xfilter and xlen not in xfilter:
                continue
            key = (c.category, drv, xlen)
            cell = cells.setdefault(key, {
                "category": c.category, "driver": drv, "xlen": xlen, "needs": set(),
            })
            cell["needs"].update(c.needs)
    out = []
    for cell in cells.values():
        cell["needs"] = sorted(cell["needs"])
        out.append(cell)
    out.sort(key=lambda c: (c["category"], c["driver"], c["xlen"]))
    print(json.dumps(out))
    return 0


def cmd_select(args):
    print(" ".join(sorted({c.category for c in _filter(load_all(), args)})))
    return 0


def cmd_lint(args):
    cases = load_all()
    errors, seen = [], {}
    for c in cases:
        if c.id in seen:
            errors.append("duplicate id: {}".format(c.id))
        seen[c.id] = c
        if c.via not in VALID_VIA:
            errors.append("{}: invalid via {!r}".format(c.id, c.via))
        if c.driver is not None and c.driver not in VALID_DRIVERS:
            errors.append("{}: invalid driver {!r}".format(c.id, c.driver))
        if c.via == "blackbox" and not c.app:
            errors.append("{}: blackbox case missing 'app'".format(c.id))
        if c.via == "make-run" and not (c.dir and c.target):
            errors.append("{}: make-run case needs 'dir' and 'target'".format(c.id))
        if c.via == "script" and not c.run:
            errors.append("{}: script case missing 'run'".format(c.id))
        if any(int(x) not in (32, 64) for x in c.xlens):
            errors.append("{}: xlen must be 32 and/or 64".format(c.id))
    if errors:
        for e in errors:
            sys.stderr.write("LINT ERROR: " + e + "\n")
        return 1
    print("OK: {} test cases across {} categories".format(
        len(cases), len({c.category for c in cases})))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Vortex test-case planner")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("matrix", help="emit JSON (category x driver x xlen) cells")
    m.add_argument("--drivers")
    m.add_argument("--tier")
    m.add_argument("--xlen")
    m.add_argument("--changed-from", dest="changed_from")
    m.set_defaults(func=cmd_matrix)

    s = sub.add_parser("select", help="categories whose touches[] hit a diff")
    s.add_argument("--drivers")
    s.add_argument("--tier")
    s.add_argument("--changed-from", dest="changed_from")
    s.set_defaults(func=cmd_select)

    sub.add_parser("lint", help="validate the test cases").set_defaults(func=cmd_lint)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
