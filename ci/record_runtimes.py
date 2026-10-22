#!/usr/bin/env python3

# Copyright © 2019-2026
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Record per-case CI runtimes into ci/runtimes.json from a GitHub Actions run.

Reads the junit report every test cell of the run uploaded -- the wall time the
CI runners measured, never a local estimate -- and updates the per-xlen table
the planner splits long cells with (testcase.shard_plan). Cases the run did not
report keep their recorded time, so a run with a cancelled cell does not erase
that cell's history.

  ci/record_runtimes.py --run 37241939097 [--repo vortexgpgpu/vortex]

Needs an authenticated `gh` CLI.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402

# junit-<cell>-<driver>-<xlen>[-<k>of<n>], as ci.yml's tests job names them.
ARTIFACT_RE = re.compile(r"^junit-.+-(32|64)(?:-\d+of\d+)?$")
CASE_RE = re.compile(r"^test_case\[(.+)\]$")


def gh_api(path, raw=False):
    out = subprocess.run(["gh", "api", path], capture_output=True, check=True)
    return out.stdout if raw else json.loads(out.stdout)


def artifacts(repo, run):
    page, found = 1, []
    while True:
        batch = gh_api("repos/{}/actions/runs/{}/artifacts?per_page=100&page={}"
                       .format(repo, run, page))["artifacts"]
        found += batch
        if len(batch) < 100:
            return found
        page += 1


def case_times(blob):
    """{case id: seconds} from one junit artifact zip."""
    times = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            for case in ET.fromstring(zf.read(name)).iter("testcase"):
                m = CASE_RE.match(case.get("name", ""))
                if m and case.find("skipped") is None:
                    times[m.group(1)] = round(float(case.get("time", 0)))
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="GitHub Actions run id of a CI run")
    p.add_argument("--repo", default="vortexgpgpu/vortex")
    args = p.parse_args()

    run = gh_api("repos/{}/actions/runs/{}".format(args.repo, args.run))
    try:
        with open(tc.RUNTIMES_FILE) as fh:
            table = json.load(fh)
    except OSError:
        table = {}

    updated = 0
    for art in artifacts(args.repo, args.run):
        m = ARTIFACT_RE.match(art["name"])
        if not m or art.get("expired"):
            continue
        blob = gh_api("repos/{}/actions/artifacts/{}/zip".format(args.repo, art["id"]),
                      raw=True)
        times = case_times(blob)
        table.setdefault(m.group(1), {}).update(times)
        updated += len(times)

    table["source"] = {"run": int(args.run), "sha": run["head_sha"]}
    with open(tc.RUNTIMES_FILE, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("recorded {} case times from run {} ({})".format(
        updated, args.run, run["head_sha"][:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
