#!/usr/bin/env python3
"""Print the newest commit HEAD descends from whose CI run passed.

A push plans its sim drivers from the paths changed since this commit, not
since the previous push. A run that failed, was cancelled or never reached a
driver verified nothing, so the changes it carried stay in the diff until a run
passes. Only push and schedule runs count: a workflow_dispatch run may have
narrowed the drivers. Only commits that carry this file count too: an older run
was planned from the previous push and may have skipped a driver a change
needed.

Prints nothing when no such run is found or the API is unavailable; the caller
treats that as an indeterminate diff and plans every driver.
"""

import argparse
import json
import subprocess
import sys
import urllib.parse

POLICY_FILE = "ci/last_green.py"
COUNTED_EVENTS = ("push", "schedule")


def _successful_runs(repo, workflow, branch, limit):
    path = "repos/{}/actions/workflows/{}/runs?branch={}&status=success&per_page={}".format(
        repo, workflow, urllib.parse.quote(branch, safe=""), limit)
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write("last_green: {}: {}\n".format(path, out.stderr.strip()))
        return None
    return json.loads(out.stdout).get("workflow_runs", [])


def _git_ok(*args):
    return subprocess.run(["git"] + list(args), capture_output=True).returncode == 0


def last_green(runs):
    """The head commit of the newest counted run that HEAD descends from."""
    for run in sorted(runs, key=lambda r: r["created_at"], reverse=True):
        sha = run["head_sha"]
        if run["event"] not in COUNTED_EVENTS:
            continue
        if not _git_ok("merge-base", "--is-ancestor", sha, "HEAD"):
            continue
        if not _git_ok("cat-file", "-e", "{}:{}".format(sha, POLICY_FILE)):
            continue
        return sha
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True)
    p.add_argument("--workflow", default="ci.yml")
    p.add_argument("--branch", action="append", required=True,
                   help="branch whose runs count; repeat for the default branch")
    p.add_argument("--limit", type=int, default=50, help="runs read per branch")
    args = p.parse_args(argv)

    runs = []
    for branch in dict.fromkeys(args.branch):
        got = _successful_runs(args.repo, args.workflow, branch, args.limit)
        if got is None:
            return 0
        runs += got
    sha = last_green(runs)
    if sha:
        print(sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
