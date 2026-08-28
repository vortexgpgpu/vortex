#!/usr/bin/env python3
"""Shape x epilogue sweep with serial builds and parallel simulations.

Each epilogue is a compile-time variant. Build it once under the shared lock, snapshot the
executable, kernels, and SimX libraries, then run every shape/mode in parallel.
"""

import argparse
import concurrent.futures
import csv
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


BUILD_DIR = Path(os.environ.get(
    "VX_BUILD", "/nethome/sjeong306/vortex_scheduler/vortex/build")).resolve()
SOURCE_DIR = Path(__file__).resolve().parent
TEST_DIR = BUILD_DIR / "tests/regression/cgo27_motivation"
RUNTIME_DIR = BUILD_DIR / "sw/runtime"

APPS = {
    1: "baseline", 2: "relu", 5: "scale", 4: "residual", 3: "gelu",
    6: "softmax_row", 9: "mean_broadcast_col",
}
MODES = {
    0: "SIMT", 1: "TCU", 2: "TCU+DXA", 3: "TCU_wg+DXA", 4: "TCU_wg",
    5: "TCU_wg+Acol", 6: "TCU_wg+Acol_SB", 7: "DTCU_socket", 8: "DTCU_cluster",
    14: "DTCU_socket_pipe", 15: "DTCU_cluster_pipe",
}
DEFAULT_MODES = (1, 2, 3, 5, 6, 7, 8, 14, 15)
# Some epilogues become long-tail cases at larger shapes (including ReLU in the largest
# cube due to simulator path sensitivity). Running apps 2/4/6 once in every
# mode-homogeneous batch serialises independent long jobs. Batch them across modes after
# the regular batches.
CROSS_MODE_APPS = (2, 4, 6)

MOTI_RE = re.compile(
    r"\[MOTI\]\s+app=(?P<app>\d+)\s+M=(?P<M>\d+)\s+N=(?P<N>\d+)\s+K=(?P<K>\d+)\s+"
    r"mode=(?P<mode>\d+)\s+name=(?P<name>\S+)\s+cycles=(?P<cycles>\d+)\s+"
    r"errors=(?P<errors>-?\d+)(?:\s+skipped=(?P<skipped>\d+))?"
)


def parse_ids(text, valid, what):
    values = tuple(int(x) for x in text.split(",") if x)
    bad = [x for x in values if x not in valid]
    if bad:
        raise SystemExit(f"unsupported {what}: {bad}; choose from {tuple(valid)}")
    return values


def shapes_for(family, values):
    if family == "cube":
        # Even rungs keep K >= 32 and N compatible with mode 5's four-column CTA.
        return [(f"r{r}", 64 * r, 32 * r, 16 * r) for r in values]
    if family == "attention":
        return [(f"s{s}", s, s, 64) for s in values]
    if family == "ffn":
        return [(f"h{h}", 128, 4 * h, h) for h in values]
    raise AssertionError(family)


def parse_explicit_shapes(values):
    """Parse repeatable TAG:M:N:K shapes without tying a study to a built-in family."""
    shapes = []
    tags = set()
    for value in values:
        fields = value.split(":")
        if len(fields) != 4 or not fields[0]:
            raise SystemExit(f"invalid --shape '{value}'; expected TAG:M:N:K")
        tag = fields[0]
        try:
            M, N, K = (int(field) for field in fields[1:])
        except ValueError as exc:
            raise SystemExit(f"invalid --shape '{value}'; M/N/K must be integers") from exc
        if min(M, N, K) <= 0:
            raise SystemExit(f"invalid --shape '{value}'; M/N/K must be positive")
        if tag in tags:
            raise SystemExit(f"duplicate --shape tag '{tag}'")
        tags.add(tag)
        shapes.append((tag, M, N, K))
    return shapes


def run_checked(cmd, *, cwd, env, timeout):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return output


def build_and_snapshot(app, work_root, timeout, bootstrap_runtime):
    env = dict(os.environ)
    # DEBUG is a make variable (an integer trace level), not a release/debug label.
    # IDE shells sometimes export DEBUG=release; forwarding that would compile the
    # simulator with the invalid token VX_DBG_DEBUG_LEVEL=release.
    env.pop("DEBUG", None)
    env["MOTI_APP"] = str(app)

    # The build tree owns a configure-time Makefile copy. Refresh only that file;
    # configure would remove this machine's pinned host-compatible LLVM path.
    shutil.copy2(SOURCE_DIR / "Makefile", TEST_DIR / "Makefile")
    if bootstrap_runtime:
        # Configure/build the shared simulator once. MOTI_APP is irrelevant to SimX but
        # blackbox propagates it in CONFIGS, so doing this for every app needlessly
        # recompiles the whole simulator.
        cmd = [
            "./ci/blackbox.sh", "--driver=simx", "--app=cgo27_motivation",
            "--perf=1", f"--args=-a {app} -m 1 -M 128 -N 64 -K 32",
        ]
        output = run_checked(cmd, cwd=BUILD_DIR, env=env, timeout=timeout)
    else:
        # Only the host and per-mode device binaries contain the compile-time epilogue.
        run_checked(["make", "all"], cwd=TEST_DIR, env=env, timeout=timeout)
        smoke_env = dict(env)
        smoke_env["VORTEX_DRIVER"] = "simx"
        smoke_env["LD_LIBRARY_PATH"] = str(RUNTIME_DIR) + (
            ":" + smoke_env["LD_LIBRARY_PATH"]
            if smoke_env.get("LD_LIBRARY_PATH") else "")
        output = run_checked(
            ["./cgo27_motivation", "-a", str(app), "-m", "1",
             "-M", "128", "-N", "64", "-K", "32"],
            cwd=TEST_DIR, env=smoke_env, timeout=timeout)
    if "PASSED!" not in output:
        raise RuntimeError(f"app {app} build smoke did not pass")

    snapshot = work_root / f"app{app}"
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copy2(TEST_DIR / "cgo27_motivation", snapshot)
    for mode in MODES:
        kernel = TEST_DIR / f"kernel_m{mode}.vxbin"
        if kernel.exists():
            shutil.copy2(kernel, snapshot)
    for pattern in ("libsimx.so", "libvortex*.so"):
        for library in RUNTIME_DIR.glob(pattern):
            shutil.copy2(library, snapshot)
    return snapshot


def run_point(snapshot, app, shape, mode, timeout):
    tag, M, N, K = shape
    env = dict(os.environ)
    env["VORTEX_DRIVER"] = "simx"
    env["LD_LIBRARY_PATH"] = str(snapshot) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    cmd = [
        "./cgo27_motivation", "-a", str(app), "-m", str(mode),
        "-M", str(M), "-N", str(N), "-K", str(K),
    ]
    output = run_checked(cmd, cwd=snapshot, env=env, timeout=timeout)
    match = MOTI_RE.search(output)
    if not match:
        raise RuntimeError(f"missing [MOTI] line for app={app} {tag} mode={mode}")
    got = match.groupdict()
    if int(got["app"]) != app or int(got["mode"]) != mode:
        raise RuntimeError(f"result identity mismatch for app={app} {tag} mode={mode}")
    if got["name"] != MODES[mode]:
        raise RuntimeError(f"mode {mode}: binary says {got['name']}, script says {MODES[mode]}")
    if got["skipped"]:
        raise RuntimeError(f"mode {mode} was skipped in a supposedly complete snapshot")
    errors = int(got["errors"])
    if errors != 0 or "PASSED!" not in output:
        raise RuntimeError(f"incorrect result: app={app} {tag} mode={mode} errors={errors}")
    return {
        "app": app, "app_name": APPS[app], "shape": tag,
        "M": M, "N": N, "K": K, "mode": mode,
        "mode_name": MODES[mode], "cycles": int(got["cycles"]), "errors": errors,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("cube", "attention", "ffn"), default="cube")
    parser.add_argument("--sizes", default=None,
                        help="cube rungs, attention sequence lengths, or FFN hidden sizes")
    parser.add_argument("--shape", action="append", default=[], metavar="TAG:M:N:K",
                        help="explicit shape; repeatable and mutually exclusive with --sizes")
    parser.add_argument("--apps", default=",".join(map(str, APPS)))
    parser.add_argument("--modes", default=",".join(map(str, DEFAULT_MODES)))
    # SimX uses two busy host threads per process.  This server scales cleanly through
    # eight mixed-mode runs, while 16 mixed modes already delay the WGMMA straggler and
    # 32 mixed-size runs make a one-minute case take more than 25 minutes from LLC/memory
    # contention.  Keep the measured safe default; callers can still override it after
    # benchmarking their own host.
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--out", default="exp1_results.csv")
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--snapshot-root", type=Path,
                        help="reuse previously built app<N> snapshots and skip all builds")
    parser.add_argument("--resume-from", type=Path,
                        help="reuse verified rows from an earlier CSV or .partial file")
    parser.add_argument("--rerun-apps", default="",
                        help="comma-separated app ids to ignore in --resume-from")
    parser.add_argument("--cross-mode-apps", default=",".join(map(str, CROSS_MODE_APPS)),
                        help="apps batched across modes after regular mode batches")
    parser.add_argument("--cross-mode-flat", action="store_true",
                        help="queue all cross-mode apps together per shape so one long mode "
                             "does not block later apps")
    parser.add_argument("--flat", action="store_true",
                        help="queue every pending shape/app/mode point in one worker pool")
    args = parser.parse_args()

    defaults = {"cube": "2,4,8,12,16", "attention": "128,256,512,1024",
                "ffn": "256,512,1024"}
    if args.shape and args.sizes is not None:
        raise SystemExit("--shape and --sizes are mutually exclusive")
    apps = parse_ids(args.apps, APPS, "apps")
    modes = parse_ids(args.modes, MODES, "modes")
    rerun_apps = (parse_ids(args.rerun_apps, APPS, "rerun apps")
                  if args.rerun_apps else ())
    cross_mode_apps = (parse_ids(args.cross_mode_apps, APPS, "cross-mode apps")
                       if args.cross_mode_apps else ())
    if args.shape:
        shapes = parse_explicit_shapes(args.shape)
    else:
        values = tuple(int(x) for x in (args.sizes or defaults[args.family]).split(","))
        shapes = shapes_for(args.family, values)
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")

    owns_work_root = args.snapshot_root is None
    work_root = (Path(tempfile.mkdtemp(prefix="cgo27-exp1-"))
                 if owns_work_root else args.snapshot_root.resolve())
    try:
        snapshots = {}
        if owns_work_root:
            lock_path = TEST_DIR / ".moti-build.lock"
            lock_path.touch(exist_ok=True)
            with lock_path.open("r+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise SystemExit("another cgo27 build holds .moti-build.lock") from exc
                for index, app in enumerate(apps):
                    print(f"[build] app={app} ({APPS[app]})", flush=True)
                    snapshots[app] = build_and_snapshot(
                        app, work_root, args.timeout, bootstrap_runtime=(index == 0))
        else:
            print(f"[reuse] snapshots={work_root}", flush=True)
            for app in apps:
                snapshot = work_root / f"app{app}"
                required = [snapshot / "cgo27_motivation"] + [
                    snapshot / f"kernel_m{mode}.vxbin" for mode in modes]
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise SystemExit(f"snapshot app {app} is incomplete: {missing}")
                snapshots[app] = snapshot

        rows = []
        fields = ("app", "app_name", "shape", "M", "N", "K",
                  "mode", "mode_name", "cycles", "errors", "status")
        wanted_shapes = {tag: (M, N, K) for tag, M, N, K in shapes}
        completed = set()
        if args.resume_from:
            with args.resume_from.open(newline="") as stream:
                for old in csv.DictReader(stream):
                    app = int(old["app"])
                    mode = int(old["mode"])
                    tag = old["shape"]
                    dims = (int(old["M"]), int(old["N"]), int(old["K"]))
                    key = (app, tag, mode)
                    old_status = old.get("status") or "ok"
                    old_errors = old.get("errors", "")
                    reusable = (old_status == "timeout" or
                                (old_status == "ok" and int(old_errors) == 0))
                    if (app in apps and app not in rerun_apps and mode in modes
                            and wanted_shapes.get(tag) == dims and reusable
                            and key not in completed):
                        row = dict(old)
                        row["status"] = old_status
                        for name in ("app", "M", "N", "K", "mode"):
                            row[name] = int(row[name])
                        for name in ("cycles", "errors"):
                            if row.get(name, "") != "":
                                row[name] = int(row[name])
                        rows.append(row)
                        completed.add(key)
            print(f"[resume] verified rows={len(rows)} from {args.resume_from}", flush=True)
        partial_path = Path(str(args.out) + ".partial")
        with open(partial_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        total_tasks = len(apps) * len(shapes) * len(modes)
        print(f"[run] {total_tasks - len(completed)}/{total_tasks} simulations "
              f"with jobs={args.jobs}", flush=True)
        def run_batch(tasks, label):
            if not tasks:
                return
            print(f"[batch] {label} simulations={len(tasks)}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(args.jobs, len(tasks))) as pool:
                future_to_task = {
                    pool.submit(run_point, snap, app, one_shape, one_mode,
                                args.timeout): (app, one_shape, one_mode)
                    for snap, app, one_shape, one_mode in tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    app, task_shape, one_mode = future_to_task[future]
                    shape_tag, M, N, K = task_shape
                    try:
                        row = future.result()
                    except subprocess.TimeoutExpired:
                        row = {
                            "app": app, "app_name": APPS[app], "shape": shape_tag,
                            "M": M, "N": N, "K": K, "mode": one_mode,
                            "mode_name": MODES[one_mode], "cycles": "", "errors": "",
                            "status": "timeout",
                        }
                    rows.append(row)
                    # Preserve every verified result immediately. A single SimX case
                    # can run for hours, so a later timeout or interrupted shell must
                    # not discard the completed prefix of the sweep.
                    with open(partial_path, "a", newline="") as stream:
                        csv.DictWriter(stream, fieldnames=fields).writerow(row)
                    print(f"  app={app} shape={shape_tag} mode={one_mode:<2} "
                          f"cycles={row['cycles'] or '-'} status={row['status']}", flush=True)

        # Real batch barriers between both sizes AND modes are load-bearing for throughput.
        # Merely enqueueing size-major work is insufficient: as soon as a fast r2 future
        # ends, the executor admits r4 while r2's WGMMA straggler is still live.  Even at
        # one size, mixing WGMMA with the DTCU pipeline turns a ~69-second homogeneous
        # WGMMA batch into 3+ minutes.  Each pool therefore runs one shape/mode across the
        # seven apps; all concurrent SimX processes execute the same kernel over the same
        # size working set. Apps 2/4/6 are the exception: at large shapes their long tails
        # dominate, so running their modes concurrently avoids making the same app the
        # serial straggler of successive batches.
        if args.flat:
            tasks = [(snapshots[app], app, shape, mode)
                     for shape in reversed(shapes)
                     for mode in modes
                     for app in apps
                     if (app, shape[0], mode) not in completed]
            run_batch(tasks, "flat heavy-first")
        else:
            for shape in shapes:
                for mode in modes:
                    tasks = [(snapshots[a], a, shape, mode) for a in apps
                             if a not in cross_mode_apps
                             and (a, shape[0], mode) not in completed]
                    run_batch(tasks, f"shape={shape[0]} mode={mode}")
            # Only after every regular cell is checkpointed do we admit the long-tail apps.
            # A six-hour timeout in one of them must not prevent useful later shapes from ever
            # entering the queue.
            for shape in shapes:
                if args.cross_mode_flat:
                    # Mode-major ordering keeps one pathological epilogue from occupying
                    # every worker. The executor still mixes modes, but its first wave has
                    # at most one task per app for a given mode before proceeding through
                    # the remaining mode/app grid.
                    tasks = [(snapshots[app], app, shape, mode)
                             for mode in modes
                             for app in apps if app in cross_mode_apps
                             and (app, shape[0], mode) not in completed]
                    run_batch(tasks, f"shape={shape[0]} flat cross-mode")
                else:
                    for app in apps:
                        if app not in cross_mode_apps:
                            continue
                        tasks = [(snapshots[app], app, shape, mode) for mode in modes
                                 if (app, shape[0], mode) not in completed]
                        run_batch(tasks, f"shape={shape[0]} app={app} cross-mode")

        rows.sort(key=lambda r: (r["app"], r["M"], r["N"], r["K"], r["mode"]))
        with open(args.out, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        print("\n===== WINNERS =====")
        for app in apps:
            for shape in shapes:
                candidates = [r for r in rows if r["app"] == app
                              and r["shape"] == shape[0] and r["status"] == "ok"]
                if candidates:
                    winner = min(candidates, key=lambda r: r["cycles"])
                    print(f"app={app:<2} {shape[0]:<6} {winner['mode_name']:<20} "
                          f"{winner['cycles']} cycles")
                else:
                    print(f"app={app:<2} {shape[0]:<6} no completed mode")
        print(f"\nCSV -> {args.out}")
    finally:
        if not owns_work_root:
            print(f"reused snapshots -> {work_root}")
        elif args.keep_snapshots:
            print(f"snapshots -> {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
