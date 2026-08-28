#!/usr/bin/env python3
"""Sweep the mode 14/15 consumer-warp width with serial builds and parallel SimX runs."""

import argparse
import concurrent.futures
import csv
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import sweep_exp1 as exp1


SUPPORTED_WIDTHS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16)
DEFAULT_WIDTHS = (1, 2, 3, 4, 8)
APPS = (2, 3, 4, 5)
MODES = (14, 15)
SHAPES = (
    ("r2", 128, 64, 32),
    ("r4", 256, 128, 64),
    ("r8", 512, 256, 128),
    ("r12", 768, 384, 192),
    ("r16", 1024, 512, 256),
    ("s256", 256, 256, 64),
    ("s512", 512, 512, 64),
    ("s1024", 1024, 1024, 64),
)
FIELDS = (
    "consumer_warps", "app", "app_name", "shape", "M", "N", "K",
    "mode", "mode_name", "cycles", "errors", "status", "failure",
)


def parse_ints(text, allowed, name):
    values = tuple(int(value) for value in text.split(",") if value)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise SystemExit(f"unsupported {name}: {invalid}; choose from {allowed}")
    if len(set(values)) != len(values):
        raise SystemExit(f"duplicate {name}: {values}")
    return values


def read_completed(path, widths, apps):
    rows = []
    completed = set()
    if not path.exists():
        return rows, completed
    wanted_shapes = {tag: (M, N, K) for tag, M, N, K in SHAPES}
    with path.open(newline="") as stream:
        for old in csv.DictReader(stream):
            width = int(old["consumer_warps"])
            app = int(old["app"])
            mode = int(old["mode"])
            tag = old["shape"]
            dims = (int(old["M"]), int(old["N"]), int(old["K"]))
            status = old.get("status") or "ok"
            errors = old.get("errors", "")
            key = (width, app, tag, mode)
            if (width not in widths or app not in apps or mode not in MODES
                    or wanted_shapes.get(tag) != dims or key in completed):
                continue
            if status != "ok" or errors == "" or int(errors) != 0:
                continue
            row = dict(old)
            for field in ("consumer_warps", "app", "M", "N", "K", "mode",
                          "cycles", "errors"):
                row[field] = int(row[field])
            row.setdefault("failure", "")
            rows.append(row)
            completed.add(key)
    return rows, completed


def snapshot_all(snapshot_root, widths, apps, timeout):
    snapshots = {}
    lock_path = exp1.TEST_DIR / ".moti-build.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another cgo27 build holds .moti-build.lock") from exc
        first = True
        for width in widths:
            width_root = snapshot_root / f"c{width}"
            for app in apps:
                print(f"[build] consumer={width} app={app} ({exp1.APPS[app]})", flush=True)
                os.environ["MOTI_PIPE_CONSUMER_WARPS"] = str(width)
                os.environ["MOTI_L2_ARB_ENGINE_BYPASS_LIMIT"] = "1000"
                snapshots[(width, app)] = exp1.build_and_snapshot(
                    app, width_root, timeout, bootstrap_runtime=first)
                first = False
    return snapshots


def restore_release(timeout):
    """Leave the shared build tree in the documented app1/c15/bypass1000 state."""
    env = dict(os.environ)
    env.pop("DEBUG", None)
    env["MOTI_APP"] = "1"
    env["MOTI_PIPE_CONSUMER_WARPS"] = "15"
    env["MOTI_L2_ARB_ENGINE_BYPASS_LIMIT"] = "1000"
    shutil.copy2(exp1.SOURCE_DIR / "Makefile", exp1.TEST_DIR / "Makefile")
    exp1.run_checked(["make", "all"], cwd=exp1.TEST_DIR, env=env, timeout=timeout)
    smoke_env = dict(env)
    smoke_env["VORTEX_DRIVER"] = "simx"
    smoke_env["LD_LIBRARY_PATH"] = str(exp1.RUNTIME_DIR) + (
        ":" + smoke_env["LD_LIBRARY_PATH"] if smoke_env.get("LD_LIBRARY_PATH") else "")
    output = exp1.run_checked(
        ["./cgo27_motivation", "-a", "1", "-m", "1",
         "-M", "128", "-N", "64", "-K", "32"],
        cwd=exp1.TEST_DIR, env=smoke_env, timeout=timeout)
    if "PASSED!" not in output:
        raise RuntimeError("release restore anchor did not pass")
    match = exp1.MOTI_RE.search(output)
    if not match:
        raise RuntimeError("release restore anchor has no [MOTI] result")
    print(f"[restore] r2 a1 m1 cycles={match.group('cycles')} errors={match.group('errors')}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--apps", default=",".join(map(str, APPS)))
    parser.add_argument("--jobs", type=int, default=min(48, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--timeout", type=int, default=43200,
                        help="per-simulation and per-build timeout in seconds")
    parser.add_argument("--out", type=Path,
                        default=Path("result/260827_data/consumer_width_sweep_20260827.csv"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--snapshot-root", type=Path,
                        help="reuse c<width>/app<app> snapshots and skip builds")
    parser.add_argument("--keep-snapshots", action="store_true")
    args = parser.parse_args()

    widths = parse_ints(args.widths, SUPPORTED_WIDTHS, "consumer widths")
    apps = parse_ints(args.apps, APPS, "apps")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(args.out) + ".partial")
    resume = args.resume_from or partial
    rows, completed = read_completed(resume, widths, apps)
    if completed:
        print(f"[resume] verified rows={len(completed)} from {resume}", flush=True)

    owns_snapshots = args.snapshot_root is None
    snapshot_root = (Path(tempfile.mkdtemp(prefix="cgo27-consumer-width-"))
                     if owns_snapshots else args.snapshot_root.resolve())
    success = False
    snapshots = {}
    try:
        if owns_snapshots:
            snapshots = snapshot_all(snapshot_root, widths, apps, args.timeout)
        else:
            for width in widths:
                for app in apps:
                    snapshot = snapshot_root / f"c{width}" / f"app{app}"
                    required = (snapshot / "cgo27_motivation",
                                snapshot / "kernel_m14.vxbin",
                                snapshot / "kernel_m15.vxbin",
                                snapshot / "libsimx.so")
                    missing = [str(path) for path in required if not path.is_file()]
                    if missing:
                        raise SystemExit(f"incomplete snapshot {width}/{app}: {missing}")
                    snapshots[(width, app)] = snapshot

        with partial.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        # Sort by GEMM work so all 40 configurations of the same heavy shape enter first.
        shapes = sorted(SHAPES, key=lambda item: item[1] * item[2] * item[3], reverse=True)
        tasks = [
            (width, app, shape, mode)
            for shape in shapes
            for width in widths
            for app in apps
            for mode in MODES
            if (width, app, shape[0], mode) not in completed
        ]
        print(f"[run] {len(tasks)}/{len(widths) * len(apps) * len(SHAPES) * len(MODES)} "
              f"simulations jobs={args.jobs} timeout={args.timeout}s", flush=True)
        failures = 0
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.jobs, max(1, len(tasks)))) as pool:
            future_to_task = {
                pool.submit(exp1.run_point, snapshots[(width, app)], app, shape, mode,
                            args.timeout): (width, app, shape, mode)
                for width, app, shape, mode in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                width, app, shape, mode = future_to_task[future]
                tag, M, N, K = shape
                try:
                    row = future.result()
                    row["failure"] = ""
                except subprocess.TimeoutExpired:
                    row = {
                        "app": app, "app_name": exp1.APPS[app], "shape": tag,
                        "M": M, "N": N, "K": K, "mode": mode,
                        "mode_name": exp1.MODES[mode], "cycles": "", "errors": "",
                        "status": "timeout", "failure": f"timeout after {args.timeout}s",
                    }
                    failures += 1
                except Exception as exc:  # Preserve the rest of a long sweep after one failure.
                    row = {
                        "app": app, "app_name": exp1.APPS[app], "shape": tag,
                        "M": M, "N": N, "K": K, "mode": mode,
                        "mode_name": exp1.MODES[mode], "cycles": "", "errors": "",
                        "status": "failed", "failure": str(exc).replace("\n", " | "),
                    }
                    failures += 1
                row = {"consumer_warps": width, **row}
                rows.append(row)
                with partial.open("a", newline="") as stream:
                    csv.DictWriter(
                        stream, fieldnames=FIELDS, lineterminator="\n").writerow(row)
                print(f"  c={width:<2} app={app} shape={tag:<5} mode={mode} "
                      f"cycles={row['cycles'] or '-'} status={row['status']}", flush=True)

        rows.sort(key=lambda row: (
            int(row["consumer_warps"]), int(row["app"]),
            next(index for index, shape in enumerate(SHAPES) if shape[0] == row["shape"]),
            int(row["mode"])))
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] rows={len(rows)} failures={failures} CSV={args.out}", flush=True)
        success = failures == 0 and len(rows) == len(widths) * len(apps) * len(SHAPES) * len(MODES)
    finally:
        if owns_snapshots and success and not args.keep_snapshots:
            shutil.rmtree(snapshot_root, ignore_errors=True)
        else:
            print(f"[snapshots] {snapshot_root}", flush=True)
        # Snapshot execution is read-only, so restoring after it cannot invalidate results.
        lock_path = exp1.TEST_DIR / ".moti-build.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            restore_release(args.timeout)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
