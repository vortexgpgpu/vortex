#!/usr/bin/env python3
"""Run hard-vs-soft DXA barrier release-latency sweeps."""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


PERF_RE = re.compile(r"PERF:\s*instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)")
DXA_PERF_RE = re.compile(
    r"PERF:\s*dxa:\s*transfers=(\d+),\s*gmem_reads=(\d+),\s*"
    r"gmem_dedup=(\d+)\s*\(rate=(\d+)%\),\s*lmem_writes=(\d+),\s*"
    r"avg_gmem_lat=([0-9.]+)"
)
RESULT_RE = re.compile(r"\bDXA_BARRIER_RESULT\b\s+(.*)")

DEFAULT_SHAPES = {
    1024: (16, 16),
    2048: (16, 32),
    4096: (32, 32),
    8192: (32, 64),
    16384: (64, 64),
    32768: (64, 128),
}


def parse_list(value):
    return [int(x) for x in value.split(",") if x]


def parse_shape(value):
    if "x" not in value:
        raise argparse.ArgumentTypeError("shape must be PAYLOAD:ROWSxCOLS")
    payload_text, shape_text = value.split(":", 1)
    rows_text, cols_text = shape_text.lower().split("x", 1)
    payload = int(payload_text)
    rows = int(rows_text)
    cols = int(cols_text)
    if rows <= 0 or cols <= 0 or payload <= 0:
        raise argparse.ArgumentTypeError("payload and shape entries must be positive")
    if rows * cols * 4 != payload:
        raise argparse.ArgumentTypeError(
            f"{rows}x{cols} float tile is {rows * cols * 4} bytes, not {payload}"
        )
    return payload, rows, cols


def parse_kv_tail(text):
    result = {}
    for item in text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def compact_output(output):
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        keep = (
            stripped.startswith("mode:")
            or stripped.startswith("softbar:")
            or stripped.startswith("timing:")
            or stripped.startswith("sizes:")
            or stripped.startswith("tiles:")
            or stripped.startswith("total_elems:")
            or stripped.startswith("DXA_BARRIER_RESULT")
            or stripped.startswith("PERF:")
            or stripped in ("PASSED", "FAILED")
            or stripped.startswith("Error:")
            or stripped.startswith("*** error:")
        )
        if keep:
            lines.append(line)
    return "\\n".join(lines[-60:])


def run_case(args, payload, rows, cols, mode):
    app_args = [
        "-d2",
        "-s0", str(cols),
        "-s1", str(rows),
        "-t0", str(cols),
        "-t1", str(rows),
        "-B",
    ]
    if mode == "soft_dxa_completion":
        app_args.append("-S")

    cmd = [
        "timeout",
        "-k",
        "5s",
        f"{args.timeout}s",
        "./ci/blackbox.sh",
        f"--driver={args.driver}",
        "--cores=1",
        f"--warps={args.warps}",
        f"--threads={args.threads}",
        "--l2cache",
        "--perf=16",
        "--app=dxa_copy",
        "--args=" + " ".join(app_args),
    ]

    env = os.environ.copy()
    configs = list(args.extra_config)
    configs.extend(["-DVX_CFG_EXT_DXA_ENABLE", "-DVX_CFG_EXT_A_ENABLE"])
    if args.lmem_log_size:
        configs.append(f"-DVX_CFG_LMEM_LOG_SIZE={args.lmem_log_size}")
    env["CONFIGS"] = " ".join(configs)

    row = {
        "driver": args.driver,
        "mode": mode,
        "payload_bytes": str(payload),
        "tile_rows": str(rows),
        "tile_cols": str(cols),
        "warps": str(args.warps),
        "threads": str(args.threads),
        "l2cache": "1",
        "command": " ".join(cmd),
        "configs": env["CONFIGS"],
    }

    if args.dry_run:
        row["status"] = "DRY_RUN"
        return row

    proc = subprocess.run(
        cmd,
        cwd=args.build_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout
    row["returncode"] = str(proc.returncode)

    perf_match = PERF_RE.search(output)
    if perf_match:
        row["instrs"] = perf_match.group(1)
        row["cycles"] = perf_match.group(2)
        row["ipc"] = perf_match.group(3)

    dxa_perf_matches = list(DXA_PERF_RE.finditer(output))
    if dxa_perf_matches:
        dxa_perf_match = dxa_perf_matches[-1]
        row["dxa_transfers"] = dxa_perf_match.group(1)
        row["dxa_gmem_reads"] = dxa_perf_match.group(2)
        row["dxa_gmem_dedup"] = dxa_perf_match.group(3)
        row["dxa_gmem_dedup_rate"] = dxa_perf_match.group(4)
        row["dxa_lmem_writes"] = dxa_perf_match.group(5)
        row["dxa_avg_gmem_lat"] = dxa_perf_match.group(6)

    result_match = RESULT_RE.search(output)
    if result_match:
        row.update(parse_kv_tail(result_match.group(1)))

    if proc.returncode == 124:
        row["status"] = "TIMEOUT"
    elif proc.returncode == 0 and "PASSED" in output and result_match:
        row["status"] = "PASS"
    else:
        row["status"] = "FAIL"

    row["output_tail"] = compact_output(output)
    return row


def add_pair_metrics(rows):
    by_payload = {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        by_payload.setdefault(row["payload_bytes"], {})[row["mode"]] = row

    for payload_rows in by_payload.values():
        hard = payload_rows.get("hard_dxa_completion")
        soft = payload_rows.get("soft_dxa_completion")
        if not hard or not soft:
            continue
        try:
            hard_release = int(hard["release_cycles"])
            soft_release = int(soft["release_cycles"])
            hard_cycles = int(hard.get("cycles", "0") or "0")
            soft_cycles = int(soft.get("cycles", "0") or "0")
        except ValueError:
            continue
        extra = soft_release - hard_release
        cycle_extra = soft_cycles - hard_cycles
        for row in (hard, soft):
            row["baseline_release_cycles"] = str(hard_release)
            row["soft_extra_release_cycles"] = str(extra)
            row["soft_extra_release_ratio_vs_hard"] = f"{extra / max(1, hard_release):.6f}"
            row["soft_extra_total_cycles"] = str(cycle_extra)
            row["soft_extra_total_ratio_vs_hard"] = f"{cycle_extra / max(1, hard_cycles):.6f}"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--driver", choices=("simx", "rtlsim"), default="simx")
    parser.add_argument("--workloads", type=parse_list,
                        default=parse_list("1024,2048,4096,8192,16384,32768"))
    parser.add_argument("--shape", type=parse_shape, action="append", default=[],
                        help="override/add a shape as PAYLOAD:ROWSxCOLS; TYPE is float")
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--lmem-log-size", type=int, default=18)
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument("--output", type=Path,
                        default=Path("docs/results/dxa_barrier_overhead/dxa_barrier_latency.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.build_dir / "ci" / "blackbox.sh").exists():
        sys.exit(f"missing generated blackbox.sh under {args.build_dir}")

    shapes = dict(DEFAULT_SHAPES)
    for payload, rows, cols in args.shape:
        shapes[payload] = (rows, cols)

    run_shapes = []
    for payload in args.workloads:
        if payload not in shapes:
            sys.exit(f"missing tile shape for payload {payload}; pass --shape {payload}:ROWSxCOLS")
        rows, cols = shapes[payload]
        run_shapes.append((payload, rows, cols))

    all_rows = []
    for payload, rows, cols in run_shapes:
        for mode in ("hard_dxa_completion", "soft_dxa_completion"):
            print(f"{args.driver} {mode} payload={payload} tile={rows}x{cols}", flush=True)
            all_rows.append(run_case(args, payload, rows, cols, mode))

    add_pair_metrics(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "driver", "mode", "payload_bytes", "tile_rows", "tile_cols",
        "warps", "threads", "l2cache", "status", "returncode",
        "failures", "groups", "register_cycles", "issue_cycles",
        "release_cycles", "wait_iters", "checksum",
        "baseline_release_cycles", "soft_extra_release_cycles",
        "soft_extra_release_ratio_vs_hard", "soft_extra_total_cycles",
        "soft_extra_total_ratio_vs_hard",
        "instrs", "cycles", "ipc", "dxa_transfers", "dxa_gmem_reads",
        "dxa_gmem_dedup", "dxa_gmem_dedup_rate", "dxa_lmem_writes",
        "dxa_avg_gmem_lat", "configs", "command", "output_tail",
    ]

    extras = sorted({k for row in all_rows for k in row if k not in fieldnames})
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(all_rows)

    passes = sum(1 for row in all_rows if row.get("status") == "PASS")
    print(f"wrote {args.output} ({passes}/{len(all_rows)} PASS)")
    if args.dry_run:
        return 0
    if passes != len(all_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
