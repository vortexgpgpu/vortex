#!/usr/bin/env python3
"""Run hard and shared-memory atomic barrier overhead sweeps."""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


PERF_RE = re.compile(r"PERF:\s*instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)")
RESULT_RE = re.compile(r"\bSMEM_BARRIER_RESULT\b\s+(.*)")


def parse_list(value):
    return [int(x) for x in value.split(",") if x]


def parse_modes(value):
    modes = []
    for item in value.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in ("hard", "soft"):
            raise argparse.ArgumentTypeError("modes must be hard, soft, or hard,soft")
        modes.append(mode)
    return modes


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
            stripped.startswith("threads:")
            or stripped.startswith("warps:")
            or stripped.startswith("mode:")
            or stripped.startswith("payload bytes:")
            or stripped.startswith("iterations:")
            or stripped.startswith("PERF:")
            or stripped.startswith("result:")
            or stripped.startswith("SMEM_BARRIER_RESULT")
            or stripped in ("PASSED!", "FAILED!")
            or stripped.startswith("Error:")
        )
        if keep:
            lines.append(line)
    return "\\n".join(lines[-40:])


def enrich(row):
    try:
        iterations = max(1, int(row.get("iterations", "1")))
        event_cycles = int(row.get("event_cycles", "0"))
        release_cycles = int(row.get("release_cycles", "0"))
        register_cycles = int(row.get("register_cycles", "0"))
        wait_iters = int(row.get("wait_iters", "0"))
    except ValueError:
        return row

    overhead = max(0, release_cycles - event_cycles)
    row["barrier_overhead_cycles"] = str(overhead)
    row["barrier_overhead_per_iter"] = f"{overhead / iterations:.3f}"
    row["release_cycles_per_iter"] = f"{release_cycles / iterations:.3f}"
    row["event_cycles_per_iter"] = f"{event_cycles / iterations:.3f}"
    row["register_cycles_per_iter"] = f"{register_cycles / iterations:.3f}"
    row["wait_iters_per_iter"] = f"{wait_iters / iterations:.3f}"
    row["overhead_ratio"] = f"{(overhead / max(1, event_cycles)):.6f}"
    row["overhead_release_ratio"] = f"{(overhead / max(1, release_cycles)):.6f}"
    return row


def run_case(args, payload, mode):
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
        "--app=smem_atomic_barrier",
        f"--args=-b{payload} -i{args.iterations} -m{mode}",
    ]

    env = os.environ.copy()
    configs = list(args.extra_config)
    configs.extend(["-DVX_CFG_EXT_A_ENABLE", "-DVX_CFG_EXT_DXA_ENABLE"])
    if args.lmem_log_size:
        configs.append(f"-DVX_CFG_LMEM_LOG_SIZE={args.lmem_log_size}")
    env["CONFIGS"] = " ".join(configs)

    row = {
        "driver": args.driver,
        "mode": mode,
        "payload_bytes": str(payload),
        "warps": str(args.warps),
        "threads": str(args.threads),
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

    result_match = RESULT_RE.search(output)
    if result_match:
        row.update(parse_kv_tail(result_match.group(1)))

    if proc.returncode == 124:
        row["status"] = "TIMEOUT"
    elif proc.returncode == 0 and "PASSED!" in output:
        row["status"] = "PASS"
    else:
        row["status"] = "FAIL"

    row["output_tail"] = compact_output(output)
    return enrich(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--driver", choices=("simx", "rtlsim"), default="simx")
    parser.add_argument("--workloads", type=parse_list,
                        default=parse_list("1024,2048,4096,8192,16384,32768"))
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("soft"))
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--lmem-log-size", type=int, default=18)
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument("--output", type=Path,
                        default=Path("docs/results/dxa_barrier_overhead/soft_smem_barrier.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.build_dir / "ci" / "blackbox.sh").exists():
        sys.exit(f"missing generated blackbox.sh under {args.build_dir}")

    rows = [
        run_case(args, payload, mode)
        for payload in args.workloads
        for mode in args.modes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "driver", "mode", "payload_bytes", "warps", "threads", "status",
        "returncode", "iterations", "failures", "pending", "phase",
        "arrived", "register_cycles", "event_cycles", "release_cycles",
        "barrier_overhead_cycles", "register_cycles_per_iter",
        "event_cycles_per_iter", "release_cycles_per_iter",
        "barrier_overhead_per_iter", "overhead_ratio",
        "overhead_release_ratio", "wait_iters",
        "wait_iters_per_iter", "checksum", "instrs", "cycles", "ipc",
        "configs", "command", "output_tail",
    ]

    extras = sorted({k for row in rows for k in row if k not in fieldnames})
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    passes = sum(1 for row in rows if row.get("status") == "PASS")
    print(f"wrote {args.output} ({passes}/{len(rows)} PASS)")
    if passes != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
