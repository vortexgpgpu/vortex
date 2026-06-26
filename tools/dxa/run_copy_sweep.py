#!/usr/bin/env python3
"""Run DXA copy and multicast copy sweeps using existing regression apps."""

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
MCAST_RE = re.compile(r"\bDXA_COPY_MCAST_RESULT\b\s+(.*)")


def parse_list(value):
    return [int(x) for x in value.split(",") if x]


def parse_status_set(value):
    return {x.strip().upper() for x in value.split(",") if x.strip()}


def parse_kv_tail(text):
    result = {}
    for item in text.split():
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def compact_output(output):
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        keep = (
            stripped.startswith("mode:")
            or stripped.startswith("writeback:")
            or stripped.startswith("source:")
            or stripped.startswith("tiles:")
            or stripped.startswith("total_elems:")
            or stripped == "start"
            or stripped.startswith("PERF:")
            or stripped.startswith("DXA_COPY_MCAST_RESULT")
            or stripped in ("PASSED", "FAILED")
            or stripped.startswith("Error:")
            or stripped.startswith("*** error:")
        )
        if keep:
            lines.append(line)
    return "\\n".join(lines[-40:])


def run_case(args, row, app_args, configs):
    perf_class = "16" if row.get("figure") == "3c" else "6"
    cmd = [
        "timeout",
        "-k",
        "5s",
        f"{args.timeout}s",
        "./ci/blackbox.sh",
        f"--driver={row['driver']}",
        "--cores=1",
        f"--warps={row['warps']}",
        f"--threads={row['threads']}",
        "--l2cache",
        f"--perf={perf_class}",
        f"--app={row['app']}",
        "--args=" + " ".join(app_args),
    ]

    env = os.environ.copy()
    if configs:
        env["CONFIGS"] = " ".join(configs)
    if row.get("figure") == "3b" and row.get("variant") == "lsu":
        env["DXA_COPY_FORCE_LSU"] = "1"

    row["command"] = " ".join(cmd)
    row["configs"] = env.get("CONFIGS", "")
    row["env_overrides"] = "DXA_COPY_FORCE_LSU=1" if env.get("DXA_COPY_FORCE_LSU") == "1" else ""

    if args.dry_run:
        row["status"] = "DRY_RUN"
        row["attempts"] = "0"
        return row

    for attempt in range(1, args.retries + 2):
        row["returncode"] = ""
        row["instrs"] = ""
        row["cycles"] = ""
        row["ipc"] = ""
        row["summary_status"] = ""

        proc = subprocess.run(
            cmd,
            cwd=args.build_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = proc.stdout
        row["returncode"] = proc.returncode
        row["attempts"] = str(attempt)

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

        mcast_match = MCAST_RE.search(output)
        if mcast_match:
            for key, value in parse_kv_tail(mcast_match.group(1)).items():
                row[f"summary_{key}"] = value

        if proc.returncode == 124:
            row["status"] = "TIMEOUT"
        elif proc.returncode == 0 and "PASSED" in output:
            row["status"] = "PASS"
        else:
            row["status"] = "FAIL"

        row["output_tail"] = compact_output(output)
        if row["status"] == "PASS" or attempt > args.retries:
            break

    return row


def base_configs(args, enable_dxa):
    configs = list(args.extra_config)
    if args.lmem_log_size:
        configs.append(f"-DVX_CFG_LMEM_LOG_SIZE={args.lmem_log_size}")
    if enable_dxa:
        configs.append("-DVX_CFG_EXT_DXA_ENABLE")
    return configs


def figure_b_rows(args):
    for warps in args.warps:
        for threads in args.threads:
            for variant, enable_dxa in (("lsu", False), ("dxa", True)):
                for tile_rows in args.tile_sizes:
                    for tile_cols in args.tile_sizes:
                        row = {
                            "figure": "3b",
                            "driver": args.driver,
                            "app": "dxa_copy",
                            "variant": variant,
                            "warps": warps,
                            "threads": threads,
                            "tile_rows": tile_rows,
                            "tile_cols": tile_cols,
                            "matrix_rows": args.matrix_size,
                            "matrix_cols": args.matrix_size,
                        }
                        app_args = [
                            "-d2",
                            "-s0", str(args.matrix_size),
                            "-s1", str(args.matrix_size),
                            "-t0", str(tile_cols),
                            "-t1", str(tile_rows),
                        ]
                        yield row, app_args, base_configs(args, enable_dxa)


def figure_c_rows(args):
    for warps in args.warps:
        for threads in args.threads:
            for variant in ("percta", "mcast"):
                for tile_rows in args.tile_sizes:
                    for tile_cols in args.tile_sizes:
                        row = {
                            "figure": "3c",
                            "driver": args.driver,
                            "app": "dxa_copy_mcast",
                            "variant": variant,
                            "warps": warps,
                            "threads": threads,
                            "tile_rows": tile_rows,
                            "tile_cols": tile_cols,
                            "matrix_rows": args.matrix_size,
                            "matrix_cols": args.matrix_size,
                            "num_ctas": args.num_ctas,
                            "writeback": args.mcast_writeback,
                            "pipeline_depth": args.mcast_pipeline_depth,
                        }
                        app_args = [
                            f"--mode={variant}",
                            f"--writeback={args.mcast_writeback}",
                            f"--pipeline-depth={args.mcast_pipeline_depth}",
                            f"--num-ctas={args.num_ctas}",
                            "-r", str(args.matrix_size),
                            "-c", str(args.matrix_size),
                            "-R", str(tile_rows),
                            "-C", str(tile_cols),
                            f"--verify={args.verify}",
                        ]
                        yield row, app_args, base_configs(args, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--driver", choices=("simx", "rtlsim"), default="simx")
    parser.add_argument("--figure", choices=("3b", "3c", "both"), default="both")
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--warps", type=parse_list, default=parse_list("8,16,32"))
    parser.add_argument("--threads", type=parse_list, default=parse_list("8,16,32"))
    parser.add_argument("--tile-sizes", type=parse_list, default=parse_list("16,32,64,128"))
    parser.add_argument("--num-ctas", type=int, default=4)
    parser.add_argument("--mcast-writeback", choices=("full", "sample", "none"), default="full")
    parser.add_argument("--mcast-pipeline-depth", type=int, default=1)
    parser.add_argument("--verify", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--lmem-log-size", type=int, default=18)
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("docs/results/dxa_copy_sweep/raw.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="append missing rows and skip keys already present in the output CSV")
    parser.add_argument("--retries", type=int, default=0,
                        help="retry a non-PASS case this many times before writing the row")
    parser.add_argument("--rerun-status", type=parse_status_set, default=set(),
                        help="with --resume, rerun existing rows whose status is in this comma-separated set")
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / args.build_dir / "ci" / "blackbox.sh").exists():
        sys.exit(f"missing generated blackbox.sh under {repo_root / args.build_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    generators = []
    if args.figure in ("3b", "both"):
        generators.append(figure_b_rows(args))
    if args.figure in ("3c", "both"):
        generators.append(figure_c_rows(args))

    fieldnames = [
        "figure", "driver", "app", "variant", "warps", "threads",
        "tile_rows", "tile_cols", "matrix_rows", "matrix_cols", "num_ctas", "writeback",
        "pipeline_depth",
        "status", "returncode", "instrs", "cycles", "ipc",
        "dxa_transfers", "dxa_gmem_reads", "dxa_gmem_dedup",
        "dxa_gmem_dedup_rate", "dxa_lmem_writes", "dxa_avg_gmem_lat",
        "summary_status", "attempts", "configs", "env_overrides", "command", "output_tail",
    ]

    def row_key(row):
        return tuple(str(row.get(k, "")) for k in (
            "figure", "driver", "app", "variant", "warps", "threads",
            "tile_rows", "tile_cols", "matrix_rows", "matrix_cols", "num_ctas", "writeback",
            "pipeline_depth",
        ))

    completed = {}
    kept_rows = []
    append = args.resume and args.output.exists()
    if append:
        with args.output.open(newline="") as f:
            for row in csv.DictReader(f):
                status = row.get("status", "").upper()
                if status in args.rerun_status:
                    continue
                kept_rows.append(row)
                completed[row_key(row)] = status

        if args.rerun_status:
            with args.output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(kept_rows)

    with args.output.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for gen in generators:
            for row, app_args, configs in gen:
                for key in fieldnames:
                    row.setdefault(key, "")
                key = row_key(row)
                existing_status = completed.get(key)
                if existing_status and existing_status not in args.rerun_status:
                    print(
                        f"skip {row['figure']} {row['variant']} "
                        f"w{row['warps']} t{row['threads']} "
                        f"tile={row['tile_rows']}x{row['tile_cols']}",
                        flush=True,
                    )
                    continue
                if existing_status:
                    print(
                        f"rerun {existing_status} {row['figure']} {row['variant']} "
                        f"w{row['warps']} t{row['threads']} "
                        f"tile={row['tile_rows']}x{row['tile_cols']}",
                        flush=True,
                    )
                else:
                    print(
                        f"{row['figure']} {row['variant']} "
                        f"w{row['warps']} t{row['threads']} "
                        f"tile={row['tile_rows']}x{row['tile_cols']}",
                        flush=True,
                    )
                writer.writerow(run_case(args, row, app_args, configs))
                f.flush()


if __name__ == "__main__":
    main()
