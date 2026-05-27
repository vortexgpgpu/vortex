#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


DETAIL_RE = re.compile(
    r"Sparsity Mode:(?P<sparsity>\d+).*?"
    r"MxNxK=(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+).*?"
    r"A sparsity=(?P<a>[0-9.]+)%.*?"
    r"B sparsity=(?P<b>[0-9.]+)%.*?"
    r"It=(?P<itype>\S+)"
)
INT_METRIC_RE = re.compile(
    r"^(?P<name>Total kernel cycles|Total kernel instructions|"
    r"Kernel body cycles|Kernel body instructions|MUL active cycles):\s*(?P<value>\d+|N/A)$"
)
XBAR_RE = re.compile(r"^XBAR stalls:\s*(?P<pct>[0-9.]+)%\s*\((?P<cycles>\d+) cycles\)$")
LMEM_RE = re.compile(
    r"^(?:PERF2:\s*)?core(?P<core>\d+): lmem: reqs=(?P<reqs>\d+), "
    r"bank_stalls=(?P<bank>\d+) \(utility=(?P<utility>[0-9.]+)%\)$"
)
COALESCER_RE = re.compile(r"^(?:PERF2:\s*)?core(?P<core>\d+): coalescer: misses=(?P<misses>\d+)$")
ICACHE_RE = re.compile(
    r"^(?:PERF2:\s*)?core(?P<core>\d+): icache: reads=(?P<reads>\d+), "
    r"miss=(?P<miss>\d+) \(hit=(?P<hit>[0-9.]+)%\), "
    r"mshr_st=(?P<mshr>\d+) \(utility=(?P<utility>[0-9.]+)%\)$"
)
DCACHE_RE = re.compile(
    r"^(?:PERF2:\s*)?core(?P<core>\d+): dcache: reqs=(?P<reqs>\d+), "
    r"miss_r=(?P<miss_r>\d+) \(hit=(?P<hit_r>[0-9.]+)%\), "
    r"miss_w=(?P<miss_w>\d+) \(hit=(?P<hit_w>[0-9.]+)%\), "
    r"bank_st=(?P<bank>\d+) \(utility=(?P<utility>[0-9.]+)%\)$"
)
MEMORY_RE = re.compile(
    r"^(?:PERF2:\s*)?memory: reqs=(?P<reqs>\d+) "
    r"\(r=(?P<reads>\d+), w=(?P<writes>\d+)\), "
    r"lat=(?P<lat>[0-9.]+) cyc, bank_st=(?P<bank>\d+) "
    r"\(utility=(?P<utility>[0-9.]+)%\)$"
)
INSTRS_RE = re.compile(
    r"^(?:PERF2:\s*)?instrs=(?P<instrs>\d+), cycles=(?P<cycles>\d+), IPC=(?P<ipc>[0-9.]+)$"
)
TCU_CYCLES_RE = re.compile(
    r"^TCU cycles \(from first TCU issue to last TCU commit\): (?P<value>\d+)$"
)
MULS_ACTIVE_RE = re.compile(r"^MULs active: (?P<value>[0-9.]+)$")
TCU_UTIL_RE = re.compile(r"^TCU utilization: (?P<value>[0-9.]+)%$")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge stat files from split run_all kernels.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--sparsity", required=True, type=int)
    parser.add_argument("parts", nargs="+", type=Path)
    return parser.parse_args()


def fmt_float(value, places=2):
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def add_int(target, key, value):
    target[key] = target.get(key, 0) + int(value)


def add_float(target, key, value):
    target.setdefault(key, []).append(float(value))


def parse_stat(path):
    stat = {
        "path": path,
        "testbench": "",
        "details": {},
        "flags": "",
        "feop": "",
        "arch": "",
        "perf": "",
        "status": "FAIL",
        "int_metrics": {},
        "xbar_pct": None,
        "xbar_cycles": 0,
        "perf2": {},
        "tcu_cycles": None,
        "muls_active": None,
        "tcu_util": None,
    }

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Testbench:"):
            stat["testbench"] = line
            continue
        detail_match = DETAIL_RE.search(line)
        if detail_match:
            stat["details"] = detail_match.groupdict()
            continue
        if line.startswith("Flags:"):
            stat["flags"] = line
            continue
        if line.startswith("BLOCK_M="):
            stat["feop"] = line
            continue
        if line.startswith("Threads:"):
            stat["arch"] = line
            continue
        if line.startswith("perf="):
            stat["perf"] = line
            continue
        if line in ("PASS", "FAIL"):
            stat["status"] = line
            continue

        match = INT_METRIC_RE.match(line)
        if match and match.group("value") != "N/A":
            stat["int_metrics"][match.group("name")] = int(match.group("value"))
            continue

        match = XBAR_RE.match(line)
        if match:
            stat["xbar_pct"] = float(match.group("pct"))
            stat["xbar_cycles"] = int(match.group("cycles"))
            continue

        for key, regex in (
            ("lmem", LMEM_RE),
            ("coalescer", COALESCER_RE),
            ("icache", ICACHE_RE),
            ("dcache", DCACHE_RE),
            ("memory", MEMORY_RE),
            ("instrs", INSTRS_RE),
        ):
            match = regex.match(line)
            if match and key not in stat["perf2"]:
                stat["perf2"][key] = match.groupdict()
                break
        else:
            match = TCU_CYCLES_RE.match(line)
            if match:
                stat["tcu_cycles"] = int(match.group("value"))
                continue
            match = MULS_ACTIVE_RE.match(line)
            if match:
                stat["muls_active"] = float(match.group("value"))
                continue
            match = TCU_UTIL_RE.match(line)
            if match:
                stat["tcu_util"] = float(match.group("value"))

    return stat


def average(values):
    return sum(values) / len(values) if values else 0.0


def aggregate(parts):
    out = {
        "testbench": parts[0]["testbench"],
        "details": parts[0]["details"],
        "flags": parts[0]["flags"],
        "feop": parts[0]["feop"],
        "arch": parts[0]["arch"],
        "perf": parts[0]["perf"],
        "status": "PASS" if all(part["status"] == "PASS" for part in parts) else "FAIL",
        "int_metrics": {},
        "xbar_pcts": [],
        "xbar_cycles": 0,
        "perf2": {},
        "tcu_cycles": 0,
        "muls_active": 0.0,
        "tcu_utils": [],
        "a_sparsities": [],
        "b_sparsities": [],
    }

    for part in parts:
        for key, value in part["int_metrics"].items():
            add_int(out["int_metrics"], key, value)
        if part["xbar_pct"] is not None:
            out["xbar_pcts"].append(part["xbar_pct"])
            out["xbar_cycles"] += part["xbar_cycles"]
        if part["tcu_cycles"] is not None:
            out["tcu_cycles"] += part["tcu_cycles"]
        if part["muls_active"] is not None:
            out["muls_active"] += part["muls_active"]
        if part["tcu_util"] is not None:
            out["tcu_utils"].append(part["tcu_util"])
        if part["details"]:
            out["a_sparsities"].append(float(part["details"]["a"]))
            out["b_sparsities"].append(float(part["details"]["b"]))

        for key, fields in part["perf2"].items():
            bucket = out["perf2"].setdefault(key, {})
            if key == "lmem":
                add_int(bucket, "reqs", fields["reqs"])
                add_int(bucket, "bank", fields["bank"])
                add_float(bucket, "utility", fields["utility"])
            elif key == "coalescer":
                add_int(bucket, "misses", fields["misses"])
            elif key == "icache":
                add_int(bucket, "reads", fields["reads"])
                add_int(bucket, "miss", fields["miss"])
                add_int(bucket, "mshr", fields["mshr"])
                add_float(bucket, "hit", fields["hit"])
                add_float(bucket, "utility", fields["utility"])
            elif key == "dcache":
                add_int(bucket, "reqs", fields["reqs"])
                add_int(bucket, "miss_r", fields["miss_r"])
                add_int(bucket, "miss_w", fields["miss_w"])
                add_int(bucket, "bank", fields["bank"])
                add_float(bucket, "hit_r", fields["hit_r"])
                add_float(bucket, "hit_w", fields["hit_w"])
                add_float(bucket, "utility", fields["utility"])
            elif key == "memory":
                reqs = int(fields["reqs"])
                add_int(bucket, "reqs", reqs)
                add_int(bucket, "reads", fields["reads"])
                add_int(bucket, "writes", fields["writes"])
                add_int(bucket, "bank", fields["bank"])
                bucket["lat_weighted"] = bucket.get("lat_weighted", 0.0) + float(fields["lat"]) * reqs
                add_float(bucket, "utility", fields["utility"])
            elif key == "instrs":
                add_int(bucket, "instrs", fields["instrs"])
                add_int(bucket, "cycles", fields["cycles"])

    return out


def write_merged(output, merged, m, n, k, sparsity):
    details = merged["details"]
    a_sparsity = average(merged["a_sparsities"]) if merged["a_sparsities"] else float(details.get("a", 0.0))
    b_sparsity = average(merged["b_sparsities"]) if merged["b_sparsities"] else float(details.get("b", 0.0))
    itype = details.get("itype", "")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        if merged["testbench"]:
            f.write(f"{merged['testbench']}\n")
        f.write(
            f"Sparsity Mode:{sparsity} | MxNxK={m}x{n}x{k} | "
            f"A sparsity={fmt_float(a_sparsity, 4)}% | "
            f"B sparsity={fmt_float(b_sparsity, 4)}% | It={itype}\n"
        )
        for key in ("flags", "feop", "arch", "perf"):
            if merged[key]:
                f.write(f"{merged[key]}\n")
        f.write("\n")
        f.write(f"{merged['status']}\n")

        for name in (
            "Total kernel cycles",
            "Total kernel instructions",
            "Kernel body cycles",
            "Kernel body instructions",
            "MUL active cycles",
        ):
            value = merged["int_metrics"].get(name)
            f.write(f"{name}: {'N/A' if value is None else value}\n")

        f.write(
            f"XBAR stalls: {fmt_float(average(merged['xbar_pcts']), 2)}% "
            f"({merged['xbar_cycles']} cycles)\n"
        )
        f.write("\n")

        perf_lines = format_perf_lines(merged)
        for line in perf_lines:
            f.write(f"PERF2: {line}\n")
        if perf_lines:
            f.write("\nMEMORY (perf=2):\n")
            for line in perf_lines[:-1]:
                f.write(f"{line}\n")
            f.write("\n")

        f.write(
            "TCU cycles (from first TCU issue to last TCU commit): "
            f"{merged['tcu_cycles']}\n"
        )
        f.write(f"MULs active: {fmt_float(merged['muls_active'], 2)}\n")
        f.write(f"TCU utilization: {fmt_float(average(merged['tcu_utils']), 2)}%\n")


def format_perf_lines(merged):
    lines = []
    perf2 = merged["perf2"]
    if "lmem" in perf2:
        item = perf2["lmem"]
        lines.append(
            f"core0: lmem: reqs={item['reqs']}, bank_stalls={item['bank']} "
            f"(utility={fmt_float(average(item['utility']), 2)}%)"
        )
    if "coalescer" in perf2:
        item = perf2["coalescer"]
        lines.append(f"core0: coalescer: misses={item['misses']}")
    if "icache" in perf2:
        item = perf2["icache"]
        lines.append(
            f"core0: icache: reads={item['reads']}, miss={item['miss']} "
            f"(hit={fmt_float(average(item['hit']), 2)}%), mshr_st={item['mshr']} "
            f"(utility={fmt_float(average(item['utility']), 2)}%)"
        )
    if "dcache" in perf2:
        item = perf2["dcache"]
        lines.append(
            f"core0: dcache: reqs={item['reqs']}, "
            f"miss_r={item['miss_r']} (hit={fmt_float(average(item['hit_r']), 2)}%), "
            f"miss_w={item['miss_w']} (hit={fmt_float(average(item['hit_w']), 2)}%), "
            f"bank_st={item['bank']} (utility={fmt_float(average(item['utility']), 2)}%)"
        )
    if "memory" in perf2:
        item = perf2["memory"]
        lat = item["lat_weighted"] / item["reqs"] if item["reqs"] else 0.0
        lines.append(
            f"memory: reqs={item['reqs']} (r={item['reads']}, w={item['writes']}), "
            f"lat={fmt_float(lat, 1)} cyc, bank_st={item['bank']} "
            f"(utility={fmt_float(average(item['utility']), 2)}%)"
        )
    if "instrs" in perf2:
        item = perf2["instrs"]
        ipc = item["instrs"] / item["cycles"] if item["cycles"] else 0.0
        lines.append(f"instrs={item['instrs']}, cycles={item['cycles']}, IPC={fmt_float(ipc, 3)}")
    return lines


def main():
    args = parse_args()
    parts = [parse_stat(path) for path in args.parts]
    write_merged(args.output, aggregate(parts), args.m, args.n, args.k, args.sparsity)


if __name__ == "__main__":
    main()
