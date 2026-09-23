#!/usr/bin/env python3
"""Append the [LLAMA] lines of one llama_dtcu run log to the results CSV (same columns as
sweep_llama.py), for runs that were started by hand rather than by the sweep.

  usage: log2csv.py <run.log> [<out.csv>]
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_llama import FIELDS, TEST_DIR, kv  # noqa: E402

# Same defaults as sweep_llama.py --out: the test's build directory, overridable via VX_TEST_DIR.
DEFAULT_OUT = str(TEST_DIR / "llama_dtcu_results.csv")
DECODE_OUT = str(TEST_DIR / "llama_dtcu_decode_results.csv")


def main():
    log = Path(sys.argv[1])
    text = log.read_text()
    is_decode = any(l.startswith("[LLAMA] ") and " B=" in l for l in text.splitlines())
    out = Path(sys.argv[2] if len(sys.argv) > 2 else (DECODE_OUT if is_decode else DEFAULT_OUT))
    date = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(log.stat().st_mtime))
    rows = []
    for line in text.splitlines():
        if not line.startswith("[LLAMA] "):
            continue
        d = kv(line)
        run = d.get("run", "")
        mode = run.split(":")[1] if run.startswith("uniform:") else "map"
        d["T"] = d.get("T") or d.get("B")      # decode lines carry B (sequences) instead of T
        if "site" in d:
            rows.append(dict(date=date, T=d.get("T"), mode=mode, run=run, site=d["site"], site_mode=d.get("mode"),
                             launches=d.get("launches"), cycles=d.get("cycles"), status="ok"))
        elif "total_cycles" in d:
            rows.append(dict(date=date, T=d.get("T"), mode=mode, run=run, site="TOTAL", site_mode=mode,
                             launches=d.get("total_launches"), cycles=d.get("total_cycles"),
                             gemm_cycles=d.get("gemm_cycles"), pass_cycles=d.get("pass_cycles"),
                             cycles_per_token=d.get("cycles_per_token"), tok_per_s_400MHz=d.get("tok_per_s_400MHz"),
                             verify_fail_sites=d.get("verify_fail_sites"), next_dev=d.get("next_dev"),
                             next_ref=d.get("next_ref"), next_fp32=d.get("next_fp32"),
                             program_swaps=d.get("program_swaps"), wall_s=d.get("wall_s"), status="ok"))
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows from {log} to {out}")


if __name__ == "__main__":
    main()
