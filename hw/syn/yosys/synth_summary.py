#!/usr/bin/env python3
"""Collapse the Yosys/OpenSTA report set into one synth_summary.csv.

The Xilinx flow's project.tcl already emits a single synth_summary.csv, and
ci/fpga_gate.py reads only that file. The Yosys flow instead scatters its
numbers across reports/{stat_lib,sram_area,wns,tns,power}.rpt, so a gate over
this flow would need a parser per report. Emitting the same one-row CSV here
keeps report-format knowledge next to the tool that produces it, and lets the
gate stay tool-neutral.

Columns mirror the Xilinx summary where the concept survives the FPGA->ASIC
change (fmax_mhz, wns_ns) and replace it where it does not: LUT/DSP/BRAM give
way to cell_area_um2 / sram_area_um2 / cell_count / seq_area_um2.

  usage: synth_summary.py --reports <dir> --top <module> --clock-mhz <f>
                          [--output <path>]
"""

import argparse
import csv
import os
import re
import sys

COLUMNS = ("top", "clock_mhz", "fmax_mhz", "wns_ns", "tns_ns", "cell_area_um2",
           "seq_area_um2", "sram_area_um2", "cell_count", "power_mw")


def read(path):
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def slack(path):
    """OpenSTA's slack reports emit a single '<kind> max <value>' line."""
    m = re.search(r"^\s*(?:worst slack|wns|tns)\s+max\s+(-?[\d.eE+]+)",
                  read(path), re.M)
    return float(m.group(1)) if m else None


def area(path):
    """Total and sequential cell area, plus cell count, from `stat -liberty`."""
    text = read(path)
    out = {}
    m = re.search(r"Chip area for module\s+'\\?[^']*':\s*([\d.]+)", text)
    if m:
        out["cell_area_um2"] = round(float(m.group(1)), 3)
    m = re.search(r"used for sequential elements:\s*([\d.]+)", text)
    if m:
        out["seq_area_um2"] = round(float(m.group(1)), 3)
    # "  46213 5.24E+03 cells" -- count is the first column of the `cells` row.
    m = re.search(r"^\s*(\d+)\s+\S+\s+cells\s*$", text, re.M)
    if m:
        out["cell_count"] = int(m.group(1))
    return out


def sram(path):
    m = re.search(r"TOTAL SRAM ESTIMATED AREA:\s*([\d.]+)", read(path))
    return round(float(m.group(1)), 3) if m else None


def power(path):
    """Total power in mW. OpenSTA's report_power prints Watts."""
    m = re.search(r"^Total\s+\S+\s+\S+\s+\S+\s+([\d.eE+-]+)", read(path), re.M)
    return round(float(m.group(1)) * 1000.0, 6) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", required=True, help="reports/ directory")
    ap.add_argument("--top", required=True, help="top module name")
    ap.add_argument("--clock-mhz", type=float, required=True,
                    help="target clock the design was constrained to")
    ap.add_argument("--output", help="default: <reports>/../synth_summary.csv")
    args = ap.parse_args(argv)

    rpt = args.reports
    row = {"top": args.top, "clock_mhz": args.clock_mhz}
    row.update(area(os.path.join(rpt, "stat_lib.rpt")))
    row["sram_area_um2"] = sram(os.path.join(rpt, "sram_area.rpt"))
    row["power_mw"] = power(os.path.join(rpt, "power.rpt"))
    # Signed worst slack, from report_worst_slack. NOT report_wns: that one is
    # worst *negative* slack and clamps at 0, so every design that closes reports
    # 0 and the Fmax below would just be the target clock read back. ABC maps to
    # the target period and stops, so closing with a few picoseconds of margin is
    # the normal outcome and the sign is the whole signal. wns.rpt is the
    # fallback for an OpenSTA without report_worst_slack.
    row["wns_ns"] = slack(os.path.join(rpt, "worst_slack.rpt"))
    if row["wns_ns"] is None:
        row["wns_ns"] = slack(os.path.join(rpt, "wns.rpt"))
    row["tns_ns"] = slack(os.path.join(rpt, "tns.rpt"))

    # Achieved frequency from the constrained period and its worst slack. A
    # positive WNS means the design closed with margin, so Fmax is above target;
    # negative means it did not meet the clock it was built for.
    if row["wns_ns"] is not None and args.clock_mhz:
        period = 1000.0 / args.clock_mhz
        achieved = period - row["wns_ns"]
        row["fmax_mhz"] = round(1000.0 / achieved, 3) if achieved > 0 else None
    else:
        row["fmax_mhz"] = None

    out = args.output or os.path.join(os.path.dirname(rpt.rstrip("/")) or ".",
                                      "synth_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row[k]) for k in COLUMNS})

    missing = [k for k in ("cell_area_um2", "wns_ns") if row.get(k) is None]
    if missing:
        print("WARNING: %s: no %s in %s" % (out, ", ".join(missing), rpt),
              file=sys.stderr)
    print("synth_summary.csv  area=%s um^2  Fmax=%s MHz  WNS=%s ns  cells=%s"
          % (row.get("cell_area_um2"), row.get("fmax_mhz"), row.get("wns_ns"),
             row.get("cell_count")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
