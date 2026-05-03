#!/usr/bin/env python3
import argparse
import os
import re

TICK_RE = re.compile(r'^\s*\w+\s+(\d+):|^\s*(\d+):')
ISSUE_RE = re.compile(
    r'^\s*\w+\s+(\d+):.*Issuing (?:TCU )?u-op.*sel_wis=(\d+)'
    r'(?:.*ex_type=(\d+))?(?:.*op_type=0x([0-9a-fA-F]+))?'
)
ACCU_C_ACCUM_RE = re.compile(r'\baccumulate_c=1\b')
MATRIX_A_RE = re.compile(r'matrix A:\s*(\d+)\s*x\s*(\d+)', re.IGNORECASE)
MATRIX_B_RE = re.compile(r'matrix B:\s*(\d+)\s*x\s*(\d+)', re.IGNORECASE)
ACTUAL_SPARSITY_RE = re.compile(r'Actual sparsity:\s*([0-9]+(?:\.[0-9]+)?)%?', re.IGNORECASE)
LMEM_READ_RSP_MATRIX_RE = re.compile(r'\bLMEM:\s*read\s*rsp:.*\bmatrix\s*=\s*(Bitmap|A|B|C)\b', re.IGNORECASE)
INPUT_DTYPE_RE = re.compile(r'input data type:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
OUTPUT_DTYPE_RE = re.compile(r'output data type:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
SPARSE_MODE_LEVEL_RE = re.compile(
    r'Sparse mode enabled \(-s\) with sparsity level:\s*([0-2])',
    re.IGNORECASE,
)
SPARSITY_TYPE_RE = re.compile(r'\bsparsity\s*=\s*([0-2])\b', re.IGNORECASE)
FEOP_PARAMS_RE = re.compile(
    r'\[feop_accu\]:\s*Parameters:\s*'
    r'BLOCK_M\s*[:=]\s*(\d+),\s*'
    r'BLOCK_N\s*[:=]\s*(\d+),\s*'
    r'XBAR_QUEUE_DEPTH\s*[:=]\s*(\d+)',
    re.IGNORECASE,
)
CONFIG_WARPS_RE = re.compile(r'\bCONFIGS:.*\bnum_warps=(\d+)\b', re.IGNORECASE)
CONFIG_THREADS_RE = re.compile(r'\bCONFIGS:.*\bnum_threads=(\d+)\b', re.IGNORECASE)
TESTBENCH_RUN_RE = re.compile(r'\b\./([A-Za-z0-9_.+-]+)\b')
TESTBENCH_MAKE_RE = re.compile(r"/tests/(?:regression|opencl)/([^/'\s]+)")
FLAGS_LINE_RE = re.compile(r'^\s*Flags:\s*(.+?)\s*$', re.IGNORECASE)
TCU_TYPE_DEFINED_RE = re.compile(r'\b(TCU_TYPE_[A-Za-z0-9_]+)\b.*\bdefined\b', re.IGNORECASE)
TCU_OP_DEFINED_RE = re.compile(r'^\s*TCU_OP\s+defined\s*$', re.IGNORECASE)
BLUE_MARKER_HEX = "0x12345677"
GREEN_MARKER_HEX = "0x12345676"
MARKER_HEX = "0x12345678"
PRE_TCU_MARKER_HEX = "0x12345679"
MARKER_HEXES = {
    GREEN_MARKER_HEX.lower(),
    BLUE_MARKER_HEX.lower(),
    MARKER_HEX.lower(),
    PRE_TCU_MARKER_HEX.lower(),
}
MARKER_RE = re.compile(r'\b(?:0x12345676|0x12345677|0x12345678|0x12345679)\b', re.IGNORECASE)
PASS_RE = re.compile(r'^\s*PASSED\b', re.IGNORECASE)
FAIL_RE = re.compile(r'^\s*FAILED!\s*$', re.IGNORECASE)
FAIL_COUNT_RE = re.compile(r'ERROR:\s*Found\s+(\d+)\s*/\s*(\d+)\s+errors!', re.IGNORECASE)
KERNEL_CYCLES_RE = re.compile(r'\bcycles=(\d+)\b')
PERF_INSTRS_RE = re.compile(r'PERF:\s*instrs=(\d+)', re.IGNORECASE)
PERF_LINE_RE = re.compile(r'^\s*PERF:\s*(.+?)\s*$', re.IGNORECASE)
KERNEL_BODY_CYCLES_RE = re.compile(r'Kernel body cycles:\s*(\d+)', re.IGNORECASE)
KERNEL_BODY_INSTR_RE = re.compile(r'Kernel body instructions:\s*(\d+)', re.IGNORECASE)
DXA_CYCLES_A_RE = re.compile(r'DXA cycles A:\s*(\d+)', re.IGNORECASE)
DXA_CYCLES_B_RE = re.compile(r'DXA cycles B:\s*(\d+)', re.IGNORECASE)
DXA_CYCLES_C_RE = re.compile(r'DXA cycles C:\s*(\d+)', re.IGNORECASE)
DXA_CYCLES_A_BITMAP_RE = re.compile(r'DXA cycles A bitmap:\s*(\d+)', re.IGNORECASE)
DXA_CYCLES_B_BITMAP_RE = re.compile(r'DXA cycles B bitmap:\s*(\d+)', re.IGNORECASE)
ROI_ENABLED_RE = re.compile(r'\bENABLE_ROI(?:\s+is\s+defined)?\b', re.IGNORECASE)
MUL_ACTIVE_CYCLES_RE = re.compile(r'Issue Busy processing')
FEOP_ENQ_RE = re.compile(r'\bFEOP-enq\b')
FEDP_ENQ_RE = re.compile(r'\bFEDP-enq\b')


def find_log_file(name):
    """
    Locate the log file by searching common locations:
    - ../build/<name>
    - Provided path / ./<name>
    - Any subdirectory under the current directory
    """
    candidates = [
        os.path.join("..", "build", name),
        name,
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    for root, _, files in os.walk(".", followlinks=False):
        if name in files:
            return os.path.join(root, name)

    return None


def extract_tick(line):
    m = TICK_RE.match(line)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def dedup_sorted(values):
    return sorted(set(values))


def has_tcu_op_enabled(run_meta):
    return "TCU_OP" in run_meta["flags"]


def parse_issue_events(lines):
    events = []
    tcu_issue_ticks = []
    dxa_issue_ticks = []
    dxa_issue_re = re.compile(r'\bdxa_op=(\d+)\b')

    for line in lines:
        if "Issuing TCU u-op" in line:
            tick = extract_tick(line)
            if tick is not None:
                tcu_issue_ticks.append(tick)
            continue
        if "Issuing DXA u-op" in line:
            tick = extract_tick(line)
            dxa_match = dxa_issue_re.search(line)
            dxa_op = int(dxa_match.group(1)) if dxa_match is not None else None
            # The scoreboard logs both transfer launches and descriptor retiles
            # as "Issuing DXA u-op". Only count actual launches on the plot.
            if tick is not None and dxa_op != 6:
                dxa_issue_ticks.append(tick)
            continue
        if "Issuing u-op" in line:
            m = ISSUE_RE.match(line)
            if not m:
                continue
            tick = int(m.group(1))
            wis = int(m.group(2))
            ex_type = int(m.group(3)) if m.group(3) is not None else None
            op_type = int(m.group(4), 16) if m.group(4) is not None else None
            events.append((tick, wis, ex_type, op_type))

    return tcu_issue_ticks, dxa_issue_ticks, events


def parse_other_issue_ticks(lines):
    ticks = []
    for line in lines:
        if "Issuing u-op" not in line:
            continue
        if "Issuing load u-op" in line:
            continue
        if "Issuing store u-op" in line:
            continue
        if "Issuing TCU" in line:
            continue
        tick = extract_tick(line)
        if tick is not None:
            ticks.append(tick)
    return ticks


def parse_pattern_ticks(lines, patterns):
    ticks = []
    for line in lines:
        if any(pat in line for pat in patterns):
            tick = extract_tick(line)
            if tick is not None:
                ticks.append(tick)
    return ticks


def parse_regex_ticks(lines, regex):
    ticks = []
    for line in lines:
        if regex.search(line):
            tick = extract_tick(line)
            if tick is not None:
                ticks.append(tick)
    return ticks


def parse_tcu_commit_ticks(lines):
    return parse_regex_ticks(lines, re.compile(r'commit:.*\bex=TCU\b'))


def parse_tcu_dispatch_flush_flags(lines):
    flags = []
    for line in lines:
        if "issue0-dispatch:" not in line:
            continue
        if "ex=TCU" not in line:
            continue
        if "rs2_data={" not in line:
            continue

        rs2_match = re.search(r'rs2_data=\{([^}]*)\}', line)
        if rs2_match is None:
            continue

        rs2_vals = [v.strip() for v in rs2_match.group(1).split(",") if v.strip()]
        if len(rs2_vals) <= 7:
            continue

        try:
            flush_flag = int(rs2_vals[7], 16) & 0x1
        except ValueError:
            continue
        flags.append(bool(flush_flag))
    return flags


def parse_dxa_done_ticks(lines):
    consume_done_ticks = set(parse_regex_ticks(
        lines,
        re.compile(r'\[wctl-txbar\]\s+consume txbar addr=\d+ done=1\b')
    ))
    dxa_mem_done_ticks = set(parse_regex_ticks(
        lines,
        re.compile(r'\[sfu-txbar\].*dxa_mem\(v=1\b.*\bd=1\)')
    ))
    return sorted(consume_done_ticks & dxa_mem_done_ticks)


def parse_marker_events(lines, regex):
    events = []
    for line in lines:
        if "issue0-dispatcher dispatch:" not in line:
            continue
        if "ex=LSU, op=SW" not in line:
            continue
        if not regex.search(line):
            continue

        rs2_match = re.search(r'rs2_data=\{([^}]*)\}', line)
        tmask_match = re.search(r'tmask=([01]{32})', line)
        if rs2_match is None or tmask_match is None:
            continue

        rs2_vals = [v.strip() for v in rs2_match.group(1).split(",") if v.strip()]
        if not rs2_vals:
            continue

        marker_value = rs2_vals[-1].lower()
        if marker_value not in MARKER_HEXES:
            continue

        # Marker stores are emitted as single-lane scalar stores.
        if tmask_match.group(1).count("1") != 1:
            continue

        tick = extract_tick(line)
        if tick is not None:
            events.append((tick, marker_value))
    return events


def parse_run_metadata(lines):
    meta = {
        "M": None,
        "N": None,
        "K": None,
        "A_sparsity": None,
        "B_sparsity": None,
        "input_dtype": None,
        "output_dtype": None,
        "sparsity_type": None,
        "block_m": None,
        "block_n": None,
        "queue_depth": None,
        "flags": [],
        "tcu_type": None,
        "testbench": None,
        "num_threads": None,
        "warps": None,
    }

    def add_flag(flag_name):
        if flag_name not in meta["flags"]:
            meta["flags"].append(flag_name)

    sparsities = []
    for line in lines:
        if meta["testbench"] is None:
            m_testbench = TESTBENCH_RUN_RE.search(line)
            if m_testbench:
                meta["testbench"] = m_testbench.group(1)
            else:
                m_testbench = TESTBENCH_MAKE_RE.search(line)
                if m_testbench:
                    meta["testbench"] = m_testbench.group(1)

        m_flags = FLAGS_LINE_RE.search(line)
        if m_flags:
            for flag_name in (flag.strip() for flag in m_flags.group(1).split(",")):
                if flag_name:
                    upper_flag = flag_name.upper()
                    if upper_flag.startswith("TCU_TYPE_") and meta["tcu_type"] is None:
                        meta["tcu_type"] = upper_flag
                    elif upper_flag != "TCU_OP":
                        add_flag(upper_flag)
            continue

        if ROI_ENABLED_RE.search(line):
            add_flag("ENABLE_ROI")

        m_tcu_type = TCU_TYPE_DEFINED_RE.search(line)
        if m_tcu_type:
            meta["tcu_type"] = m_tcu_type.group(1).upper()

        if TCU_OP_DEFINED_RE.search(line):
            add_flag("TCU_OP")

        m_threads = CONFIG_THREADS_RE.search(line)
        if m_threads and meta["num_threads"] is None:
            meta["num_threads"] = int(m_threads.group(1))

        m_warps = CONFIG_WARPS_RE.search(line)
        if m_warps and meta["warps"] is None:
            meta["warps"] = int(m_warps.group(1))
            continue

        m_a = MATRIX_A_RE.search(line)
        if m_a:
            m_val, k_val = int(m_a.group(1)), int(m_a.group(2))
            meta["M"] = m_val
            meta["K"] = k_val
            continue

        m_b = MATRIX_B_RE.search(line)
        if m_b:
            k_val, n_val = int(m_b.group(1)), int(m_b.group(2))
            if meta["K"] is None:
                meta["K"] = k_val
            meta["N"] = n_val
            continue

        m_s = ACTUAL_SPARSITY_RE.search(line)
        if m_s:
            sparsities.append(f"{m_s.group(1)}%")
            continue

        m_in_dtype = INPUT_DTYPE_RE.search(line)
        if m_in_dtype:
            meta["input_dtype"] = m_in_dtype.group(1)
            continue

        m_out_dtype = OUTPUT_DTYPE_RE.search(line)
        if m_out_dtype:
            meta["output_dtype"] = m_out_dtype.group(1)
            continue

        m_sparse_mode = SPARSE_MODE_LEVEL_RE.search(line)
        if m_sparse_mode and meta["sparsity_type"] is None:
            meta["sparsity_type"] = int(m_sparse_mode.group(1))
            continue

        m_sparsity_type = SPARSITY_TYPE_RE.search(line)
        if m_sparsity_type and meta["sparsity_type"] is None:
            meta["sparsity_type"] = int(m_sparsity_type.group(1))
            continue

        m_feop_params = FEOP_PARAMS_RE.search(line)
        if m_feop_params and meta["block_m"] is None:
            meta["block_m"] = int(m_feop_params.group(1))
            meta["block_n"] = int(m_feop_params.group(2))
            meta["queue_depth"] = int(m_feop_params.group(3))

    if len(sparsities) >= 1:
        meta["A_sparsity"] = sparsities[0]
    if len(sparsities) >= 2:
        meta["B_sparsity"] = sparsities[1]

    return meta


def parse_perf_info(lines):
    perf_lines = []
    perf_class = None

    for line in lines:
        m_perf = PERF_LINE_RE.match(line)
        if not m_perf:
            continue

        perf_text = m_perf.group(1)
        perf_lines.append(perf_text)
        perf_text_lower = perf_text.lower()

        if "dxa:" in perf_text_lower:
            perf_class = 6
            continue

        if any(token in perf_text_lower for token in (
                "lmem:",
                "coalescer:",
                "icache:",
                "dcache:",
                "l2cache:",
                "l3cache:",
                "memory: reqs=",
        )):
            if perf_class is None or perf_class != 6:
                perf_class = 2
            continue

        if any(token in perf_text_lower for token in (
                "scheduler:",
                "stalls:",
                "inst_mix:",
                "branches:",
                "memory: ifetches=",
                "roofline:",
        )):
            if perf_class is None:
                perf_class = 1

    if perf_class is None and perf_lines:
        perf_class = 0

    all_perf_lines = []
    if perf_class is not None:
        all_perf_lines = [f"PERF{perf_class}: {perf_text}" for perf_text in perf_lines]

    memory_lines = []
    if perf_class == 2:
        for perf_text in perf_lines:
            perf_text_lower = perf_text.lower()
            if perf_text_lower.startswith("instrs="):
                continue
            memory_lines.append(perf_text)

    perf1_lines = []
    if perf_class == 1:
        global_perf_lines = []
        core_perf_lines = []

        for perf_text in perf_lines:
            perf_text_lower = perf_text.lower()
            if perf_text_lower.startswith("instrs="):
                continue
            if re.match(r'^core\d+:\s*', perf_text_lower):
                core_perf_lines.append(perf_text)
            else:
                global_perf_lines.append(perf_text)

        perf1_lines = global_perf_lines if global_perf_lines else core_perf_lines

    return {
        "perf_class": perf_class,
        "all_perf_lines": all_perf_lines,
        "memory_lines": memory_lines,
        "perf1_lines": perf1_lines,
    }


def parse_run_result(lines):
    passed = any(PASS_RE.search(line) for line in lines)
    failed = any(FAIL_RE.search(line) for line in lines)

    error_count = None
    total_errors = None
    for line in lines:
        m_fail_count = FAIL_COUNT_RE.search(line)
        if m_fail_count:
            error_count = int(m_fail_count.group(1))
            total_errors = int(m_fail_count.group(2))

    if passed:
        return {"status": "PASS", "error_count": 0, "total_errors": total_errors}
    if failed or error_count is not None:
        return {"status": "FAIL", "error_count": error_count, "total_errors": total_errors}
    return {"status": "FAIL", "error_count": error_count, "total_errors": total_errors}


def parse_roi_metrics(lines):
    metrics = {
        "total_kernel_cycles": None,
        "total_kernel_instructions": None,
        "kernel_body_cycles": None,
        "kernel_body_instructions": None,
        "tcu_cycles": None,
        "dxa_cycles_a": None,
        "dxa_cycles_b": None,
        "dxa_cycles_c": None,
        "dxa_cycles_a_bitmap": None,
        "dxa_cycles_b_bitmap": None,
        "roi_enabled": False,
        "mul_active_cycles": 0,
        "muls_active": None,
        "tcu_utilization": None,
    }

    metrics["roi_enabled"] = any(ROI_ENABLED_RE.search(line) for line in lines)
    metrics["mul_active_cycles"] = sum(1 for line in lines if MUL_ACTIVE_CYCLES_RE.search(line))

    for line in lines:
        m_total_instr = PERF_INSTRS_RE.search(line)
        if m_total_instr:
            metrics["total_kernel_instructions"] = int(m_total_instr.group(1))
            continue

        m_kernel_body = KERNEL_BODY_CYCLES_RE.search(line)
        if m_kernel_body:
            metrics["kernel_body_cycles"] = int(m_kernel_body.group(1))
            continue

        m_kernel_instr = KERNEL_BODY_INSTR_RE.search(line)
        if m_kernel_instr:
            metrics["kernel_body_instructions"] = int(m_kernel_instr.group(1))
            continue

        m_dxa_a = DXA_CYCLES_A_RE.search(line)
        if m_dxa_a:
            metrics["dxa_cycles_a"] = int(m_dxa_a.group(1))
            continue

        m_dxa_b = DXA_CYCLES_B_RE.search(line)
        if m_dxa_b:
            metrics["dxa_cycles_b"] = int(m_dxa_b.group(1))
            continue

        m_dxa_c = DXA_CYCLES_C_RE.search(line)
        if m_dxa_c:
            metrics["dxa_cycles_c"] = int(m_dxa_c.group(1))
            continue

        m_dxa_a_bitmap = DXA_CYCLES_A_BITMAP_RE.search(line)
        if m_dxa_a_bitmap:
            metrics["dxa_cycles_a_bitmap"] = int(m_dxa_a_bitmap.group(1))
            continue

        m_dxa_b_bitmap = DXA_CYCLES_B_BITMAP_RE.search(line)
        if m_dxa_b_bitmap:
            metrics["dxa_cycles_b_bitmap"] = int(m_dxa_b_bitmap.group(1))

    for line in reversed(lines):
        m_cycles = KERNEL_CYCLES_RE.search(line)
        if m_cycles:
            metrics["total_kernel_cycles"] = int(m_cycles.group(1))
            break

    if metrics["kernel_body_cycles"] is None:
        metrics["kernel_body_cycles"] = metrics["total_kernel_cycles"]

    return metrics


def count_muls_active(lines, run_meta):
    if has_tcu_op_enabled(run_meta):
        enq_re = FEOP_ENQ_RE
        divisor = run_meta["block_m"] or 2
    else:
        enq_re = FEDP_ENQ_RE
        divisor = run_meta["num_threads"] or 32

    enq_count = sum(1 for line in lines if enq_re.search(line))
    return enq_count / divisor


def calculate_tcu_utilization(metrics):
    kernel_body_cycles = metrics["kernel_body_cycles"]
    muls_active = metrics["muls_active"]
    if kernel_body_cycles is None or kernel_body_cycles == 0 or muls_active is None:
        return None
    return muls_active / kernel_body_cycles


def parse_total_kernel_cycles(lines):
    for line in reversed(lines):
        m_cycles = KERNEL_CYCLES_RE.search(line)
        if m_cycles:
            return int(m_cycles.group(1))
    return None


def format_error_count(error_count, total_errors, unknown_text="unknown"):
    if error_count is None:
        return unknown_text
    if total_errors is None:
        return str(error_count)
    return f"{error_count} / {total_errors}"


def write_stats_file(log_path, result, metrics, run_meta, perf_info, xbar_stall_cycles, stall_pct):
    stats_dir = os.path.join(os.path.dirname(__file__), "stats")
    os.makedirs(stats_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(log_path))[0]
    stats_path = os.path.join(stats_dir, f"{base}.stat")

    with open(stats_path, "w") as f:
        testbench_text = format_testbench(run_meta)
        title = format_run_details(run_meta)
        flags_text = format_flags(run_meta)
        feop_params = format_feop_params(run_meta)
        arch_params = format_arch_params(run_meta)
        perf_text = format_perf_class(perf_info)
        if testbench_text:
            f.write(f"{testbench_text}\n")
        if title:
            f.write(f"{title}\n")
        if flags_text:
            f.write(f"{flags_text}\n")
        if feop_params:
            f.write(f"{feop_params}\n")
        if arch_params:
            f.write(f"{arch_params}\n")
        if perf_text:
            f.write(f"{perf_text}\n")
        if testbench_text or title or flags_text or feop_params or arch_params or perf_text:
            f.write("\n")
        f.write(f"{result['status']}\n")
        if result["status"] != "PASS":
            error_text = format_error_count(result["error_count"], result["total_errors"])
            f.write(f"Error count: {error_text}\n")
        total_cycles_text = "N/A" if metrics["total_kernel_cycles"] is None else str(metrics["total_kernel_cycles"])
        total_instr_text = "N/A" if metrics["total_kernel_instructions"] is None else str(metrics["total_kernel_instructions"])
        kernel_body_text = "N/A" if metrics["kernel_body_cycles"] is None else str(metrics["kernel_body_cycles"])
        kernel_instr_text = "N/A" if metrics["kernel_body_instructions"] is None else str(metrics["kernel_body_instructions"])
        mul_active_text = str(metrics["mul_active_cycles"])
        xbar_stalls_text = f"{stall_pct:.2f}% ({xbar_stall_cycles} cycles)"
        f.write(f"Total kernel cycles: {total_cycles_text}\n")
        f.write(f"Total kernel instructions: {total_instr_text}\n")
        f.write(f"Kernel body cycles: {kernel_body_text}\n")
        f.write(f"Kernel body instructions: {kernel_instr_text}\n")
        f.write(f"MUL active cycles: {mul_active_text}\n")
        f.write(f"XBAR stalls: {xbar_stalls_text}\n")
        f.write("\n")
        all_perf_lines = perf_info["all_perf_lines"]
        if all_perf_lines:
            for line in all_perf_lines:
                f.write(f"{line}\n")
            f.write("\n")
        memory_lines = format_memory_section(perf_info)
        if memory_lines:
            f.write("MEMORY (perf=2):\n")
            for line in memory_lines:
                f.write(f"{line}\n")
            f.write("\n")
        perf1_lines = format_perf1_section(perf_info)
        if perf1_lines:
            f.write("PERF (perf=1):\n")
            for line in perf1_lines:
                f.write(f"{line}\n")
            f.write("\n")
        for line in format_roi_variables(metrics, run_meta):
            f.write(f"{line}\n")

    return stats_path


def format_run_details(run_meta):
    details = []
    if run_meta["sparsity_type"] is not None:
        details.append(f"Sparsity Mode:{run_meta['sparsity_type']}")
    if (run_meta["M"] is not None
            and run_meta["N"] is not None
            and run_meta["K"] is not None):
        details.append(f"MxNxK={run_meta['M']}x{run_meta['N']}x{run_meta['K']}")
    if run_meta["A_sparsity"] is not None:
        details.append(f"A sparsity={run_meta['A_sparsity']}")
    if run_meta["B_sparsity"] is not None:
        details.append(f"B sparsity={run_meta['B_sparsity']}")
    if run_meta["input_dtype"] is not None:
        details.append(f"It={run_meta['input_dtype']}")
    return " | ".join(details)


def format_testbench(run_meta):
    if run_meta["testbench"] is None:
        return ""
    return f"Testbench: {run_meta['testbench']}"


def format_feop_params(run_meta):
    if (run_meta["block_m"] is None
            or run_meta["block_n"] is None
            or run_meta["queue_depth"] is None):
        return ""
    return (
        f"BLOCK_M={run_meta['block_m']}, "
        f"BLOCK_N={run_meta['block_n']}, "
        f"XBAR_QUEUE_DEPTH={run_meta['queue_depth']}"
    )


def format_arch_params(run_meta):
    arch_params = []
    if run_meta["num_threads"] is not None:
        arch_params.append(f"Threads: {run_meta['num_threads']}")
    if run_meta["warps"] is not None:
        arch_params.append(f"Warps: {run_meta['warps']}")
    if not arch_params:
        return ""
    return ", ".join(arch_params)


def format_flags(run_meta):
    flags = list(run_meta["flags"])
    if run_meta["tcu_type"] is not None:
        flags.append(run_meta["tcu_type"])
    if not flags:
        return ""

    def flag_sort_key(flag_name):
        if flag_name == "ENABLE_ROI":
            return (0, flag_name)
        if flag_name == "TCU_OP":
            return (1, flag_name)
        if flag_name.startswith("TCU_TYPE_"):
            return (2, flag_name)
        return (3, flag_name)

    ordered_flags = sorted(flags, key=flag_sort_key)
    return f"Flags: {', '.join(ordered_flags)}"


def format_perf_class(perf_info):
    perf_class = perf_info["perf_class"]
    if perf_class is None:
        return ""
    return f"perf={perf_class}"


def format_memory_section(perf_info):
    if perf_info["perf_class"] != 2:
        return []
    return perf_info["memory_lines"]


def format_perf1_section(perf_info):
    if perf_info["perf_class"] != 1:
        return []

    formatted_lines = []
    for line in perf_info["perf1_lines"]:
        if line.lower().startswith("stalls:"):
            line = re.sub(r'\bscrb=', 'scoreboard=', line)
            line = re.sub(r'\bopds=', 'operands=', line)
        formatted_lines.append(line)
    return formatted_lines


def format_roi_variables(metrics, run_meta):
    def metric_text(key):
        value = metrics[key]
        if value is None:
            return "N/A"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def utilization_text():
        value = metrics["tcu_utilization"]
        return "N/A" if value is None else f"{value * 100.0:.2f}%"

    lines = [
        f"TCU cycles (from first TCU issue to last TCU commit): {metric_text('tcu_cycles')}",
        f"MULs active: {metric_text('muls_active')}",
        f"TCU utilization: {utilization_text()}",
    ]

    return lines


def classify_mem_line(line):
    s = line.lower()
    if "mem:" in s:
        if "read request" in s:
            return "rd_req"
        if "write request" in s:
            return "wr_req"
        if "read rsp" in s or "read response" in s:
            return "rd_rsp"
        if "write rsp" in s or "write response" in s:
            return "wr_rsp"
    if not any(tok in s for tok in ("mem", "lmem", "cache", "dcache", "icache", "bank", "core")):
        return None
    if "rd_req" in s:
        return "rd_req"
    if "wr_req" in s:
        return "wr_req"
    if "rd_rsp" in s:
        return "rd_rsp"
    if "wr_rsp" in s:
        return "wr_rsp"
    if "rd-rsp" in s or "rsp-rd" in s:
        return "rd_rsp"
    if "wr-rsp" in s or "rsp-wr" in s:
        return "wr_rsp"
    if "rd-req" in s or "req-rd" in s:
        return "rd_req"
    if "wr-req" in s or "req-wr" in s:
        return "wr_req"
    return None


def parse_mem_events(lines, include_re=None, exclude_re=None):
    mem = {
        "rd_req_global": [],
        "wr_req_global": [],
        "rd_rsp_global": [],
        "wr_rsp_global": [],
        "rd_req_lmem": [],
        "wr_req_lmem": [],
        "rd_rsp_lmem": [],
        "wr_rsp_lmem": [],
    }
    for line in lines:
        if include_re and not include_re.search(line):
            continue
        if exclude_re and exclude_re.search(line):
            continue
        kind = classify_mem_line(line)
        if not kind:
            continue
        if "mem:" in line.lower() or "lmem" in line.lower():
            scope = "lmem"
        else:
            scope = "global"
        tick = extract_tick(line)
        if tick is None:
            continue
        mem[f"{kind}_{scope}"].append(tick)
    return mem


def parse_lmem_read_rsp_matrix_events(lines, include_re=None, exclude_re=None):
    events = []
    for line in lines:
        if include_re and not include_re.search(line):
            continue
        if exclude_re and exclude_re.search(line):
            continue
        m = LMEM_READ_RSP_MATRIX_RE.search(line)
        if not m:
            continue
        tick = extract_tick(line)
        if tick is None:
            continue
        events.append((tick, m.group(1)))
    return events


def auto_detect_tcu_ex_type(issue_events):
    ex_counts = {}
    for _, _, ex_type, _ in issue_events:
        if ex_type is None:
            continue
        ex_counts[ex_type] = ex_counts.get(ex_type, 0) + 1
    if not ex_counts:
        return None
    # Prefer the most frequent non-zero ex_type (avoids ALU-heavy streams).
    non_zero = {k: v for k, v in ex_counts.items() if k != 0}
    if non_zero:
        return max(non_zero.items(), key=lambda kv: kv[1])[0]
    return max(ex_counts.items(), key=lambda kv: kv[1])[0]


def pair_tcu_issue_commit_ticks(issue_ticks, commit_ticks):
    """
    Greedily pair each issue tick with the first commit tick at or after it.
    Preserves chronological ordering and avoids reusing commits.
    """
    issues = sorted(issue_ticks)
    commits = sorted(commit_ticks)
    pairs = []
    ci = 0
    for it in issues:
        while ci < len(commits) and commits[ci] < it:
            ci += 1
        if ci >= len(commits):
            break
        pairs.append((it, commits[ci]))
        ci += 1
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract TCU/FEOP/memory stats from a Vortex run log."
    )
    parser.add_argument(
        "name",
        help="Log file name; searched in ../build, current dir, and subdirectories."
    )
    parser.add_argument(
        "--tcu-ex-type",
        type=int,
        help="ex_type value to treat as TCU issue when 'Issuing TCU u-op' is absent."
    )
    parser.add_argument(
        "--tcu-op-type",
        type=lambda s: int(s, 0),
        default=None,
        help="Filter TCU issue events by op_type (hex or decimal). "
             "Default: 0x3 when the log says 'TCU_OP defined', otherwise no op_type filter."
    )
    parser.add_argument(
        "--feop-pattern",
        action="append",
        default=["FEOP-enq", "FEOP execution fired"],
        help="Substring to identify FEOP assignment lines. Repeatable."
    )
    parser.add_argument(
        "--mem-include",
        default=r"MEM:|LMEM",
        help="Regex to include only matching memory lines (default: MEM:|LMEM)."
    )
    parser.add_argument(
        "--mem-exclude",
        help="Regex to exclude matching memory lines."
    )
    parser.add_argument(
        "--no-dedup-mem",
        action="store_true",
        help="Do not de-duplicate memory events per tick."
    )
    parser.add_argument(
        "--no-dedup-feop",
        action="store_true",
        help="Do not de-duplicate FEOP events per tick."
    )
    args = parser.parse_args()

    log_path = find_log_file(args.name)
    if not log_path:
        print("Error: log file not found under ../build, current directory, or any subdirectory.")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()
    run_result = parse_run_result(lines)
    roi_metrics = parse_roi_metrics(lines)
    total_cycles = 0
    for line in lines:
        tick = extract_tick(line)
        if tick is not None and tick > total_cycles:
            total_cycles = tick
    error_count = run_result["error_count"] if run_result["error_count"] is not None else 0
    has_errors = run_result["status"] != "PASS"
    run_meta = parse_run_metadata(lines)
    roi_metrics["muls_active"] = count_muls_active(lines, run_meta)
    roi_metrics["tcu_utilization"] = calculate_tcu_utilization(roi_metrics)
    perf_info = parse_perf_info(lines)

    tcu_op_enabled = has_tcu_op_enabled(run_meta)
    tcu_op_type = args.tcu_op_type
    if tcu_op_type is None and tcu_op_enabled:
        tcu_op_type = 0x3

    # Issue events (TCU + all scoreboard issues)
    if tcu_op_enabled:
        tcu_issue_ticks = parse_pattern_ticks(lines, ["Issuing TCU MMA_OP"])
    else:
        tcu_issue_ticks = parse_pattern_ticks(lines, [
            "Issuing TCU u-op",
            "Issuing TCU MMA_OP",
        ])
    tcu_commit_ticks_all = parse_tcu_commit_ticks(lines)
    tcu_dispatch_flush_flags = parse_tcu_dispatch_flush_flags(lines)
    # Stage markers: only consider explicit MARKER writes (0x12345678).
    marker_events = parse_marker_events(lines, MARKER_RE)
    parsed_tcu_issue_ticks, dxa_issue_ticks, issue_events = parse_issue_events(lines)
    if not tcu_op_enabled:
        tcu_issue_ticks.extend(parsed_tcu_issue_ticks)
    load_issue_ticks = parse_pattern_ticks(lines, ["Issuing load u-op"])
    store_issue_ticks = parse_pattern_ticks(lines, ["Issuing store u-op"])
    other_issue_ticks = parse_other_issue_ticks(lines)
    c_accum_ticks = parse_regex_ticks(lines, ACCU_C_ACCUM_RE)
    xbar_stall_ticks = parse_pattern_ticks(lines, [
        "[feop_accu]: ERROR: xbar queues are full - must stall",
        "[feop_accu]: xbar queues are full - must stall",
    ])
    dxa_done_ticks = parse_dxa_done_ticks(lines)

    tcu_ex_type = args.tcu_ex_type
    if tcu_ex_type is None and not tcu_issue_ticks:
        tcu_ex_type = auto_detect_tcu_ex_type(issue_events)
        if tcu_ex_type is not None:
            print(f"Auto-detected TCU ex_type={tcu_ex_type} (most frequent non-zero ex_type)")

    for tick, _, ex_type, op_type in issue_events:
        if tcu_op_type is not None and op_type != tcu_op_type:
            continue
        if tcu_ex_type is not None and ex_type != tcu_ex_type:
            continue
        tcu_issue_ticks.append(tick)
    if tcu_issue_ticks:
        tcu_issue_ticks = dedup_sorted(tcu_issue_ticks)
    if dxa_issue_ticks:
        dxa_issue_ticks = dedup_sorted(dxa_issue_ticks)
    if dxa_done_ticks:
        dxa_done_ticks = dedup_sorted(dxa_done_ticks)
    tcu_commit_ticks_raw = list(tcu_commit_ticks_all)
    if tcu_commit_ticks_all:
        tcu_commit_ticks_all = dedup_sorted(tcu_commit_ticks_all)
    if tcu_dispatch_flush_flags:
        if len(tcu_dispatch_flush_flags) != len(tcu_commit_ticks_raw):
            print(
                "Warning: TCU dispatch/commit count mismatch "
                f"(dispatches={len(tcu_dispatch_flush_flags)}, commits={len(tcu_commit_ticks_raw)}); "
                "flush commit filtering will use chronological pairing."
            )
        tcu_commit_ticks = [
            tick for tick, flush_flag in zip(tcu_commit_ticks_raw, tcu_dispatch_flush_flags)
            if flush_flag
        ]
        tcu_commit_ticks = dedup_sorted(tcu_commit_ticks)
        if not tcu_commit_ticks and tcu_commit_ticks_all:
            print("No flush-tagged TCU commits found; falling back to all TCU commit ticks.")
            tcu_commit_ticks = list(tcu_commit_ticks_all)
    else:
        tcu_commit_ticks = list(tcu_commit_ticks_all)
    marker_ticks = [t for t, value in marker_events if value == MARKER_HEX.lower()]
    if marker_ticks:
        marker_ticks = dedup_sorted(marker_ticks)
    green_marker_ticks = [t for t, value in marker_events if value == GREEN_MARKER_HEX.lower()]
    if green_marker_ticks:
        green_marker_ticks = dedup_sorted(green_marker_ticks)
    blue_marker_ticks = [t for t, value in marker_events if value == BLUE_MARKER_HEX.lower()]
    if blue_marker_ticks:
        blue_marker_ticks = dedup_sorted(blue_marker_ticks)
    pre_tcu_marker_ticks = [t for t, value in marker_events if value == PRE_TCU_MARKER_HEX.lower()]
    if pre_tcu_marker_ticks:
        pre_tcu_marker_ticks = dedup_sorted(pre_tcu_marker_ticks)

    first_tcu_tick = min(tcu_issue_ticks) if tcu_issue_ticks else None
    last_tcu_commit = max(tcu_commit_ticks_all) if tcu_commit_ticks_all else None

    # FEOP assignment (or fallback to FEOP accu activity)
    feop_ticks = parse_pattern_ticks(lines, args.feop_pattern)
    if not feop_ticks:
        feop_ticks = parse_pattern_ticks(lines, ["[feop_accu]"])
        if feop_ticks:
            print("FEOP patterns not found; using [feop_accu] activity as FEOP work markers.")

    # Memory events
    include_re = re.compile(args.mem_include) if args.mem_include else None
    exclude_re = re.compile(args.mem_exclude) if args.mem_exclude else None
    mem = parse_mem_events(lines, include_re=include_re, exclude_re=exclude_re)
    lmem_read_rsp_matrix_events = parse_lmem_read_rsp_matrix_events(
        lines, include_re=include_re, exclude_re=exclude_re
    )
    mem_right = {k: list(v) for k, v in mem.items()}
    lmem_read_rsp_matrix_events_right = list(lmem_read_rsp_matrix_events)
    if first_tcu_tick is not None:
        mem_right["wr_req_global"] = [t for t in mem_right["wr_req_global"] if t >= first_tcu_tick]
        mem_right["wr_req_lmem"] = [t for t in mem_right["wr_req_lmem"] if t >= first_tcu_tick]
        lmem_read_rsp_matrix_events_right = [
            (t, m) for (t, m) in lmem_read_rsp_matrix_events_right if t >= first_tcu_tick
        ]

    if not args.no_dedup_mem:
        mem = {k: dedup_sorted(v) for k, v in mem.items()}
    feop_ticks = [t for t in feop_ticks if t >= 4]
    if not args.no_dedup_feop:
        feop_ticks = dedup_sorted(feop_ticks)
    if xbar_stall_ticks:
        xbar_stall_ticks = dedup_sorted(xbar_stall_ticks)
    if other_issue_ticks:
        other_issue_ticks = dedup_sorted(other_issue_ticks)

    # FEOP window metrics: xbar stall cycles are counted in the FEOP window, but the
    # stall percentage is normalized by MUL-active cycles plus stall cycles.
    feop_first_tick = min(feop_ticks) if feop_ticks else None
    feop_last_tick = max(feop_ticks) if feop_ticks else None
    feop_cycles = 0
    feop_assigned = len(feop_ticks)
    xbar_stall_cycles = 0
    stall_pct = 0.0
    if feop_first_tick is not None and feop_last_tick is not None:
        feop_cycles = (feop_last_tick - feop_first_tick + 1)
        xbar_stalls_in_feop = [
            t for t in xbar_stall_ticks
            if feop_first_tick <= t <= feop_last_tick
        ]
        xbar_stall_cycles = len(xbar_stalls_in_feop)

    # TCU issue->commit stall metrics (per paired TCU op).
    tcu_pairs = pair_tcu_issue_commit_ticks(tcu_issue_ticks, tcu_commit_ticks_all)
    tcu_total_cycles = 0
    tcu_active_cycles = 0
    tcu_stall_cycles = 0
    tcu_stall_pct = None
    active_ticks_tcu = dedup_sorted(
        mem_right["rd_req_lmem"]
        + mem_right["wr_req_global"]
        + mem_right["wr_req_lmem"]
        + c_accum_ticks
        + feop_ticks
    )
    for issue_t, commit_t in tcu_pairs:
        if commit_t < issue_t:
            continue
        total_cycles_this = ((commit_t - issue_t) // 2) + 1
        if total_cycles_this <= 0:
            continue
        active_cycle_slots = set()
        for t in active_ticks_tcu:
            if t < issue_t or t > commit_t:
                continue
            active_cycle_slots.add((t - issue_t) // 2)
        active_cycles_this = min(len(active_cycle_slots), total_cycles_this)
        stall_cycles_this = total_cycles_this - active_cycles_this

        tcu_total_cycles += total_cycles_this
        tcu_active_cycles += active_cycles_this
        tcu_stall_cycles += stall_cycles_this
    if tcu_total_cycles > 0:
        tcu_stall_pct = (tcu_stall_cycles / tcu_total_cycles) * 100.0
        roi_metrics["tcu_cycles"] = tcu_total_cycles

    mul_active_cycles = roi_metrics["mul_active_cycles"]
    xbar_stall_denom = xbar_stall_cycles + mul_active_cycles
    if xbar_stall_denom > 0:
        stall_pct = (xbar_stall_cycles / xbar_stall_denom) * 100.0

    if not any([
        tcu_issue_ticks,
        feop_ticks,
        mem["rd_req_global"],
        mem["wr_req_global"],
        mem["rd_rsp_global"],
        mem["wr_rsp_global"],
        mem["rd_req_lmem"],
        mem["wr_req_lmem"],
        mem["rd_rsp_lmem"],
        mem["wr_rsp_lmem"],
        tcu_commit_ticks,
        marker_ticks,
        green_marker_ticks,
        blue_marker_ticks,
        xbar_stall_ticks,
        dxa_issue_ticks,
        dxa_done_ticks,
        other_issue_ticks,
        load_issue_ticks,
        store_issue_ticks,
    ]):
        print("No matching events found (TCU/FEOP/memory).")
        return

    if marker_events:
        marker_events_sorted = sorted(marker_events, key=lambda x: x[0])
        unique_markers = []
        seen = set()
        for tick, value in marker_events_sorted:
            key = (tick, value)
            if key in seen:
                continue
            seen.add(key)
            unique_markers.append((tick, value))
        print("Marker ticks:")
        for tick, value in unique_markers:
            print(f"  {value} @ tick {tick}")

    marker_to_tcu_delta = None
    pre_tcu_marker_tick = None
    pre_tcu_issue_tick = None
    if pre_tcu_marker_ticks and tcu_issue_ticks:
        pre_tcu_marker_tick = pre_tcu_marker_ticks[0]
        pre_tcu_issue_tick = next((t for t in tcu_issue_ticks if t >= pre_tcu_marker_tick), None)
        if pre_tcu_issue_tick is not None:
            marker_to_tcu_delta = pre_tcu_issue_tick - pre_tcu_marker_tick

    stats_path = write_stats_file(
        log_path,
        run_result,
        roi_metrics,
        run_meta,
        perf_info,
        xbar_stall_cycles,
        stall_pct,
    )
    print(f"Saved stats to {stats_path}")
    return


if __name__ == "__main__":
    main()
