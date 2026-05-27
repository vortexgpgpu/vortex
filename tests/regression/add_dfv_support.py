#!/usr/bin/env python3
"""Add DFV stall injection support to all regression tests.

Modifies common.h, kernel.cpp, and main.cpp for each test to:
1. Add DFV CSR defines and enable_dfv_test field to kernel_arg_t
2. Add DFV setup/cleanup in kernel.cpp main() (device side)
3. Add -d flag parsing and enable_dfv_test plumbing in main.cpp (host side)

Tests that already have DFV support (demo, conv3) are skipped.
"""

import os
import re
import sys

REG_DIR = os.path.dirname(os.path.abspath(__file__))
SKIP = {"demo", "conv3"}

DFV_DEFINES = """\

//==============================================================================
// DFV (Design-for-Verification) CSR Definitions
//==============================================================================
#define VX_CSR_DFV_CTRL           0x7C0
#define VX_CSR_DFV_ICACHE_STALL   0x7C1
#define VX_CSR_DFV_RANDOM_SEED    0x7C2
#define VX_CSR_DFV_STALL_THRESHOLD 0x7C3
#define VX_CSR_DFV_DCACHE_STALL   0x7C4
#define VX_CSR_DFV_WRITEBACK_STALL 0x7C5

"""

DFV_SETUP = """\
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_STALL_THRESHOLD, 64);
        csr_write(VX_CSR_DFV_ICACHE_STALL, 1);
        csr_write(VX_CSR_DFV_DCACHE_STALL, 1);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 1);
    }
"""

errors = []


def patch_common_h(path):
    with open(path) as f:
        text = f.read()
    if "VX_CSR_DFV_CTRL" in text:
        return "skip (already has DFV)"

    # Insert DFV defines before typedef struct
    if "typedef struct {" not in text:
        errors.append(f"{path}: no 'typedef struct {{' found")
        return "ERROR"
    text = text.replace("typedef struct {", DFV_DEFINES + "typedef struct {")

    # Insert enable_dfv_test as last field before } kernel_arg_t;
    if "} kernel_arg_t;" not in text:
        errors.append(f"{path}: no '}} kernel_arg_t;' found")
        return "ERROR"
    text = text.replace("} kernel_arg_t;", "  uint32_t enable_dfv_test;\n} kernel_arg_t;")

    with open(path, "w") as f:
        f.write(text)
    return "ok"


def patch_kernel_cpp(path):
    with open(path) as f:
        lines = f.readlines()
    if any("VX_CSR_DFV_CTRL" in l for l in lines):
        return "skip (already has DFV)"

    # Check for csr_read(VX_CSR_MSCRATCH)
    mscratch_idx = None
    for i, line in enumerate(lines):
        if "csr_read(VX_CSR_MSCRATCH)" in line:
            mscratch_idx = i
            break
    if mscratch_idx is None:
        errors.append(f"{path}: no csr_read(VX_CSR_MSCRATCH) found")
        return "ERROR"

    # Find the return vx_spawn_threads(...) line
    spawn_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("return vx_spawn_threads("):
            spawn_idx = i
            break
    if spawn_idx is None:
        errors.append(f"{path}: no 'return vx_spawn_threads(' found")
        return "ERROR"

    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if i == mscratch_idx:
            # Emit the csr_read line, then DFV setup
            out.append(line)
            out.append(DFV_SETUP)
            i += 1
            continue

        if i == spawn_idx:
            # Collect the full spawn statement (may span multiple lines)
            stmt = line
            while ";" not in stmt:
                i += 1
                stmt += lines[i]
            # Extract indentation
            indent = line[: len(line) - len(line.lstrip())]
            # Extract the vx_spawn_threads(...) expression
            m = re.search(r"return\s+(vx_spawn_threads\(.+\))\s*;", stmt, re.DOTALL)
            if m:
                spawn_expr = m.group(1).strip()
                out.append(f"{indent}int __ret = {spawn_expr};\n")
                out.append(f"{indent}if (arg->enable_dfv_test) {{\n")
                out.append(f"{indent}    csr_write(VX_CSR_DFV_CTRL, 0);\n")
                out.append(f"{indent}}}\n")
                out.append(f"{indent}return __ret;\n")
            else:
                errors.append(f"{path}: could not parse spawn statement")
                out.append(stmt)
            i += 1
            continue

        out.append(line)
        i += 1

    with open(path, "w") as f:
        f.writelines(out)
    return "ok"


def patch_main_cpp(path):
    with open(path) as f:
        text = f.read()
    if "enable_dfv_test" in text:
        return "skip (already has DFV)"

    # 1. Add enable_dfv_test variable after kernel_arg declaration
    if "kernel_arg_t kernel_arg" in text:
        text = re.sub(
            r"(kernel_arg_t kernel_arg\s*=\s*\{[^}]*\};\n)",
            r"\1bool enable_dfv_test = false;\n",
            text,
        )
    else:
        errors.append(f"{path}: no 'kernel_arg_t kernel_arg' found")
        return "ERROR"

    # 2. Add 'd' to getopt string (insert at start of option string)
    if 'getopt(argc, argv, "' in text:
        text = text.replace('getopt(argc, argv, "', 'getopt(argc, argv, "d')
    else:
        errors.append(f"{path}: no getopt() call found")
        return "ERROR"

    # 3. Add case 'd' handler before default:
    if "    default:" in text:
        text = text.replace(
            "    default:",
            "    case 'd':\n      enable_dfv_test = true;\n      break;\n    default:",
            1,
        )
    else:
        errors.append(f"{path}: no 'default:' in switch")
        return "ERROR"

    # 4. Set kernel_arg.enable_dfv_test before upload
    if "// upload kernel argument" in text:
        text = text.replace(
            "// upload kernel argument",
            "kernel_arg.enable_dfv_test = enable_dfv_test ? 1 : 0;\n\n  // upload kernel argument",
        )
    elif "vx_upload_bytes" in text:
        text = re.sub(
            r"(  RT_CHECK\(vx_upload_bytes)",
            r"  kernel_arg.enable_dfv_test = enable_dfv_test ? 1 : 0;\n\n\1",
            text,
            count=1,
        )
    else:
        errors.append(f"{path}: no upload point found")
        return "ERROR"

    with open(path, "w") as f:
        f.write(text)
    return "ok"


def main():
    print("Adding DFV support to regression tests...\n")
    for d in sorted(os.listdir(REG_DIR)):
        dp = os.path.join(REG_DIR, d)
        if not os.path.isdir(dp) or d in SKIP:
            continue

        ch = os.path.join(dp, "common.h")
        kc = os.path.join(dp, "kernel.cpp")
        mc = os.path.join(dp, "main.cpp")

        print(f"[{d}]")
        for label, path, fn in [
            ("common.h", ch, patch_common_h),
            ("kernel.cpp", kc, patch_kernel_cpp),
            ("main.cpp", mc, patch_main_cpp),
        ]:
            if os.path.exists(path):
                result = fn(path)
                print(f"  {label}: {result}")
            else:
                print(f"  {label}: not found")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\nAll tests patched successfully.")


if __name__ == "__main__":
    main()
