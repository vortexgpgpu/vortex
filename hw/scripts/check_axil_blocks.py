#!/usr/bin/env python3
"""Fail if any AXI-Lite 16-byte block in the CP regfile is partially decoded.

Measured on V80 silicon (2026-08-14, 80-address sweep, 80/80 fit): a
16-byte-aligned block answers reads if and only if all four of its words are
decoded. A partially-populated block DECERRs on *every* word, including the
words that are implemented -- the shell -> s_axi_control path resolves at
128-bit granularity and rejects a block it cannot fully resolve, so the
per-word decode in VX_cp_axil_regfile never gets consulted.

That defect cost ten days to find. It is invisible in simulation, because no
simulator models the crossbar; it only appears on hardware, where the symptom
is a register reading 0xFFFFFFFF for no reason visible in the RTL. This check
makes it a build failure instead.

Usage:  check_axil_blocks.py [path/to/VX_cp_axil_regfile.sv]
Exit:   0 = every block fully populated, 1 = a partial block was found.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "rtl/cp/VX_cp_axil_regfile.sv"

# Offsets accepted by is_decoded(). Globals appear as is_global(addr, 8'hNN);
# per-queue offsets appear as bare 6'hNN literals in the case label.
GLOBAL_RE = re.compile(r"is_global\s*\(\s*addr\s*,\s*8'h([0-9A-Fa-f]{2})\s*\)")
QUEUE_RE = re.compile(r"6'h([0-9A-Fa-f]{2})")


def decoded_offsets(text):
    """Return (globals, queue_offsets) taken from the body of is_decoded()."""
    start = text.find("function automatic logic is_decoded")
    if start < 0:
        sys.exit("check_axil_blocks: is_decoded() not found -- has it been renamed?")
    end = text.find("endfunction", start)
    body = text[start:end]

    globals_ = {int(m, 16) for m in GLOBAL_RE.findall(body)}

    # Only the case label inside decode_queue(...) lists queue offsets; take
    # everything between that `case (off)` and its `default:`.
    qstart = body.find("case (off)")
    qend = body.find("default:", qstart) if qstart >= 0 else -1
    queue = set()
    if qstart >= 0 and qend > qstart:
        queue = {int(m, 16) for m in QUEUE_RE.findall(body[qstart:qend])}
    return globals_, queue


def partial_blocks(offsets):
    """Group offsets into 16-byte blocks; return those missing any of 4 words."""
    blocks = defaultdict(set)
    for off in offsets:
        blocks[off & ~0xF].add(off & 0xF)
    bad = []
    for base in sorted(blocks):
        missing = {0x0, 0x4, 0x8, 0xC} - blocks[base]
        if missing:
            bad.append((base, sorted(missing)))
    return bad


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.is_file():
        sys.exit(f"check_axil_blocks: no such file: {path}")

    globals_, queue = decoded_offsets(path.read_text())
    if not globals_ and not queue:
        sys.exit("check_axil_blocks: parsed no offsets -- the decoder shape changed")

    failures = []
    for label, offsets in (("global", globals_), ("per-queue", queue)):
        for base, missing in partial_blocks(offsets):
            words = ", ".join(f"0x{base + m:02X}" for m in missing)
            failures.append(
                f"  {label} block 0x{base:02X}: missing {words}"
            )

    if failures:
        print("check_axil_blocks: partially-populated 16-byte block(s) in "
              f"{path}", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        print(
            "\nEvery word of a 16-byte block must be decoded, or the whole "
            "block DECERRs on hardware\n(including the words that ARE "
            "implemented). Pad the block in is_decoded() and return a\nvalue "
            "for it in read_reg() -- zero is fine for padding.",
            file=sys.stderr,
        )
        return 1

    n = len(globals_) + len(queue)
    print(f"check_axil_blocks: OK -- {n} decoded offsets, every 16-byte block "
          "fully populated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
