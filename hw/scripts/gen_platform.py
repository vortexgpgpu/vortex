#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve a Xilinx card descriptor and emit what a build flow consumes.

Card facts live in hw/syn/xilinx/platforms/<card>.toml, one file per card. A
flow names the card the way it already does -- the XRT flow by platform XSA
and FPGA family, the AVED flow by device part -- and this script resolves the
descriptor and emits one artifact:

  mk          make fragment for hw/syn/xilinx/{xrt,aved}/Makefile
  cfg         slashkit linker config (AVED flow)
  firesim-mk  make fragment for hw/syn/firesim and sim/firesim
  card        the resolved card name
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional

PLATFORM_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "syn", "xilinx", "platforms"))

FLOWS = ("xrt", "aved")

# The AFU kernel instance name each linker assigns.
KERNEL_INSTANCE = {"xrt": "vortex_afu_1", "aved": "vortex_afu_0"}


# Width of the PLATFORM_MEMORY_OFFSET literal the AFU adds to device addresses.
OFFSET_BITS = 40


class DescriptorError(Exception):
    pass


def verilog_offset(base: int) -> str:
    # Sized hex, escaped for make and the shell: Vivado truncates an oversized
    # decimal constant to 0 with only a warning.
    return f"{OFFSET_BITS}\\'h{base:X}"


@dataclass
class Card:
    name: str
    path: str
    flow: str
    part: str
    family: str
    match_xsa: List[str]
    match_family: List[str]
    match_device: List[str]
    banks: int
    addr_width: int
    merged: bool
    base: int
    mem_tags: List[str]
    host_tag: str
    host_refuse: List[str]
    host_refuse_why: str
    kernel_mhz: int
    axil_block_bytes: int
    firesim: dict = field(default_factory=dict)

    def specificity(self) -> int:
        # An exact XSA or device pattern beats a family match, which beats a
        # catch-all, so a card-specific descriptor always wins.
        if any(p != "*" for p in self.match_xsa + self.match_device):
            return 2
        if self.match_family:
            return 1
        return 0


def _require(d: dict, key: str, path: str, ctx: str):
    if key not in d:
        raise DescriptorError(f"{path}: missing [{ctx}] {key}")
    return d[key]


def load_card(path: str) -> Card:
    with open(path, "rb") as f:
        t = tomllib.load(f)
    card = _require(t, "card", path, "card")
    mem = _require(t, "memory", path, "memory")
    host = t.get("host", {})
    clock = t.get("clock", {})
    control = t.get("control", {})
    flow = _require(card, "flow", path, "card")
    if flow not in FLOWS:
        raise DescriptorError(f"{path}: flow must be one of {FLOWS} (got '{flow}')")
    c = Card(
        name=_require(card, "name", path, "card"),
        path=path,
        flow=flow,
        part=card.get("part", ""),
        family=card.get("family", ""),
        match_xsa=list(card.get("match_xsa", [])),
        match_family=list(card.get("match_family", [])),
        match_device=list(card.get("match_device", [])),
        banks=int(_require(mem, "banks", path, "memory")),
        addr_width=int(_require(mem, "addr_width", path, "memory")),
        merged=bool(mem.get("merged", False)),
        base=int(mem.get("base", 0)),
        mem_tags=list(mem.get("tags", [])),
        host_tag=str(host.get("tag", "")),
        host_refuse=list(host.get("refuse", [])),
        host_refuse_why=str(host.get("refuse_why", "")),
        kernel_mhz=int(clock.get("kernel_mhz", 300)),
        axil_block_bytes=int(control.get("axil_block_bytes", 0)),
        firesim=dict(t.get("firesim", {})),
    )
    if not 0 <= c.base < (1 << OFFSET_BITS):
        raise DescriptorError(f"{path}: [memory] base 0x{c.base:x} does not fit {OFFSET_BITS} bits")
    if len(c.mem_tags) > (1 if c.merged else c.banks):
        raise DescriptorError(f"{path}: {len(c.mem_tags)} memory tags for "
                              f"{1 if c.merged else c.banks} memory port(s)")
    if c.host_tag and c.host_tag in c.host_refuse:
        raise DescriptorError(f"{path}: host tag '{c.host_tag}' is in its own refuse list")
    return c


def load_all(platform_dir: str) -> List[Card]:
    cards = []
    for fn in sorted(os.listdir(platform_dir)):
        if fn.endswith(".toml"):
            cards.append(load_card(os.path.join(platform_dir, fn)))
    names = [c.name for c in cards]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        raise DescriptorError(f"duplicate card names: {sorted(dups)}")
    return cards


def resolve(cards: List[Card], flow: str, xsa: str, family: str, device: str, name: str) -> Card:
    pool = [c for c in cards if c.flow == flow]
    if name:
        hits = [c for c in pool if c.name == name]
        if not hits:
            raise DescriptorError(f"no {flow} card named '{name}'; known: {sorted(c.name for c in pool)}")
        return hits[0]

    def matches(c: Card) -> bool:
        if xsa and any(fnmatch.fnmatchcase(xsa, p) for p in c.match_xsa):
            return True
        if device and any(fnmatch.fnmatchcase(device, p) for p in c.match_device):
            return True
        if family and family in c.match_family:
            return True
        return False

    hits = [c for c in pool if matches(c)]
    if not hits:
        what = xsa or device or family or "<nothing>"
        raise DescriptorError(f"no {flow} card descriptor matches '{what}'; known: "
                              f"{sorted(c.name for c in pool)} (add one under {PLATFORM_DIR})")
    best = max(c.specificity() for c in hits)
    top = [c for c in hits if c.specificity() == best]
    if len(top) > 1:
        raise DescriptorError(f"ambiguous card for '{xsa or device or family}': "
                              f"{sorted(c.name for c in top)}")
    return top[0]


def emit_mk(c: Card) -> str:
    # override: a command-line CONFIGS=... must not drop the card settings.
    ov = "override "
    inst = KERNEL_INSTANCE[c.flow]
    lines = [f"# Generated by gen_platform.py from {os.path.relpath(c.path, PLATFORM_DIR)}. Do not edit.",
             f"PLATFORM_CARD := {c.name}"]
    if c.part:
        lines.append(f"PLATFORM_PART := {c.part}")
    lines.append(f"PLATFORM_AXIL_BLOCK_BYTES := {c.axil_block_bytes}")
    lines.append(f"PLATFORM_HOST_REFUSE := {' '.join(c.host_refuse)}")
    lines.append(f"PLATFORM_HOST_REFUSE_WHY := {c.host_refuse_why}")

    lines.append(f"{ov}CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS={c.banks} "
                 f"-DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH={c.addr_width}")
    if c.merged:
        lines.append(f"{ov}CONFIGS += -DPLATFORM_MERGED_MEMORY_INTERFACE")
    if c.base:
        lines.append(f"{ov}CONFIGS += -DPLATFORM_MEMORY_OFFSET={verilog_offset(c.base)}")

    if c.flow == "xrt":
        if c.host_tag:
            lines.append(f"HOST_TAG = {c.host_tag}")
            lines.append(f"VPP_FLAGS += --connectivity.sp {inst}.m_axi_host:$(HOST_TAG)")
        for i, tag in enumerate(c.mem_tags):
            lines.append(f"VPP_FLAGS += --connectivity.sp {inst}.m_axi_mem_{i}:{tag}")
    else:
        if len(c.mem_tags) != 1:
            raise DescriptorError(f"{c.path}: the aved flow binds exactly one memory port")
        lines.append(f"MEM_TAG = {c.mem_tags[0]}")
        lines.append(f"HOST_TAG = {c.host_tag}")
        if c.part:
            lines.append(f"DEVICE ?= {c.part}")
    lines.append(f"KERNEL_FREQ ?= {c.kernel_mhz}")
    return "\n".join(lines) + "\n"


def emit_cfg(c: Card, mem_tag: str, host_tag: str, kernel_mhz: int) -> str:
    if c.flow != "aved":
        raise DescriptorError(f"card '{c.name}' is an {c.flow} card; cfg is for the aved flow")
    check_host_tag(c, host_tag)
    inst = KERNEL_INSTANCE["aved"]
    return "\n".join([
        f"# Generated by gen_platform.py from {os.path.relpath(c.path, PLATFORM_DIR)}. Do not edit.",
        "[connectivity]",
        "# Bind to the compute-only static shell. The service shell adds the DCMAC",
        "# ethernet stack the AFU has no use for.",
        "shell=compute",
        f"nk=vortex_afu:1:{inst}",
        f"sp={inst}.m_axi_mem_0:{mem_tag}",
        f"sp={inst}.m_axi_host:{host_tag}",
        "",
        "# Kernel clock, in Hz rather than via the linker's --clock-hz flag, whose",
        "# name and help text disagree on the unit.",
        "[clock]",
        f"krnl={inst}",
        f"freqhz={kernel_mhz * 1000000}",
        "",
    ])


def emit_firesim_mk(c: Card) -> str:
    fs = c.firesim
    if not fs:
        raise DescriptorError(f"{c.path}: no [firesim] section; card '{c.name}' has no FireSim platform")
    lines = [f"# Generated by gen_platform.py from {os.path.relpath(c.path, PLATFORM_DIR)}. Do not edit.",
             f"FIRESIM_CARD := {c.name}",
             f"PLATFORM := {_require(fs, 'platform', c.path, 'firesim')}",
             f"FPGA_PART := {c.part}"]
    if "device" in fs:
        lines.append(f"DEVICE := {fs['device']}")
    if "host_mem_sp" in fs:
        lines.append(f"HOST_MEM_SP := {fs['host_mem_sp']}")
    if "platform_config" in fs:
        lines.append(f"PLATFORM_CONFIG := {fs['platform_config']}")
    return "\n".join(lines) + "\n"


def check_host_tag(c: Card, host_tag: str) -> None:
    if host_tag in c.host_refuse:
        raise DescriptorError(f"host tag '{host_tag}' is refused on card '{c.name}': {c.host_refuse_why}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", required=True, choices=FLOWS)
    ap.add_argument("--xsa", default="", help="XRT platform name (xrt flow)")
    ap.add_argument("--family", default="", help="FPGA family reported by platforminfo (xrt flow)")
    ap.add_argument("--device", default="", help="device part (aved flow)")
    ap.add_argument("--card", default="", help="select a card by name instead of matching")
    ap.add_argument("--emit", required=True, choices=("mk", "cfg", "firesim-mk", "card"))
    ap.add_argument("--mem-tag", default="")
    ap.add_argument("--host-tag", default="")
    ap.add_argument("--kernel-mhz", type=int, default=0)
    ap.add_argument("--platform-dir", default=PLATFORM_DIR)
    ap.add_argument("-o", "--output", default="-")
    a = ap.parse_args(argv)

    try:
        cards = load_all(a.platform_dir)
        c = resolve(cards, a.flow, a.xsa, a.family.strip(), a.device, a.card)
        if a.emit == "mk":
            out = emit_mk(c)
        elif a.emit == "cfg":
            out = emit_cfg(c, a.mem_tag or c.mem_tags[0], a.host_tag or c.host_tag,
                           a.kernel_mhz or c.kernel_mhz)
        elif a.emit == "firesim-mk":
            out = emit_firesim_mk(c)
        else:
            out = c.name + "\n"
    except DescriptorError as e:
        print(f"gen_platform: error: {e}", file=sys.stderr)
        return 1

    if a.output == "-":
        sys.stdout.write(out)
    else:
        # Rewrite only on change so dependents are not needlessly invalidated.
        old = None
        if os.path.exists(a.output):
            with open(a.output) as f:
                old = f.read()
        if old != out:
            os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
            with open(a.output, "w") as f:
                f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
