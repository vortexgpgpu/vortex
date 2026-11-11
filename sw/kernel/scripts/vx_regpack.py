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

# Register packing for dynamic warp allocation: renames the registers of a
# statically linked kernel image so the ones in use are x1..x(Rx-1) and
# f0..f(Rf-1), and reports the image's register counts. A consistent renaming
# of a closed image keeps its meaning, calling convention included. The field
# rules follow hw/rtl/core/VX_decode.sv.

import struct
import sys

REG_GRANULE = 4

OP_L, OP_FL, OP_EXT1, OP_FENCE, OP_I, OP_AUIPC, OP_I_W = 0x03, 0x07, 0x0B, 0x0F, 0x13, 0x17, 0x1B
OP_S, OP_FS, OP_EXT2, OP_AMO, OP_R, OP_LUI, OP_R_W = 0x23, 0x27, 0x2B, 0x2F, 0x33, 0x37, 0x3B
OP_FMADD, OP_FMSUB, OP_FNMSUB, OP_FNMADD, OP_FCI = 0x43, 0x47, 0x4B, 0x4F, 0x53
OP_EXT3, OP_B, OP_JALR, OP_JAL, OP_SYS, OP_EXT4 = 0x5B, 0x63, 0x67, 0x6F, 0x73, 0x7B

FIELD_SHIFT = {'rd': 7, 'rs1': 15, 'rs2': 20, 'rs3': 27}

# Image properties a decoded instruction can set.
FP_FIXED = 'fp_fixed'   # names FP registers implicitly or as a run: FP numbering kept
UNPACKED = 'unpacked'   # names integer registers implicitly: image left as is


class DecodeError(Exception):
    pass


def decode(w):
    """(register fields as [(field, 'x'|'f')], image properties) of one instruction."""
    if (w & 0x3) != 0x3:
        raise DecodeError("compressed instruction")
    op = w & 0x7F
    f3 = (w >> 12) & 0x7
    f7 = (w >> 25) & 0x7F
    f5 = f7 >> 2
    rs2 = (w >> 20) & 0x1F
    X, F = 'x', 'f'
    if op in (OP_LUI, OP_AUIPC, OP_JAL):
        return [('rd', X)], ()
    if op in (OP_JALR, OP_I, OP_I_W, OP_L):
        return [('rd', X), ('rs1', X)], ()
    if op in (OP_R, OP_R_W):
        return [('rd', X), ('rs1', X), ('rs2', X)], ()
    if op in (OP_B, OP_S):
        return [('rs1', X), ('rs2', X)], ()
    if op == OP_FENCE:
        return [], ()
    if op == OP_AMO:
        return [('rd', X), ('rs1', X), ('rs2', X)], ()
    if op == OP_SYS:
        if f3 & 0x3:
            return ([('rd', X)] + ([] if f3 & 0x4 else [('rs1', X)])), ()
        return [], ()
    if op == OP_FL:
        return [('rd', F), ('rs1', X)], ()
    if op == OP_FS:
        return [('rs1', X), ('rs2', F)], ()
    if op in (OP_FMADD, OP_FMSUB, OP_FNMSUB, OP_FNMADD):
        return [('rd', F), ('rs1', F), ('rs2', F), ('rs3', F)], ()
    if op == OP_FCI:
        if f5 in (0x00, 0x01, 0x02, 0x03, 0x04, 0x05):
            return [('rd', F), ('rs1', F), ('rs2', F)], ()
        if f5 in (0x08, 0x0B):
            return [('rd', F), ('rs1', F)], ()
        if f5 == 0x14:
            return [('rd', X), ('rs1', F), ('rs2', F)], ()
        if f5 in (0x18, 0x1C):
            return [('rd', X), ('rs1', F)], ()
        if f5 in (0x1A, 0x1E):
            return [('rd', F), ('rs1', X)], ()
        raise DecodeError("unknown FP operation")
    if op == OP_EXT1:
        if f7 == 0x00:
            return {0: [('rs1', X)],
                    1: [('rs1', X), ('rs2', X)],
                    2: [('rd', X), ('rs1', X)],
                    3: [('rs1', X)],
                    4: [('rs1', X), ('rs2', X)],
                    5: [('rs1', X), ('rs2', X)],
                    6: [('rd', X), ('rs1', X), ('rs2', X)],
                    7: []}[f3], ()
        if f7 == 0x01:
            return [('rd', X), ('rs1', X), ('rs2', X)], ()
        if f7 == 0x02:
            if f3 == 2:
                return [('rs1', X)], ()                 # TCU_LD: rs2/rd fields are format and slot
            if f3 & 1:
                return [], (UNPACKED,)                  # WGMMA: descriptors in x10/x11
            return [], (FP_FIXED,)                      # WMMA: fragment registers fixed
        if f7 == 0x03:
            return [('rs1', X), ('rs2', X)], ()         # DXA issue
        if f7 == 0x04 and f3 in (1, 2):
            return [('rd', F), ('rs1', X), ('rs2', X)], ()
        if f7 == 0x05:
            if f3 in (0, 4):
                return [('rs1', X)], (FP_FIXED,)        # TMEM_LD: rd starts a run, rs2 field is a count
            if f3 in (1, 5):
                return [('rs1', X)], (FP_FIXED,)        # TMEM_ST: rd starts a run, rs2 field is a format
            if f3 == 6:
                return [('rd', X), ('rs1', X)], ()
            if f3 == 7:
                return [('rs1', X), ('rs2', X)], ()
        if f7 == 0x06:
            if f3 in (0, 1):
                return [('rs1', X), ('rs2', X)], ()     # TC_IDESC's rd field is a slot
            if f3 == 2 or f3 >= 4:
                return [('rd', X), ('rs1', X), ('rs2', X)], ()
        raise DecodeError("unknown EXT1 instruction")
    if op == OP_EXT2:
        if f3 in (0, 5):
            return [('rd', X), ('rs1', X), ('rs2', X), ('rs3', X)], ()
        if f3 == 3:
            return [('rs1', X), ('rs2', X), ('rs3', X)], ()
        if f3 in (6, 7):
            return [], (UNPACKED,)                      # RTU
        raise DecodeError("unknown EXT2 instruction")
    if op == OP_EXT3:
        if f7 == 0x00 and f3 == 0:
            return [('rd', F), ('rs1', F)], ()          # MUFU: rs2 field is the function
        raise DecodeError("unknown EXT3 instruction")
    if op == OP_EXT4:
        if f3 in (0, 3, 4):
            return [('rs1', X)], (FP_FIXED,)            # fragment / wide FP loads: rd starts a run
        if f3 in (2, 5, 6):
            return [('rs1', X)], (FP_FIXED,)            # fragment / wide FP stores: rs2 starts a run
        raise DecodeError("unknown EXT4 instruction")
    raise DecodeError("unknown opcode")


def code_ranges(data):
    """(file offset, size, address) of each executable section of an ELF image."""
    if data[:4] != b'\x7fELF':
        raise DecodeError("not an ELF file")
    is64 = data[4] == 2
    if is64:
        e_shoff, = struct.unpack_from('<Q', data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x3A)
    else:
        e_shoff, = struct.unpack_from('<I', data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x2E)
    ranges = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if is64:
            sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from('<IQQQQ', data, sh + 4)
        else:
            sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from('<IIIII', data, sh + 4)
        SHT_PROGBITS, SHF_EXECINSTR = 1, 0x4
        if sh_type == SHT_PROGBITS and (sh_flags & SHF_EXECINSTR) and sh_size:
            ranges.append((sh_offset, sh_size, sh_addr))
    return ranges


def scan(data):
    """Decoded instructions as [(file offset, word, fields)], and the image properties."""
    insts, props = [], set()
    for off, size, addr in code_ranges(data):
        for i in range(0, size - size % 4, 4):
            w, = struct.unpack_from('<I', data, off + i)
            if w == 0:
                continue                                 # alignment padding
            try:
                fields, p = decode(w)
            except DecodeError as e:
                raise DecodeError("0x{:x}: {:08x}: {}".format(addr + i, w, e))
            insts.append((off + i, w, fields))
            props.update(p)
    return insts, props


def regs_used(insts, kind):
    used = set()
    for _, w, fields in insts:
        for name, k in fields:
            if k == kind:
                used.add((w >> FIELD_SHIFT[name]) & 0x1F)
    return used


def round_up(n):
    return (n + REG_GRANULE - 1) // REG_GRANULE * REG_GRANULE


def counts(data):
    """(Rx, Rf) of an image, or None when it must run with every register."""
    insts, props = scan(data)
    if UNPACKED in props:
        return None
    xs = regs_used(insts, 'x') | {0}
    fs = regs_used(insts, 'f')
    rx = round_up(max(xs) + 1)
    rf = 32 if FP_FIXED in props else (round_up(max(fs) + 1) if fs else 0)
    return rx, rf


def pack(data):
    """The image with its registers renamed, or unchanged when it cannot be packed."""
    insts, props = scan(data)
    if UNPACKED in props:
        return data
    xmap = {r: i for i, r in enumerate(sorted(regs_used(insts, 'x') | {0}))}
    fmap = {} if FP_FIXED in props else {r: i for i, r in enumerate(sorted(regs_used(insts, 'f')))}
    out = bytearray(data)
    for off, w, fields in insts:
        nw = w
        for name, k in fields:
            m = xmap if k == 'x' else fmap
            sh = FIELD_SHIFT[name]
            r = (w >> sh) & 0x1F
            nw = (nw & ~(0x1F << sh)) | (m.get(r, r) << sh)
        if nw != w:
            struct.pack_into('<I', out, off, nw)
    return bytes(out)


def main(argv):
    if len(argv) != 2:
        print("usage: vx_regpack.py <kernel.elf>")
        return 1
    with open(argv[1], 'rb') as f:
        data = f.read()
    try:
        packed = pack(data)
        r = counts(packed)
    except DecodeError as e:
        print("vx_regpack: {}: {}".format(argv[1], e))
        return 1
    if packed != data:
        with open(argv[1], 'wb') as f:
            f.write(packed)
    print("vx_regpack: {}: {}".format(argv[1], "unpacked" if r is None else "Rx={} Rf={}".format(*r)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
