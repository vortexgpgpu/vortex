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
# statically linked kernel image so the ones in use are x1..x(Rx-1),
# f0..f(Rf-1) and u1..u(Ru), and reports the image's register counts. A consistent renaming
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
FP_FIXED = 'fp_fixed'   # names FP registers implicitly: FP numbering kept, every FP register counted
FP_KEPT  = 'fp_kept'    # names a run of FP registers: FP numbering kept, the run counted
UNPACKED = 'unpacked'   # names integer registers implicitly: image left as is
FP_HI    = 'fp_hi'      # (FP_HI, n): the instruction names FP registers up to fn
WIN_HI   = 'win_hi'     # (WIN_HI, w): a vx_rwin sets a field's window up to w

REG_WINDOW = 32         # registers per register window


class DecodeError(Exception):
    pass


def decode(w):
    """(register fields as [(field, 'x'|'f'|'u')], image properties) of one instruction."""
    mode = w & 0x3
    if mode != 0x3:
        return decode_uniform(w | 0x3, mode)
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
            if f3 in (0, 1, 4, 5):                      # TMEM_LD/ST: rd starts a run of rs2 + 1 registers
                return [('rs1', X)], (FP_KEPT, (FP_HI, ((w >> 7) & 0x1F) + rs2))
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
        if f7 == 0x01:
            # vx_rwin: its fields are windows, not registers
            wins = [(w >> 7) & 7, (w >> 15) & 7, (w >> 20) & 7, (w >> 12) & 7]
            return [], ((WIN_HI, max(wins)),)
        raise DecodeError("unknown EXT3 instruction")
    if op == OP_EXT4:
        if f3 in (0, 3, 4):
            return [('rs1', X)], (FP_FIXED,)            # fragment / wide FP loads: rd starts a run
        if f3 in (2, 5, 6):
            return [('rs1', X)], (FP_FIXED,)            # fragment / wide FP stores: rs2 starts a run
        raise DecodeError("unknown EXT4 instruction")
    raise DecodeError("unknown opcode")


def decode_uniform(w, mode):
    """Fields of an XVortexU instruction (bits 1:0 = mode, w with them set to 11).
    Uniform registers are 'u'."""
    op = w & 0x7F
    f3 = (w >> 12) & 0x7
    f7 = (w >> 25) & 0x7F
    X, F, U = 'x', 'f', 'u'
    if mode == 0b10:
        if op == OP_FCI and f3 == 0 and f7 in (0x79, 0x71):
            return ([('rd', U), ('rs1', X)] if f7 == 0x79 else [('rd', X), ('rs1', U)]), ()
        # custom instructions whose integer operands are all warp-invariant:
        # tmc, wspawn, bar, arrive/wait; tcu_ld; TMEM; tensor jobs
        ext1 = op == OP_EXT1 and ((f7 == 0x00 and f3 in (0, 1, 4, 6)) or
                                  (f7 == 0x02 and f3 == 2) or f7 in (0x05, 0x06))
        if op in (OP_R, OP_R_W, OP_I, OP_I_W, OP_LUI, OP_AUIPC, OP_B, OP_JALR, OP_L, OP_S) \
           or (op == OP_SYS and f3 != 0) or ext1:
            fields, props = decode(w)
            return [(name, U if k == X else k) for name, k in fields], props
        raise DecodeError("reserved uniform encoding")
    if mode == 0b01:
        if op in (OP_R, OP_R_W):
            return [('rd', X), ('rs1', X), ('rs2', U)], ()
        if op in (OP_I, OP_I_W, OP_L):
            return [('rd', X), ('rs1', U)], ()
        if op == OP_FL:
            return [('rd', F), ('rs1', U)], ()
        if op in (OP_B, OP_S):
            return [('rs1', X), ('rs2', U)], ()
        raise DecodeError("reserved uniform-operand encoding")
    if op in (OP_R, OP_R_W):
        return [('rd', X), ('rs1', U), ('rs2', X)], ()
    if op in (OP_B, OP_S):
        return [('rs1', U), ('rs2', X)], ()
    if op == OP_FS:
        return [('rs1', U), ('rs2', F)], ()
    raise DecodeError("reserved uniform-first-operand encoding")


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


def is_zfinx(data):
    """Whether the image's RISC-V arch attribute names Zfinx: FP operands are x registers."""
    is64 = data[4] == 2
    if is64:
        e_shoff, = struct.unpack_from('<Q', data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x3A)
    else:
        e_shoff, = struct.unpack_from('<I', data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x2E)
    SHT_RISCV_ATTRIBUTES = 0x70000003
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if is64:
            sh_type, _, _, sh_offset, sh_size = struct.unpack_from('<IQQQQ', data, sh + 4)
        else:
            sh_type, _, _, sh_offset, sh_size = struct.unpack_from('<IIIII', data, sh + 4)
        if sh_type == SHT_RISCV_ATTRIBUTES and b'_zfinx' in data[sh_offset:sh_offset + sh_size]:
            return True
    return False


def zfinx_view(fields, props):
    """Under Zfinx an FP field names an x register; FP registers named implicitly
    (fragments, register runs) sit at x positions this script does not model."""
    fields = [(name, 'x' if k == 'f' else k) for name, k in fields]
    props = tuple(UNPACKED if (p in (FP_FIXED, FP_KEPT) or (isinstance(p, tuple) and p[0] == FP_HI))
                  else p for p in props)
    return fields, props


def scan(data):
    """Decoded instructions as [(file offset, word, fields)], and the image properties."""
    insts, props = [], set()
    zfinx = is_zfinx(data)
    for off, size, addr in code_ranges(data):
        for i in range(0, size - size % 4, 4):
            w, = struct.unpack_from('<I', data, off + i)
            if w == 0:
                continue                                 # alignment padding
            try:
                fields, p = decode(w)
            except DecodeError as e:
                raise DecodeError("0x{:x}: {:08x}: {}".format(addr + i, w, e))
            if zfinx:
                fields, p = zfinx_view(fields, p)
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


def counts_of(insts, props):
    """(Rx, Rf, Ru) of these instructions, or None when they must run with every
    register. Ru counts u1 up: u0 reads zero and takes no row."""
    if UNPACKED in props:
        return None
    xs = regs_used(insts, 'x') | {0}
    fs = regs_used(insts, 'f') | {p[1] for p in props if isinstance(p, tuple) and p[0] == FP_HI}
    us = regs_used(insts, 'u') | {0}
    rx = round_up(max(xs) + 1)
    rf = 32 if FP_FIXED in props else (round_up(max(fs) + 1) if fs else 0)
    ru = round_up(max(us))
    # A field in window w can name any register up to w * 32 + 31; the windows
    # each instruction runs under are not tracked, so count whole windows.
    wins = [p[1] for p in props if isinstance(p, tuple) and p[0] == WIN_HI]
    if wins:
        reach = (max(wins) + 1) * REG_WINDOW
        rx = max(rx, reach)
        rf = max(rf, reach) if rf else 0
    return rx, rf, ru


def counts(data):
    """(Rx, Rf, Ru) of an image, or None when it must run with every register."""
    return counts_of(*scan(data))


def elf_header(data):
    is64 = data[4] == 2
    entry, = struct.unpack_from('<Q' if is64 else '<I', data, 0x18)
    return is64, entry


def func_starts(data):
    """Start addresses of the image's FUNC symbols."""
    is64, _ = elf_header(data)
    if is64:
        e_shoff, = struct.unpack_from('<Q', data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x3A)
    else:
        e_shoff, = struct.unpack_from('<I', data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from('<HH', data, 0x2E)
    starts = set()
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if is64:
            sh_type, = struct.unpack_from('<I', data, sh + 4)
            sh_offset, sh_size = struct.unpack_from('<QQ', data, sh + 0x18)
            sh_entsize, = struct.unpack_from('<Q', data, sh + 0x38)
        else:
            sh_type, = struct.unpack_from('<I', data, sh + 4)
            sh_offset, sh_size = struct.unpack_from('<II', data, sh + 0x10)
            sh_entsize, = struct.unpack_from('<I', data, sh + 0x24)
        SHT_SYMTAB, STT_FUNC = 2, 2
        if sh_type != SHT_SYMTAB or not sh_entsize:
            continue
        for j in range(sh_size // sh_entsize):
            e = sh_offset + j * sh_entsize
            if is64:
                info, = struct.unpack_from('<B', data, e + 4)
                value, = struct.unpack_from('<Q', data, e + 8)
            else:
                value, = struct.unpack_from('<I', data, e + 4)
                info, = struct.unpack_from('<B', data, e + 12)
            if info & 0xF == STT_FUNC:
                starts.add(value)
    return starts


def sext(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def kernel_counts(data, pcs):
    """(Rx, Rf, Ru) of each kernel entry in `pcs`: the registers of the code a warp
    can run for that kernel, i.e. the startup code and what the kernel reaches by
    direct calls and jumps. A kernel that makes any other indirect call or jump
    gets the whole image's counts."""
    insts, props = scan(data)
    image = counts_of(insts, props)
    by_addr = {}
    regions = []
    for off, size, addr in code_ranges(data):
        regions.append((addr, addr + size - size % 4))
        base = off - addr
        for inst in insts:
            if off <= inst[0] < off + size:
                by_addr[inst[0] - base] = inst
    starts = sorted(set(a for lo, hi in regions for a in (lo,)) |
                    {a for a in func_starts(data) if any(lo <= a < hi for lo, hi in regions)})

    def extent(a):
        """[lo, hi) of the function holding address a: up to the next function start."""
        i = max(k for k, st in enumerate(starts) if st <= a)
        end = next(hi for lo, hi in regions if lo <= a < hi)
        nxt = starts[i + 1] if i + 1 < len(starts) and starts[i + 1] < end else end
        return starts[i], nxt

    def closure(entry, allow_indirect):
        seen, work, code, indirect = set(), [entry], [], False
        while work:
            lo, hi = extent(work.pop())
            if lo in seen:
                continue
            seen.add(lo)
            for a in range(lo, hi, 4):
                inst = by_addr.get(a)
                if inst is None:
                    continue
                code.append(inst)
                w = inst[1]
                op, mode = w & 0x7F, w & 0x3
                if mode != 0x3:
                    if (w | 0x3) & 0x7F == OP_JALR:
                        indirect = True
                    continue
                if op == OP_JAL:
                    imm = sext(((w >> 31) << 20) | (((w >> 12) & 0xFF) << 12) |
                               (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3FF) << 1), 21)
                    t = a + imm
                    if (w >> 7) & 0x1F or not lo <= t < hi:
                        work.append(t)
                elif op == OP_JALR:
                    rd, rs1, imm = (w >> 7) & 0x1F, (w >> 15) & 0x1F, sext(w >> 20, 12)
                    prev = by_addr.get(a - 4)
                    if rd == 0 and rs1 == 1 and imm == 0:
                        continue                         # return
                    if prev and prev[1] & 0x7F == OP_AUIPC and (prev[1] >> 7) & 0x1F == rs1:
                        work.append(a - 4 + sext(prev[1] & 0xFFFFF000, 32) + imm)
                    else:
                        indirect = True
        if indirect and not allow_indirect:
            return None
        return code

    _, entry = elf_header(data)
    startup = closure(entry, True) if any(lo <= entry < hi for lo, hi in regions) else []
    result = []
    for pc in pcs:
        code = closure(pc, False)
        if code is None:
            result.append(image)
            continue
        sub = {i[0]: i for i in startup + code}
        sub_props = set()
        zfinx = is_zfinx(data)
        for _, w, fields in sub.values():
            p = decode(w)[1]
            sub_props.update(zfinx_view(fields, p)[1] if zfinx else p)
        result.append(counts_of(list(sub.values()), sub_props))
    return result


def pack(data):
    """The image with its registers renamed, or unchanged when it cannot be packed."""
    insts, props = scan(data)
    if UNPACKED in props or any(isinstance(p, tuple) and p[0] == WIN_HI for p in props):
        return data
    maps = {
        'x': {r: i for i, r in enumerate(sorted(regs_used(insts, 'x') | {0}))},
        'f': {} if FP_FIXED in props or FP_KEPT in props
             else {r: i for i, r in enumerate(sorted(regs_used(insts, 'f')))},
        'u': {r: i for i, r in enumerate(sorted(regs_used(insts, 'u') | {0}))},
    }
    out = bytearray(data)
    for off, w, fields in insts:
        nw = w
        for name, k in fields:
            m = maps[k]
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
    print("vx_regpack: {}: {}".format(argv[1], "unpacked" if r is None else "Rx={} Rf={} Ru={}".format(*r)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
