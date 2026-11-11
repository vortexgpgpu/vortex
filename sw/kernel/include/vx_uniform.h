// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __VX_UNIFORM_H__
#define __VX_UNIFORM_H__

// Uniform registers (VX_CFG_EXT_U_ENABLE): a per-warp file u0-u31 of 64-bit
// registers, named here by number (ud, us1, us2: 1-31, 0 reads zero). An
// instruction's bits [1:0] select the file of its standard encoding:
//   0b10: every register is uniform; integer operations have RV64 semantics
//         at either XLEN, loads and stores are made once per warp.
//   0b01: a per-thread instruction whose second source (the only source of an
//         immediate form, a store's data) is uniform, broadcast to every lane.
// The compiler does not see the uniform registers, so these are volatile and
// keep their order.

#define __VX_U_STR(x) #x
#define __VX_U(n) "x" __VX_U_STR(n)

// Opcodes of the uniform (0b10) and uniform-operand (0b01) forms.
#define VX_U_OP       0x32
#define VX_U_OP_IMM   0x12
#define VX_U_OP_32    0x3a
#define VX_U_OP_IMM32 0x1a
#define VX_U_LUI      0x36
#define VX_U_LOAD     0x02
#define VX_U_STORE    0x22
#define VX_U_SYSTEM   0x72
#define VX_U_OP_FP    0x52
#define VX_UO_OP      0x31
#define VX_UO_OP_IMM  0x11
#define VX_UO_STORE   0x21
#define VX_UO_AMO     0x2d

// funct7/funct3 of the integer operations
#define VX_U_ADD    0x00, 0
#define VX_U_SUB    0x20, 0
#define VX_U_SLL    0x00, 1
#define VX_U_SLT    0x00, 2
#define VX_U_SLTU   0x00, 3
#define VX_U_XOR    0x00, 4
#define VX_U_SRL    0x00, 5
#define VX_U_SRA    0x20, 5
#define VX_U_OR     0x00, 6
#define VX_U_AND    0x00, 7
#define VX_U_MUL    0x01, 0
#define VX_U_MULH   0x01, 1
#define VX_U_MULHU  0x01, 3
#define VX_U_DIV    0x01, 4
#define VX_U_DIVU   0x01, 5
#define VX_U_REM    0x01, 6
#define VX_U_REMU   0x01, 7

// load/store widths (funct3)
#define VX_U_B  0
#define VX_U_H  1
#define VX_U_W  2
#define VX_U_D  3
#define VX_U_BU 4
#define VX_U_HU 5
#define VX_U_WU 6

#define __VX_U_R(opc, f7, f3, rd, rs1, rs2) \
  __asm__ volatile (".insn r %0, %1, %2, " rd ", " rs1 ", " rs2 \
                    :: "i"(opc), "i"(f3), "i"(f7))

// ud = us1 op us2 (64-bit), and the 32-bit W forms
#define vx_u_op(op, ud, us1, us2)  __VX_U_R(VX_U_OP, op, __VX_U(ud), __VX_U(us1), __VX_U(us2))
#define vx_u_opw(op, ud, us1, us2) __VX_U_R(VX_U_OP_32, op, __VX_U(ud), __VX_U(us1), __VX_U(us2))

// ud = us1 op imm, imm a 12-bit signed constant (funct3: 0 addi, 2 slti,
// 3 sltiu, 4 xori, 6 ori, 7 andi)
#define vx_u_opi(funct3, ud, us1, imm) \
  __asm__ volatile (".insn i %0, %1, " __VX_U(ud) ", " __VX_U(us1) ", %2" \
                    :: "i"(VX_U_OP_IMM), "i"(funct3), "i"(imm))
#define vx_u_li(ud, imm)          vx_u_opi(0, ud, 0, imm)
#define vx_u_slli(ud, us1, shamt) vx_u_opi(1, ud, us1, (shamt))
#define vx_u_srli(ud, us1, shamt) vx_u_opi(5, ud, us1, (shamt))
#define vx_u_srai(ud, us1, shamt) vx_u_opi(5, ud, us1, 0x400 | (shamt))

// ud = sext(imm20 << 12)
#define vx_u_lui(ud, imm20) \
  __asm__ volatile (".insn u %0, " __VX_U(ud) ", %1" :: "i"(VX_U_LUI), "i"(imm20))

// ud = mem[us1 + off], mem[us1 + off] = us2: once per warp
#define vx_u_load(width, ud, us1, off) \
  __asm__ volatile (".insn i %0, %1, " __VX_U(ud) ", %2(" __VX_U(us1) ")" \
                    :: "i"(VX_U_LOAD), "i"(width), "i"(off) : "memory")
#define vx_u_store(width, us2, us1, off) \
  __asm__ volatile (".insn s %0, %1, " __VX_U(us2) ", %2(" __VX_U(us1) ")" \
                    :: "i"(VX_U_STORE), "i"(width), "i"(off) : "memory")

// ud = csr (the first active lane's value)
#define vx_u_csrr(ud, csr) \
  __asm__ volatile (".insn i %0, 2, " __VX_U(ud) ", x0, %1" \
                    :: "i"(VX_U_SYSTEM), "i"(((csr) & 0x800) ? (csr) - 0x1000 : (csr)))

// ud = x of the first active lane, zero-extended
#define vx_u_mvx(ud, x) \
  __asm__ volatile (".insn r %0, 0, 0x79, " __VX_U(ud) ", %1, x0" \
                    :: "i"(VX_U_OP_FP), "r"(x))

// x = us, every lane
#define vx_x_mvu(us) ({ \
  size_t __x; \
  __asm__ volatile (".insn r %1, 0, 0x71, %0, " __VX_U(us) ", x0" \
                    : "=r"(__x) : "i"(VX_U_OP_FP)); \
  __x; })

// per thread: xs1 op us2
#define __VX_XU_R(f7, f3, xs1, us2) ({ \
  size_t __x; \
  __asm__ volatile (".insn r %1, %2, %3, %0, %4, " __VX_U(us2) \
                    : "=r"(__x) : "i"(VX_UO_OP), "i"(f3), "i"(f7), "r"(xs1)); \
  __x; })
#define vx_xu_op(op, xs1, us2) __VX_XU_R(op, xs1, us2)

// per thread: us1 op imm
#define vx_xu_opi(funct3, us1, imm) ({ \
  size_t __x; \
  __asm__ volatile (".insn i %1, %2, %0, " __VX_U(us1) ", %3" \
                    : "=r"(__x) : "i"(VX_UO_OP_IMM), "i"(funct3), "i"(imm)); \
  __x; })

// per thread: mem[xs1 + off] = us2
#define vx_xu_store(width, us2, xs1, off) \
  __asm__ volatile (".insn s %0, %1, " __VX_U(us2) ", %2(%3)" \
                    :: "i"(VX_UO_STORE), "i"(width), "i"(off), "r"(xs1) : "memory")

// per thread, global indexed: mem[us1 + zext(xidx)] (64-bit base)
#define vx_u_gload(width, us1, xidx) ({ \
  size_t __x; \
  __asm__ volatile (".insn r %1, %2, 0x78, %0, " __VX_U(us1) ", %3" \
                    : "=r"(__x) : "i"(VX_UO_AMO), "i"(width), "r"(xidx) : "memory"); \
  __x; })
#define vx_u_gstore(width, us1, xidx, xdata) \
  __asm__ volatile (".insn r %0, %1, 0x7c, %2, " __VX_U(us1) ", %3" \
                    :: "i"(VX_UO_AMO), "i"(width), "r"(xdata), "r"(xidx) : "memory")

#endif // __VX_UNIFORM_H__
