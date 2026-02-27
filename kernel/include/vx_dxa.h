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

#ifndef __VX_DXA_H__
#define __VX_DXA_H__

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VX_DXA_EXT_OPCODE RISCV_CUSTOM0
#define VX_DXA_FUNCT7 0x3

// Dimension-specific architected opcodes (funct3 field).
#define VX_DXA_OP_ISSUE_1D 0
#define VX_DXA_OP_ISSUE_2D 1
#define VX_DXA_OP_ISSUE_3D 2
#define VX_DXA_OP_ISSUE_4D 3
#define VX_DXA_OP_ISSUE_5D 4

// Packed launch metadata:
// meta[3:0]   = descriptor slot (up to 16 descriptors)
// meta[30:4]  = raw barrier id payload
// meta[31]    = packed marker

// R type: .insn r opcode7, funct3, funct7, rd, rs1, rs2
// +--------+-----+-----+--------+----+---------+
// | funct7 | rs2 | rs1 | funct3 | rd | opcode7 |
// +--------+-----+-----+--------+----+---------+
// 31       25    20    15       12   7         0


inline uint32_t vx_dxa_pack_meta(uint32_t desc_slot, uint32_t barrier_id) {
  return ((desc_slot & 0x0fu) | ((barrier_id & 0x07ffffffu) << 4) | (1u << 31));
}

// 1D issue: coord0 only, funct3=0.
inline void vx_dxa_issue_1d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                uint32_t smem_addr,
                                uint32_t coord0) {
  register uint32_t c0 __asm__("t3") = coord0;
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  __asm__ volatile (
      ".insn r %0, %1, %2, x0, %3, %4\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE_1D), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta),
        "r"(c0)
      : "memory");
}

// 2D issue: coord0, coord1, funct3=1.
inline void vx_dxa_issue_2d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                uint32_t smem_addr,
                                uint32_t coord0,
                                uint32_t coord1) {
  register uint32_t c0 __asm__("t3") = coord0;
  register uint32_t c1 __asm__("t4") = coord1;
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  __asm__ volatile (
      ".insn r %0, %1, %2, x0, %3, %4\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE_2D), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta),
        "r"(c0), "r"(c1)
      : "memory");
}

// 3D issue: coord0..coord2, funct3=2.
inline void vx_dxa_issue_3d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                uint32_t smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2) {
  register uint32_t c0 __asm__("t3") = coord0;
  register uint32_t c1 __asm__("t4") = coord1;
  register uint32_t c2 __asm__("t5") = coord2;
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  __asm__ volatile (
      ".insn r %0, %1, %2, x0, %3, %4\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE_3D), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta),
        "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}

// 4D issue: coord0..coord3, funct3=3.
inline void vx_dxa_issue_4d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                uint32_t smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3) {
  register uint32_t c0 __asm__("t3") = coord0;
  register uint32_t c1 __asm__("t4") = coord1;
  register uint32_t c2 __asm__("t5") = coord2;
  register uint32_t c3 __asm__("t6") = coord3;
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  __asm__ volatile (
      ".insn r %0, %1, %2, x0, %3, %4\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE_4D), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}

// 5D issue: coord0..coord4, funct3=4.
inline void vx_dxa_issue_5d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                uint32_t smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3,
                                uint32_t coord4) {
  register uint32_t c0 __asm__("t3") = coord0;
  register uint32_t c1 __asm__("t4") = coord1;
  register uint32_t c2 __asm__("t5") = coord2;
  register uint32_t c3 __asm__("t6") = coord3;
  register uint32_t c4 __asm__("t0") = coord4;
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  __asm__ volatile (
      ".insn r %0, %1, %2, x0, %3, %4\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE_5D), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4)
      : "memory");
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DXA_H__
