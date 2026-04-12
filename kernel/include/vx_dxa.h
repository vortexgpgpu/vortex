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
#define VX_DXA_FUNCT7     0x3

// Packed launch metadata:
// meta[3:0]   = descriptor slot (up to 16 descriptors)
// meta[30:4]  = raw barrier id payload
// meta[31]    = packed marker

// R type: .insn r opcode7, funct3, funct7, rd, rs1, rs2
// +--------+-----+-----+--------+----+---------+
// | funct7 | rs2 | rs1 | funct3 | rd | opcode7 |
// +--------+-----+-----+--------+----+---------+
// 31       25    20    15       12   7         0

// Wgather-based DXA issue.  vx_wgather distributes args into per-lane register
// slots before the DXA instruction reads them across all 4 lanes:
//   Lane 0: rs1 = smem_addr,  rs2 = coord2
//   Lane 1: rs1 = meta,       rs2 = coord3
//   Lane 2: rs1 = coord0,     rs2 = coord4
//   Lane 3: rs1 = coord1,     rs2 = 0
//
// 1D and 2D: all rs2 lanes are zero, so rs2 = x0 (no second vx_wgather).
// 3D–5D: rs2 carries coord2..coord4, requiring a second vx_wgather.

inline uint32_t vx_dxa_pack_meta(uint32_t desc_slot, uint32_t barrier_id) {
  return (barrier_id << 4) | desc_slot;
}

// 1D: rs1 = wgather(smem_addr, meta, coord0, 0), rs2 = x0
inline void vx_dxa_issue_1d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// 2D: rs1 = wgather(smem_addr, meta, coord0, coord1), rs2 = x0
inline void vx_dxa_issue_2d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// 3D–5D: rs2 = wgather(coord2, coord3, coord4, 0)
inline void vx_dxa_issue_3d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)0u,
                                            (size_t)0u,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_issue_4d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)0u,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_issue_5d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3,
                                uint32_t coord4) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)coord4,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// Multicast DXA issues read GMEM once and replay SMEM writes to multiple
// CTAs within the same SM core.

// 1D multicast: rs2 = wgather(0, 0, 0, cta_mask)
inline void vx_dxa_issue_1d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)0u);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)0,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 2D multicast: rs2 = wgather(0, 0, 0, cta_mask)
inline void vx_dxa_issue_2d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)0,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 3D multicast: rs2 = wgather(coord2, 0, 0, cta_mask)
inline void vx_dxa_issue_3d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 4D multicast: rs2 = wgather(coord2, coord3, 0, cta_mask)
inline void vx_dxa_issue_4d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t coord3,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 5D multicast: rs2 = wgather(coord2, coord3, coord4, cta_mask)
inline void vx_dxa_issue_5d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t coord3,
                                          uint32_t coord4,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)coord4,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DXA_H__
