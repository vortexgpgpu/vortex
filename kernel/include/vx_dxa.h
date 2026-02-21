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

#define VX_DXA_OP_SETUP0 0
#define VX_DXA_OP_SETUP1 1
#define VX_DXA_OP_COORD01 2
#define VX_DXA_OP_COORD23 3
#define VX_DXA_OP_ISSUE 4
#define VX_DXA_OP_LAUNCH 5

// Single-op launch path requires full-stack support (decode/execute/sfu/rtl).
// Keep disabled by default until both simx and rtlsim paths are aligned.
#ifndef VX_DXA_ENABLE_SINGLE_OP
#define VX_DXA_ENABLE_SINGLE_OP 0
#endif

// setup0: rs1=descriptor slot id, rs2=barrier id
inline void vx_dxa_setup0(uint32_t desc_slot, uint32_t barrier_id) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_SETUP0), "i"(VX_DXA_FUNCT7),
                      "r"(desc_slot), "r"(barrier_id)
                    : "memory");
#else
  (void)desc_slot;
  (void)barrier_id;
#endif
}

// setup1: rs1=shared memory address, rs2=flags/direction
inline void vx_dxa_setup1(uint32_t smem_addr, uint32_t flags) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_SETUP1), "i"(VX_DXA_FUNCT7),
                      "r"(smem_addr), "r"(flags)
                    : "memory");
#else
  (void)smem_addr;
  (void)flags;
#endif
}

// coords01: rs1=coord0, rs2=coord1
inline void vx_dxa_coords01(uint32_t coord0, uint32_t coord1) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_COORD01), "i"(VX_DXA_FUNCT7),
                      "r"(coord0), "r"(coord1)
                    : "memory");
#else
  (void)coord0;
  (void)coord1;
#endif
}

// coords23: rs1=coord2, rs2=coord3
inline void vx_dxa_coords23(uint32_t coord2, uint32_t coord3) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_COORD23), "i"(VX_DXA_FUNCT7),
                      "r"(coord2), "r"(coord3)
                    : "memory");
#else
  (void)coord2;
  (void)coord3;
#endif
}

// issue: rs1=coord4, rs2=reserved
inline void vx_dxa_issue(uint32_t coord4) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, x0"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_ISSUE), "i"(VX_DXA_FUNCT7),
                      "r"(coord4)
                    : "memory");
#else
  (void)coord4;
#endif
}

// launch: rs1=shared memory address, rs2=packed meta(desc/bar/flags)
inline void vx_dxa_launch(uint32_t smem_addr, uint32_t meta) {
#ifdef EXT_DXA_ENABLE
  __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"
                    :
                    : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_LAUNCH), "i"(VX_DXA_FUNCT7),
                      "r"(smem_addr), "r"(meta)
                    : "memory");
#else
  (void)smem_addr;
  (void)meta;
#endif
}

// meta[7:0]=desc_slot, meta[15:8]=flags, meta[30:16]=barrier_id, meta[31]=packed marker
inline uint32_t vx_dxa_pack_meta(uint32_t desc_slot, uint32_t barrier_id, uint32_t flags) {
  return ((desc_slot & 0xffu) | ((flags & 0xffu) << 8) | ((barrier_id & 0x7fffu) << 16) | (1u << 31));
}

inline void vx_dxa_set_coord_bank_5d(uint32_t coord0,
                                     uint32_t coord1,
                                     uint32_t coord2,
                                     uint32_t coord3,
                                     uint32_t coord4) {
#if defined(EXT_DXA_ENABLE) && defined(EXT_F_ENABLE)
  __asm__ volatile (
      "fmv.w.x ft5, %0\n\t"
      "fmv.w.x ft6, %1\n\t"
      "fmv.w.x ft7, %2\n\t"
      "fmv.w.x ft8, %3\n\t"
      "fmv.w.x ft9, %4\n\t"
      :
      : "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3), "r"(coord4)
      : "memory", "ft5", "ft6", "ft7", "ft8", "ft9");
#else
  (void)coord0;
  (void)coord1;
  (void)coord2;
  (void)coord3;
  (void)coord4;
#endif
}

// Leader-thread helper:
// - EXT_F path: one architected launch instruction + micro-uops for coords.
// - fallback path: legacy 5-instruction packet.
inline void vx_dxa_issue_5d_leader(
    uint32_t desc_slot,
    uint32_t barrier_id,
    uint32_t smem_addr,
    uint32_t flags,
    uint32_t coord0,
    uint32_t coord1,
    uint32_t coord2,
    uint32_t coord3,
    uint32_t coord4) {
#if defined(EXT_DXA_ENABLE) && defined(EXT_F_ENABLE) && VX_DXA_ENABLE_SINGLE_OP
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id, flags);
  __asm__ volatile (
      "fmv.w.x ft5, %0\n\t"
      "fmv.w.x ft6, %1\n\t"
      "fmv.w.x ft7, %2\n\t"
      "fmv.w.x ft8, %3\n\t"
      "fmv.w.x ft9, %4\n\t"
      ".insn r %5, %6, %7, x0, %8, %9\n\t"
      :
      : "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3), "r"(coord4),
        "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_OP_LAUNCH), "i"(VX_DXA_FUNCT7),
        "r"(smem_addr), "r"(meta)
      : "memory", "ft5", "ft6", "ft7", "ft8", "ft9");
#else
  vx_dxa_setup0(desc_slot, barrier_id);
  vx_dxa_setup1(smem_addr, flags);
  vx_dxa_coords01(coord0, coord1);
  vx_dxa_coords23(coord2, coord3);
  vx_dxa_issue(coord4);
#endif
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DXA_H__
