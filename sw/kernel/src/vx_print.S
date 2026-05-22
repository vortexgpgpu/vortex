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

#include <VX_types.h>
#include "common.h"

// void vx_putchar(int c)
//
// Console output — lossless back-pressured stream ring (proposal §10).
// Each hart owns one ring in the device-memory console buffer at
// VX_MEM_IO_COUT_ADDR. Layout:
//   uint32 wr  [VX_MEM_IO_COUT_SLOTS]      // producer (this kernel)
//   uint32 rd  [VX_MEM_IO_COUT_SLOTS]      // consumer (the host)
//   char   data[VX_MEM_IO_COUT_SLOTS][VX_MEM_IO_COUT_RING]
//
// The host drains the ring in its CP launch-wait poll and advances `rd`.
// If this hart's ring is full (wr - rd == RING) vx_putchar SPINS until
// the host frees space — it never overflows and never drops a character.
.type vx_putchar, @function
.global vx_putchar
vx_putchar:                                     // a0 = character
    csrr  t0, VX_CSR_MHARTID
    andi  t0, t0, %lo(VX_MEM_IO_COUT_SLOTS-1)   // t0 = ring slot
    li    t1, VX_MEM_IO_COUT_ADDR               // t1 = buffer base
    slli  t2, t0, 2
    add   t3, t2, t1                            // t3 = &wr[slot]
    li    t4, (VX_MEM_IO_COUT_SLOTS*4)
    add   t4, t4, t3                            // t4 = &rd[slot]
    lw    t5, 0(t3)                             // t5 = wr (this hart owns it)
    li    t6, VX_MEM_IO_COUT_RING
1:  lw    a1, 0(t4)                             // a1 = rd (reload each spin)
    sub   a2, t5, a1                            // a2 = occupancy
    bgeu  a2, t6, 1b                            // ring full -> back-pressure spin
    li    a1, (VX_MEM_IO_COUT_SLOTS*8)
    add   a1, a1, t1                            // a1 = data region base
    slli  a2, t0, VX_MEM_IO_COUT_RING_LOG2
    add   a1, a1, a2                            // + slot*RING
    addi  a2, t6, -1                            // RING-1 (ring mask)
    and   a2, t5, a2                            // wr & (RING-1)
    add   a1, a1, a2                            // &data[slot][wr mod RING]
    sb    a0, 0(a1)                             // store the character
    fence ow, ow                                // release barrier: the char
                                                // store must be globally
                                                // visible before the host
                                                // can observe the advanced
                                                // wr pointer, else it drains
                                                // not-yet-written ring bytes
    addi  t5, t5, 1
    sw    t5, 0(t3)                             // publish wr+1
    ret
