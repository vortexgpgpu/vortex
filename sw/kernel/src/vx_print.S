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
// Console output — LOSSY per-hart stream ring (proposal §10, post-O-1).
// Each hart owns one ring in the device-memory console buffer at
// VX_MEM_IO_COUT_ADDR. Layout:
//   uint32 wr   [VX_MEM_IO_COUT_SLOTS]     // producer (this kernel)
//   uint32 rd   [VX_MEM_IO_COUT_SLOTS]     // consumer (the host)
//   char   data [VX_MEM_IO_COUT_SLOTS][VX_MEM_IO_COUT_RING]
//   uint32 lost [VX_MEM_IO_COUT_SLOTS]     // overflow counter (this hart's
//                                          // bytes dropped because ring full)
//
// On a full ring (wr - rd == RING) the byte is DROPPED and lost[slot] is
// atomically incremented; vx_putchar NEVER blocks. The host drains the ring
// in its CP launch-wait poll (and post-launch), and surfaces lost[] as
// "[#N: lost K bytes]" alongside the drained text. Matches CUDA / HIP
// printf semantics — bounded buffer, lossy, with an overflow indicator;
// fixes the producer/consumer deadlock that bit the legacy spin-on-full
// path when a printf-heavy launch's COUT could only be drained post-retire.
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
    lw    a1, 0(t4)                             // a1 = rd (single snapshot)
    sub   a2, t5, a1                            // a2 = occupancy
    bgeu  a2, t6, .Lvx_putchar_drop             // ring full -> drop + bump lost
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
.Lvx_putchar_drop:
    // Ring full: drop the byte, atomically bump lost[slot] so the host
    // can surface "[#N: lost K bytes]" on the next drain. Atomic because
    // multiple harts can share a slot when total_harts > SLOTS.
    li    a1, (VX_MEM_IO_COUT_SLOTS*8 + VX_MEM_IO_COUT_SLOTS*VX_MEM_IO_COUT_RING)
    add   a1, a1, t1                            // a1 = &lost[0]
    add   a1, a1, t2                            // + slot*4 = &lost[slot]
    li    a2, 1
    amoadd.w zero, a2, (a1)                     // lost[slot] += 1
    ret
