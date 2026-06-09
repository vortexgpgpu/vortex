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
//
// PRISM RTU SBT smoke kernel — Phase 7.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// IS shader 0 — writes MAGIC_0 + cb_ret(ACCEPT) + mret.
__attribute__((naked, used))
static void rt_is_shader_0(void) {
  __asm__ volatile (
    ".insn r 0x2b, 6, 103, t0, x0, x1\n"   // t0 = payload pointer (slot 25)
    "li t1, %0\n"                           // t1 = MAGIC_0
    "sw t1, 0(t0)\n"
    "li t2, %1\n"                           // CB_ACCEPT
    ".insn r 0x2b, 6, 0, x0, t2, x0\n"
    "mret\n"
    :: "i"(RTU_SBT_MAGIC_0), "i"(VX_RT_CB_ACCEPT)
  );
}

// IS shader 1 — writes MAGIC_1.
__attribute__((naked, used))
static void rt_is_shader_1(void) {
  __asm__ volatile (
    ".insn r 0x2b, 6, 103, t0, x0, x1\n"
    "li t1, %0\n"
    "sw t1, 0(t0)\n"
    "li t2, %1\n"
    ".insn r 0x2b, 6, 0, x0, t2, x0\n"
    "mret\n"
    :: "i"(RTU_SBT_MAGIC_1), "i"(VX_RT_CB_ACCEPT)
  );
}

// Two-level trap dispatcher: read cb_type + sbt_idx + sbt_base from
// the RTU regfile, compute offset = sbt_idx * 16 + (cb_type - 1) * 4,
// load the per-shader PC, tail-jump. The matched shader exits via
// cb_ret + mret.
//
// funct7 for vx_rt_get(slot) is (slot << 2) | 1:
//   VX_RT_CB_TYPE       (29) → 117
//   VX_RT_HIT_SBT_IDX   (31) → 125
//   VX_RT_SBT_BASE      (26) → 105
__attribute__((naked, used))
static void rt_dispatcher(void) {
  __asm__ volatile (
    ".insn r 0x2b, 6, 119, t0, x0, x1\n"   // t0 = cb_type
    ".insn r 0x2b, 6, 127, t1, x0, x1\n"   // t1 = sbt_idx
    ".insn r 0x2b, 6, 107, t2, x0, x1\n"   // t2 = sbt_base
    "slli t1, t1, 4\n"                      // t1 = sbt_idx * 16 (record stride)
    "addi t0, t0, -1\n"                     // t0 = cb_type - 1
    "slli t0, t0, 2\n"                      // t0 = (cb_type - 1) * 4
    "add  t1, t1, t0\n"                     // t1 = total byte offset
    "add  t2, t2, t1\n"                     // t2 = &sbt[sbt_idx][cb_type_off]
    "lw   t3, 0(t2)\n"                      // t3 = shader_pc
    "jr   t3\n"                             // tail-jump (no return)
    ::
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Build the SBT in memory: 2 records, each holds 4 PCs (one per
  // cb_type at offset (cb_type-1)*4). Only the IS slot (offset 4) is
  // populated; the other slots stay 0.
  uint32_t* sbt = (uint32_t*)(uintptr_t)arg->sbt_addr;
  for (uint32_t i = 0; i < 8; ++i) sbt[i] = 0;   // 2 records × 4 PCs
  sbt[0 * 4 + 1] = (uint32_t)(uintptr_t)&rt_is_shader_0;   // sbt[0].is
  sbt[1 * 4 + 1] = (uint32_t)(uintptr_t)&rt_is_shader_1;   // sbt[1].is

  // Register the lookup dispatcher in mtvec and publish the SBT base
  // (dispatcher-only slot the trap handler reads via vx_rt_get).
  csr_write(0x305, (uintptr_t)&rt_dispatcher);
  vx_rt_set(VX_RT_SBT_BASE,
             (uint32_t)(arg->sbt_addr & 0xffffffffu));

  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // The trace stages the payload pointer the IS shaders read via vx_rt_get.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = hit.t;
  // Read the SBT-shader-written payload only AFTER a wait-dependent op (the get
  // above) so in-order issue holds this load until the trace and its callback
  // store have retired.
  uint32_t sbt_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].sbt_payload       = sbt_payload;
  results[0].pad               = 0;
}
