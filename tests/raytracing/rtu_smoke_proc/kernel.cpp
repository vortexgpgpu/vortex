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
// PRISM RTU procedural intersection-shader smoke kernel.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked IS dispatcher (mtvec target), integer-only (see common.h header
// note). Reads the object-space ray the RTU staged into
// VX_RT_OBJECT_RAY_* (feature 1), echoes origin.z / direction.z into
// hitAttribute slots, writes the IS hit distance + a sentinel
// hitAttribute (feature 2), and ACCEPTs.
//
// .insn funct7 = (slot<<2)|1 for GET, (slot<<2)|0 for SET:
//   get OBJECT_RAY_ORIGIN+2   (10) -> 41    get OBJECT_RAY_DIRECTION+2 (13) -> 53
//   set HIT_T                 (14) -> 56    set HIT_ATTR_0 (17) -> 68
//   set HIT_ATTR_1            (18) -> 72    set HIT_ATTR_2 (19) -> 76
__attribute__((naked, used))
static void rt_is_dispatcher(void) {
  __asm__ volatile (
    ".insn r 0x2b, 5, 41, t0, x0, x0\n"    // t0 = object_ray_origin.z   (slot 10)
    ".insn r 0x2b, 5, 53, t1, x0, x0\n"    // t1 = object_ray_direction.z(slot 13)
    "li   t2, %0\n"                        // t2 = IS hit_t (4.5f bits)
    ".insn r 0x2b, 5, 56, x0, t2, x0\n"    // set HIT_T      = t2
    "li   t2, %1\n"                        // t2 = attr magic
    ".insn r 0x2b, 5, 68, x0, t2, x0\n"    // set HIT_ATTR_0 = magic
    ".insn r 0x2b, 5, 72, x0, t0, x0\n"    // set HIT_ATTR_1 = object_ray_origin.z
    ".insn r 0x2b, 5, 76, x0, t1, x0\n"    // set HIT_ATTR_2 = object_ray_direction.z
    "li   t4, %2\n"                        // t4 = CB_ACCEPT
    ".insn r 0x2b, 6, 0, x0, t4, x0\n"     // vx_rt_cb_ret(ACCEPT)
    "mret\n"
    :: "i"(RTU_IS_HIT_T_BITS),
       "i"(RTU_IS_ATTR_MAGIC),
       "i"(VX_RT_CB_ACCEPT)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Register the IS dispatcher as the M-mode trap handler (mtvec).
  csr_write(0x305, (uintptr_t)&rt_is_dispatcher);

  vx_rt_set3(VX_RT_RAY_ORIGIN,
             vx_rt_f2u(arg->ray_origin[0]),
             vx_rt_f2u(arg->ray_origin[1]),
             vx_rt_f2u(arg->ray_origin[2]));
  vx_rt_set3(VX_RT_RAY_DIRECTION,
             vx_rt_f2u(arg->ray_direction[0]),
             vx_rt_f2u(arg->ray_direction[1]),
             vx_rt_f2u(arg->ray_direction[2]));
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);
  vx_rt_set1(VX_RT_RAY_FLAGS, 0u);   // procedural primitive → IS decides

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  // Read the committed (IS-supplied) hit distance + hitAttributes.
  uint32_t hit_t_bits = vx_rt_get_after(VX_RT_HIT_T,      sts);
  uint32_t attr0      = vx_rt_get_after(VX_RT_HIT_ATTR_0, sts);
  uint32_t attr1      = vx_rt_get_after(VX_RT_HIT_ATTR_1, sts);
  uint32_t attr2      = vx_rt_get_after(VX_RT_HIT_ATTR_2, sts);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  *(uint32_t*)&results[0].hit_t  = hit_t_bits;
  results[0].hit_attr0           = attr0;
  results[0].hit_attr1           = attr1;
  results[0].hit_attr2           = attr2;
  results[0].pad                 = 0;
}
