/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multi-set UBO descriptor test — fragment stage (issue I5). Reads one
 * uniform buffer from descriptor SET 0 and another from descriptor
 * SET 1, and combines both into the output colour:
 *
 *   out = vec4(u0.color.rgb + u1.color.rgb, 1.0)
 *
 * The set-1 UBO lives in the fragment constant buffer at index 2
 * (lavapipe binds descriptor set N at index N+1). The device dereferences
 * its lp_jit_buffer.ptr, so that pointer MUST be relocated host->device
 * for set 1 too — not just set 0. A set-0-only relocation leaves u1's
 * pointer a host address and the on-device read faults / returns garbage,
 * which the host check detects (the output would not equal u0+u1). */
#version 450

layout(set = 0, binding = 0) uniform U0 { vec4 color; } u0;
layout(set = 1, binding = 0) uniform U1 { vec4 color; } u1;

layout(location = 0) in  vec3 v_color;   /* unused: isolates the UBO path */
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(u0.color.rgb + u1.color.rgb, 1.0);
}
