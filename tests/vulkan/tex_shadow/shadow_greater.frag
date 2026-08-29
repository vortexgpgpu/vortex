/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The same depth texture sampled through a GREATER comparison sampler:
 * every reference that passed under LESS must now fail, which catches a
 * driver that ignores the compare op rather than the compare itself. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DShadow tex;

void main()
{
   float uv = 0.125;                 /* texel x = 0, depth 0.125 */
   float uv3 = 0.875;                /* texel x = 3, depth 0.5   */
   float fail_lo = texture(tex, vec3(uv,  0.5, 0.05));   /* 0.05 < 0.125 -> 0 */
   float pass_lo = texture(tex, vec3(uv,  0.5, 0.30));   /* 0.30 > 0.125 -> 1 */
   float fail_hi = texture(tex, vec3(uv3, 0.5, 0.30));   /* 0.30 < 0.5   -> 0 */
   float pass_hi = texture(tex, vec3(uv3, 0.5, 0.90));   /* 0.90 > 0.5   -> 1 */
   bool ok = pass_lo > 0.5 && fail_lo < 0.5 &&
             pass_hi > 0.5 && fail_hi < 0.5;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
