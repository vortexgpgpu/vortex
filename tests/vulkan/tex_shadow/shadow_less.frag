/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sampler2DShadow with compare LESS on a 4x4 D32F texture whose depth
 * rises with x. Each fragment compares a reference that straddles the
 * texel's depth, so the 0/1 result differs across the quad and a sampler
 * that ignored the comparison could not produce it. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DShadow tex;

void main()
{
   /* Texel depth is (x + 1) / 8: 0.125, 0.25, 0.375, 0.5 across x. */
   float uv = 0.125;                 /* texel x = 0, depth 0.125 */
   float uv3 = 0.875;                /* texel x = 3, depth 0.5   */
   /* ref < depth passes under LESS. */
   float pass_lo = texture(tex, vec3(uv,  0.5, 0.05));   /* 0.05 < 0.125 -> 1 */
   float fail_lo = texture(tex, vec3(uv,  0.5, 0.30));   /* 0.30 > 0.125 -> 0 */
   float pass_hi = texture(tex, vec3(uv3, 0.5, 0.30));   /* 0.30 < 0.5   -> 1 */
   float fail_hi = texture(tex, vec3(uv3, 0.5, 0.90));   /* 0.90 > 0.5   -> 0 */
   bool ok = pass_lo > 0.5 && fail_lo < 0.5 &&
             pass_hi > 0.5 && fail_hi < 0.5;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
