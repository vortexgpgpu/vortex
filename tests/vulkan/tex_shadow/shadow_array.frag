/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sampler2DArrayShadow: layer 1 holds depths well above layer 0's, so a
 * reference between the two layers' depths gives opposite results per
 * layer and a dropped layer index is visible. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArrayShadow tex;

void main()
{
   /* layer 0 depth = 0.125, layer 1 depth = 0.875 at texel x = 0. LESS. */
   float l0 = texture(tex, vec4(0.125, 0.5, 0.0, 0.5));   /* 0.5 > 0.125 -> 0 */
   float l1 = texture(tex, vec4(0.125, 0.5, 1.0, 0.5));   /* 0.5 < 0.875 -> 1 */
   bool ok = l0 < 0.5 && l1 > 0.5;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
