/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * textureOffset on an array: layer 1 of a two-layer array whose texels carry
 * their own column, sampled at column 1 moved two texels right. A dropped offset
 * reads column 1, a dropped layer layer 0. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray tex;

void main()
{
   vec4 t = textureOffset(tex, vec3(1.5 / 4.0, 1.5 / 4.0, 1.0), ivec2(2, 0));
   /* texel (x, y) of layer L is (30 + 50x, 40y + 20L, 0) / 255; this is (3, 1). */
   vec4 want = vec4(float(30 + 50 * 3) / 255.0, float(40 * 1 + 20 * 1) / 255.0, 0.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
