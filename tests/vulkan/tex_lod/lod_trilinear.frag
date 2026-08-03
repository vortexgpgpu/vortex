/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Mip-linear sampling at a fractional LOD: the result must be the blend of
 * levels 0 and 1, which is neither level's own colour. A sampler that
 * dropped the fraction and picked one level fails on both bounds. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   vec4 t = textureLod(tex, v_uv, 0.5);
   /* level 0 = (240,20,20), level 1 = (20,240,20); the half-way blend is
    * (130,130,20) with the red and green channels crossing over. */
   vec4 want = vec4(130.0 / 255.0, 130.0 / 255.0, 20.0 / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.06)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
