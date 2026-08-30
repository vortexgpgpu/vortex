/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * texelFetch on a 2x2 RGBA32F texture whose texels carry values outside
 * [0,1]: verifies the float carrier returns them exactly instead of
 * collapsing them through an 8-bit working space. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   ivec2 tc = ivec2(min(v_uv, vec2(0.999)) * 2.0);
   vec4  t  = texelFetch(tex, tc, 0);
   float s  = float(tc.x + 2 * tc.y + 1);
   vec4  want = vec4(2.5, -1.0, 0.25, 1.0) * s;
   bool ok = all(lessThan(abs(t - want), vec4(0.001)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
