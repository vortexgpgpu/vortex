/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * texelFetch on a 2x2 RGBA8_SINT texture through an isampler2D:
 * verifies the integer carrier returns the signed texel values exactly. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform isampler2D tex;

void main()
{
   ivec2 tc = ivec2(min(v_uv, vec2(0.999)) * 2.0);
   ivec4 t  = texelFetch(tex, tc, 0);
   ivec4 want = ivec4(tc.x * 10 - 5, tc.y * 20 - 7, 42, -1);
   bool ok = all(equal(t, want));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
