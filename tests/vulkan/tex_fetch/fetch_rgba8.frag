/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * texelFetch on a 4x4 RGBA8 texture: every fragment fetches the texel
 * its interpolated coordinate lands in and compares against the value
 * the host uploaded there, plus a textureSize check. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   ivec2 tc = ivec2(min(v_uv, vec2(0.999)) * 4.0);
   vec4  t  = texelFetch(tex, tc, 0);
   vec4  want = vec4(float(tc.x * 60 + 15) / 255.0,
                     float(tc.y * 60 + 15) / 255.0,
                     200.0 / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.01)));
   ok = ok && all(equal(textureSize(tex, 0), ivec2(4)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
