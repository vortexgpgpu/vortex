/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * texelFetch on layer 1 of a two-layer 4x4 RGBA8 array texture, plus a
 * textureSize check whose third component must report the layer count. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray tex;

void main()
{
   ivec2 tc = ivec2(min(v_uv, vec2(0.999)) * 4.0);
   vec4  t  = texelFetch(tex, ivec3(tc, 1), 0);
   vec4  want = vec4(0.0,
                     float(tc.x * 60 + 15) / 255.0,
                     float(tc.y * 60 + 15) / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.01)));
   ok = ok && all(equal(textureSize(tex, 0), ivec3(4, 4, 2)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
