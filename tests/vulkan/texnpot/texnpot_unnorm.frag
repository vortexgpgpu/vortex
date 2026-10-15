/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three probes of the 5x3 texture in texel coordinates, through a sampler with
 * unnormalizedCoordinates (explicit LOD 0, as such a sampler requires). */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   /* A jitter inside a fifth of a texel gives the lanes different addresses. */
   float j = (v_uv.x - 0.5) * 0.2;
   float col = textureLod(tex, vec2(4.5 + j, 0.5), 0.0).r;
   float row = textureLod(tex, vec2(2.5 + j, 2.5), 0.0).r;
   float clp = textureLod(tex, vec2(6.0 + j, 1.5), 0.0).r;
   out_color = vec4(col, row, clp, 1.0);
}
