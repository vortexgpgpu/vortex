#version 450
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Pixel (px, py) samples texel (8 px + 3, (129 py) / 31) of level 0, layer 1,
 * at its centre: its colour, with alpha replaced by whether it is resident.
 */
#extension GL_ARB_sparse_texture2 : require

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2DArray tex_n;

layout(push_constant) uniform Params { uvec4 e; uint level; uint count; } pc;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   vec2 texel = vec2(float(8 * p.x + 3), float((129 * p.y) / 31)) + 0.5;
   vec4 t;
   int code = sparseTextureLodARB(tex_n, vec3(texel / vec2(pc.e.xy), 1.0), 0.0, t);
   out_color = vec4(t.rgb, sparseTexelsResidentARB(code) ? 1.0 : 0.0);
}
