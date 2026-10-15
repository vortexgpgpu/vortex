/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each fragment fetches the texel under it, so the target holds the texture's
 * decode texel for texel. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   out_color = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}
