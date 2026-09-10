/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Reads sample 0 of the multisample attachment directly, rather than a resolve
 * of it. texelFetch on a sampler2DMS is the only way an application can do
 * this -- a multisample image cannot be the source of a copy -- so it is what
 * the driver's refusal has to be observed through.
 *
 * Sample 0 rather than an average: averaging would hide a plane in which only
 * some samples were written, which is one of the states this exists to catch. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DMS ms_tex;

void main()
{
   out_color = texelFetch(ms_tex, ivec2(gl_FragCoord.xy), 0);
}
