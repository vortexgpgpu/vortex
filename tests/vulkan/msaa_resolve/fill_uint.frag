/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A distinct RGBA8UI value in every sample. An integer resolve takes sample 0
 * rather than averaging, so red and green, which differ per sample, say which
 * sample was taken. Keep in step with expect_uint() in main.c. */
#version 450

layout(location = 0) out uvec4 out_color;

void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   uint k = uint(gl_SampleID);
   out_color = uvec4(10u + k, 20u + 3u * k, p.x, p.y);
}
