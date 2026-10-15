/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Reports three LINEAR samples of a 4x4 texture under a min or max reduction;
 * the host holds the expected values. No jitter: probe b relies on a u weight of
 * exactly zero. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   /* texel space (x, y) = (u, v) * 4 - 0.5 */
   float a = texture(tex, vec2(0.45, 0.775)).r;    /* (1.3, 2.6) */
   float b = texture(tex, vec2(0.625, 0.25)).r;    /* (2.0, 0.5) */
   float c = texture(tex, vec2(0.05, 0.4375)).r;   /* (-0.3, 1.25) */
   out_color = vec4(a, b, c, 1.0);
}
