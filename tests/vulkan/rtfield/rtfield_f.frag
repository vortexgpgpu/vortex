/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Four distinct channels, one of them negative so a signed format stores a
 * sign and an unsigned one clamps it. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, -0.6, 0.3, 0.8);
}
