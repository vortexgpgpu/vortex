/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Signed outputs: red is below the 8-bit range and blue is negative within it,
 * so the texels show saturation and two's-complement storage. */
#version 450

layout(location = 0) out ivec4 out_color;

void main()
{
   out_color = ivec4(-300, 100, -3, 1);
}
