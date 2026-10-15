/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Linear values below 1/255, where the sRGB encode is steepest: the encoded
 * bytes depend on the linear value at a precision an 8-bit linear channel does
 * not hold. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.002, 0.0005, 0.001, 1.0);
}
