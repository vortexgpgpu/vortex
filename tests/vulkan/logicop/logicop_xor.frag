/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The XOR source operand. Every channel is 0 or 255 so the merge is exact
 * through the 8-bit colour buffer, and the destination is the cleared
 * colour, so the result equals neither operand. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.0, 1.0, 1.0);
}
