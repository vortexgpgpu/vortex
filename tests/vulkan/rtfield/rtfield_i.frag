/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Signed outputs, red below the 8-bit range so it saturates. */
#version 450

layout(location = 0) out ivec4 out_color;

void main()
{
   out_color = ivec4(-300, 100, -3, 1);
}
