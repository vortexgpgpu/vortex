/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Unsigned outputs, blue past every 8- and 10-bit range so it saturates. */
#version 450

layout(location = 0) out uvec4 out_color;

void main()
{
   out_color = uvec4(200u, 3u, 70000u, 1u);
}
