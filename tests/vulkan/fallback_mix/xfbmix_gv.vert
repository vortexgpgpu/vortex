/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 460
/* The vertex and its index, for a geometry shader to capture. */

layout(location = 0) flat out uint v_index;

void main()
{
   uint v = uint(gl_VertexIndex);
   gl_Position = vec4(-0.5 + 0.5 * float(v % 3u), (v % 3u) == 1u ? 0.5 : -0.5, 0.0, 1.0);
   v_index = v;
}
