/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Points as they came, drawn 6 pixels wide, red and blue swapped.
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec4 vcol[];
layout(location = 0) flat out vec4 gcol;
void main()
{
   gl_Position = gl_in[0].gl_Position;
   gl_PointSize = 6.0;
   gcol = vcol[0].bgra;
   EmitVertex();
}
