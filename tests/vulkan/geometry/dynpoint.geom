/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input point becomes vn points, 2 pixels wide, along a row.
layout(points) in;
layout(points, max_vertices = 8) out;
layout(location = 0) in vec4 vcol[];
layout(location = 1) flat in uint vn[];
layout(location = 0) flat out vec4 gcol;
void main()
{
   for (uint k = 0u; k < vn[0]; k++) {
      gl_Position = gl_in[0].gl_Position + vec4(4.0 * float(k) / 32.0, 0.0, 0.0, 0.0);
      gl_PointSize = 2.0;
      gcol = vcol[0];
      EmitVertex();
   }
}
