/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each triangle with adjacency becomes its six vertices as 2-pixel points, vertex
// k of primitive p in a colour naming (p, k): a later primitive sharing a
// vertex paints over it, so the image shows every primitive's vertices, in
// order, and in which order the primitives came.
layout(triangles_adjacency) in;
layout(points, max_vertices = 6) out;
layout(location = 0) flat out vec4 gcol;
void main()
{
   for (int k = 0; k < 6; k++) {
      uint c = (uint(gl_PrimitiveIDIn) * 6u + uint(k)) % 15u + 1u;
      gl_Position = gl_in[k].gl_Position;
      gl_PointSize = 2.0;
      gcol = vec4(float(c & 1u), float((c >> 1) & 1u), float((c >> 2) & 1u), float((c >> 3) & 1u));
      EmitVertex();
   }
}
