/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// From each input triangle (origin, origin + one cell across, origin + one cell
// down): a six-vertex strip of three cells -- its odd triangles wound back the
// way the first runs -- then a two-vertex strip that never completes, then a
// separate triangle, then a lone vertex. Every vertex carries its emission
// index + 1 as its colour, one bit per channel, so each triangle shows its
// provoking vertex.
layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;
layout(location = 0) in vec4 vcol[];
layout(location = 0) flat out vec4 gcol;
vec4 tint(int k)
{
   int c = k + 1;
   return vec4(float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1), float((c >> 3) & 1));
}
void emit(vec4 p, int k)
{
   gl_Position = p;
   gcol = tint(k);
   EmitVertex();
}
void main()
{
   vec4 o = gl_in[0].gl_Position;
   vec4 dx = gl_in[1].gl_Position - o;
   vec4 dy = gl_in[2].gl_Position - o;
   int k = 0;
   for (int c = 0; c < 3; c++) {
      emit(o + float(c) * dx, k++);
      emit(o + float(c) * dx + dy, k++);
   }
   EndPrimitive();
   emit(o + 4.0 * dx, k++);
   emit(o + 4.0 * dx + dy, k++);
   EndPrimitive();
   emit(o + 2.0 * dy, k++);
   emit(o + 3.0 * dy, k++);
   emit(o + 2.0 * dy + dx, k++);
   EndPrimitive();
   emit(o + 5.0 * dx, k++);
}
