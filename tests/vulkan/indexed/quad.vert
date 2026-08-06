/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Indexed-draw vertex shader: a quad's 4 corners, gl_VertexIndex-driven.
 * Rendered via vkCmdDrawIndexed with index buffer [0,1,2, 0,2,3] (two
 * triangles sharing the 0,2 diagonal). gl_VertexIndex MUST be the index-buffer
 * value, not the draw position — a driver that uses the raw thread id would
 * read past the 4 corners (indices 4,5) and produce garbage. */
#version 450

layout(location = 0) out vec3 v_color;

vec2 pos[4] = vec2[](
   vec2(-0.6, -0.6), vec2( 0.6, -0.6), vec2( 0.6,  0.6), vec2(-0.6,  0.6)
);
vec3 col[4] = vec3[](
   vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
   vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0)
);

void main()
{
   gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
   v_color = col[gl_VertexIndex];
}
