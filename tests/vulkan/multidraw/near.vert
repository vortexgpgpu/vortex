/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multi-draw test, near triangle: covers the frame centre at z=0.25,
 * coloured blue. Self-contained (gl_VertexIndex only) so the draw needs
 * no vertex buffer and no firstVertex offset. */
#version 450

layout(location = 0) out vec3 v_color;

vec2 verts[3] = vec2[](
   vec2(-0.7, -0.7), vec2(0.7, -0.7), vec2(0.0, 0.7)
);

void main()
{
   gl_Position = vec4(verts[gl_VertexIndex], 0.25, 1.0);
   v_color = vec3(0.0, 0.0, 1.0);   /* blue */
}
