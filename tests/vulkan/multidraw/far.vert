/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multi-draw test, far triangle: covers the frame centre at z=0.75,
 * coloured red. Drawn after the near triangle; with a persisting depth
 * buffer the LESS test rejects it at the centre. */
#version 450

layout(location = 0) out vec3 v_color;

vec2 verts[3] = vec2[](
   vec2(-0.7, 0.7), vec2(0.7, 0.7), vec2(0.0, -0.7)
);

void main()
{
   gl_Position = vec4(verts[gl_VertexIndex], 0.75, 1.0);
   v_color = vec3(1.0, 0.0, 0.0);   /* red */
}
