/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Hello-triangle vertex shader. No vertex buffer: the three corner
 * positions and colors are indexed by gl_VertexIndex, so the whole
 * vertex stage is self-contained -- exactly what vortexpipe runs on
 * the Vortex device (Phase 3). */
#version 450

layout(location = 0) out vec3 v_color;

vec2 positions[3] = vec2[](
   vec2( 0.0, -0.5),
   vec2( 0.5,  0.5),
   vec2(-0.5,  0.5)
);

vec3 colors[3] = vec3[](
   vec3(1.0, 0.0, 0.0),
   vec3(0.0, 1.0, 0.0),
   vec3(0.0, 0.0, 1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   v_color = colors[gl_VertexIndex];
}
