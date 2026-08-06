/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * UBO/push-constant descriptor test — vertex stage. Self-contained
 * gl_VertexIndex triangle (no vertex buffers), same geometry as the
 * hello-triangle so the centre pixel is covered; the colour it emits
 * is unused by the fragment shader (which reads a UBO instead). */
#version 450

layout(location = 0) out vec3 v_color;

vec2 positions[3] = vec2[](
   vec2( 0.0, -0.5),
   vec2( 0.5,  0.5),
   vec2(-0.5,  0.5)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   v_color = vec3(1.0, 1.0, 1.0);
}
