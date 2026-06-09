/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Depth-test vertex shader: two triangles that both cover the centre
 * of the frame, at different depths -- triangle 0 is near (z=0.25,
 * blue), triangle 1 far (z=0.75, red). With the depth test enabled
 * the near (blue) triangle wins the centre; without it the red one
 * (drawn last) would. */
#version 450

layout(location = 0) out vec3 v_color;

vec3 verts[6] = vec3[](
   /* triangle 0 -- near, blue */
   vec3(-0.7, -0.7, 0.25), vec3( 0.7, -0.7, 0.25), vec3( 0.0,  0.7, 0.25),
   /* triangle 1 -- far, red */
   vec3(-0.7,  0.7, 0.75), vec3( 0.7,  0.7, 0.75), vec3( 0.0, -0.7, 0.75)
);

vec3 colors[6] = vec3[](
   vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0),
   vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)
);

void main()
{
   gl_Position = vec4(verts[gl_VertexIndex].xy, verts[gl_VertexIndex].z, 1.0);
   v_color = colors[gl_VertexIndex];
}
