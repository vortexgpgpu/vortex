/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Triangle-strip vertex shader: a quad's 4 corners in STRIP order
 * (BL, BR, TL, TR), gl_VertexIndex-driven. Drawn as a non-indexed 4-vertex
 * VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP -> triangles {0,1,2} and {1,2,3}, which
 * together fill the [-0.6,0.6]^2 quad. The driver expands the strip into a
 * triangle-LIST index array (W8) and renders it on the list-native front end;
 * a correct expansion reproduces the full quad (same coverage as the indexed
 * triangle-list quad). */
#version 450

layout(location = 0) out vec3 v_color;

vec2 pos[4] = vec2[](
   vec2(-0.6, -0.6), vec2( 0.6, -0.6), vec2(-0.6,  0.6), vec2( 0.6,  0.6)
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
