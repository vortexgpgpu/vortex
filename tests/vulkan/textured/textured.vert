/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Textured-quad vertex shader. No vertex buffer: a full-screen quad
 * (two triangles) is indexed by gl_VertexIndex, each corner carrying
 * a texture coordinate. The vertex stage runs on the Vortex device;
 * the texcoord is the lone generic varying (Phase 6). */
#version 450

layout(location = 0) out vec2 v_uv;

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

vec2 uvs[6] = vec2[](
   vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
   vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   v_uv = uvs[gl_VertexIndex];
}
