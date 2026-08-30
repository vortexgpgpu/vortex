/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad carrying the texture coordinate. Both mip levels are flat,
 * so the coordinate cannot change the answer -- it varies only so a warp's
 * lanes carry different addresses rather than one shared address. */
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
