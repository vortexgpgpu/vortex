/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad. The varying is not a texture coordinate -- the fragment
 * stage samples at constants -- it is only a per-pixel value the fragment
 * stage jitters its coordinates by, so the sampled addresses differ across a
 * warp's lanes instead of every lane fetching one address. */
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
