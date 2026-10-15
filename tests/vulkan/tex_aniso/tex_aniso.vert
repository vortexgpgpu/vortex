/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad whose texture coordinates run over u in [0, 2] and v in
 * [0, 0.5]: on a 32x32 target and a 64x64 texture, each pixel steps 4 texels
 * along u and 1 along v -- a footprint four times longer than it is wide. */
#version 450

layout(location = 0) out vec2 v_uv;

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   vec2 p = positions[gl_VertexIndex];
   gl_Position = vec4(p, 0.0, 1.0);
   v_uv = vec2((p.x + 1.0), (p.y + 1.0) * 0.25);
}
