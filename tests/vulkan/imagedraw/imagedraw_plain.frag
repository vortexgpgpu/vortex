/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Colour only, no image write. Paired with imagedraw_vswrite.vert so that the
 * only texels in the storage image afterwards are the vertex stage's, and a
 * fragment-stage write cannot be mistaken for one. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   o_color = vec4(0.0, 1.0, 0.0, 1.0);
}
