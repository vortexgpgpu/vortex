#version 450
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle over the whole target at depth pc.depth; the scissor picks the
 * part a pass draws.
 */
layout(push_constant) uniform Params { uint pass; float depth; } pc;

void main()
{
   vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
   gl_Position = vec4(p * 2.0 - 1.0, pc.depth, 1.0);
}
