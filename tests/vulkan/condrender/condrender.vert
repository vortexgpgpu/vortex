/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A single oversized triangle covering the whole target, so a predicate that
 * was honoured and one that was ignored differ over the entire image rather
 * than at an edge. A push-constant offset moves it wholly outside the viewport,
 * which is how the occlusion query that feeds a predicate is made to see
 * nothing. */
#version 460

layout(push_constant) uniform PC {
   vec2 offset;
} pc;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p + pc.offset, 0.0, 1.0);
}
