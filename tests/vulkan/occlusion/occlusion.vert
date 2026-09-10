/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A single oversized triangle covering the whole target, so the draw's
 * footprint is exact and the sample count it should produce is simply every
 * pixel -- no fill rule to reason about between the draw and the query. */
#version 460

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
}
