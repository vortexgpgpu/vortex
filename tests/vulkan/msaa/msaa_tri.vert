/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle, no edge axis-aligned and none at 45 degrees, so each edge
 * crosses pixels at a mix of coverage fractions rather than settling on a
 * single one. Its interior and its exterior are both large enough to check
 * separately from the partially covered band between them. */
#version 450

vec2 positions[3] = vec2[](
   vec2(-0.80, -0.90), vec2( 0.90, -0.45), vec2(-0.55,  0.85)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
