/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Opaque red with an alpha picked by screen quadrant, for alpha-to-coverage.
 * 4 * alpha is 0, 1.2, 2.4 and 4: clear of the half-way points between sample
 * counts, so each quadrant has one nearest count (0, 1, 2 and 4 samples). */
#version 450

layout(location = 0) out vec4 out_color;

/* Half the 16x16 target: the quadrant edge. */
const int QUADRANT = 8;

const float alphas[4] = float[](0.0, 0.3, 0.6, 1.0);

void main()
{
   ivec2 q = ivec2(gl_FragCoord.xy) / QUADRANT;
   out_color = vec4(1.0, 0.0, 0.0, alphas[q.y * 2 + q.x]);
}
