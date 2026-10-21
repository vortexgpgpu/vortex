/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The stippled strips' vertices, by index (main.c draws them indexed):
 *   0 -> 1 -> 2   the carry row: segment 0 to 1, then 1 to 2 after the zeros
 *   3 -> 4        the restart draw's first run
 *   5 -> 6        its second run
 * Rows sit on pixel centres; ends on pixel edges. */
#version 450

const float W = 64.0, H = 64.0;
const float X0 = 4.0, A = 5.0, B = 40.0;

void main()
{
   const int i = gl_VertexIndex;
   vec2 p;
   if (i == 0)      p = vec2(X0,         10.5);
   else if (i == 1) p = vec2(X0 + A,     10.5);
   else if (i == 2) p = vec2(X0 + A + B, 10.5);
   else if (i == 3) p = vec2(X0,         20.5);
   else if (i == 4) p = vec2(X0 + A,     20.5);
   else if (i == 5) p = vec2(X0,         30.5);
   else             p = vec2(X0 + B,     30.5);
   gl_Position = vec4(2.0 * p.x / W - 1.0, 2.0 * p.y / H - 1.0, 0.5, 1.0);
}
