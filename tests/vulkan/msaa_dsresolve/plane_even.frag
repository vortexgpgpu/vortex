/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Keeps only the pixels whose x + y is even, so the stencil reference this draw
 * writes forms a per-pixel checkerboard. Keep in step with expect_stencil() in
 * main.c. */
#version 450

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   if (((p.x + p.y) & 1) != 0) {
      discard;
   }
}
