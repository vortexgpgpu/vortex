/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Four vec4 varyings -- sixteen scalars, against a front end that carries
 * twelve interpolation planes. The last varying is the one that does not fit,
 * so the fragment shader reads it and nothing else: a driver that packs what
 * fits and drops the rest leaves that read holding a fixed-function default
 * rather than a shader value.
 *
 * Every vertex carries the same values, so the expected result is exact
 * regardless of where the pixel is sampled and does not depend on the
 * interpolator being accurate -- only on the varying arriving at all. */
#version 450

layout(location = 0) out vec4 v0;
layout(location = 1) out vec4 v1;
layout(location = 2) out vec4 v2;
layout(location = 3) out vec4 v3;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  0.9 : -0.9,
                 (gl_VertexIndex == 2) ?  0.9 : -0.9);
   gl_Position = vec4(p, 0.0, 1.0);

   v0 = vec4(0.125, 0.125, 0.125, 1.0);
   v1 = vec4(0.250, 0.250, 0.250, 1.0);
   v2 = vec4(0.375, 0.375, 0.375, 1.0);
   /* Planes 12..15 -- past the budget, and the only thing the FS reads. */
   v3 = vec4(0.500, 0.750, 0.250, 1.0);
}
