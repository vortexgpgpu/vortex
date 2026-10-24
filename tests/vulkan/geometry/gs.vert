/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Pass-through vertex stage: the geometry stage does the work.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 col;
layout(location = 2) in uint n;
layout(location = 0) out vec4 vcol;
layout(location = 1) flat out uint vn;
void main()
{
   gl_Position = vec4(pos, 0.0, 1.0);
   gl_PointSize = 1.0;
   vcol = col;
   vn = n;
}
