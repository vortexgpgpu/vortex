/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// The vertex as given; its uint attribute carries a per-point count.
layout(location = 0) in vec2 pos;
layout(location = 1) in uint n;
layout(location = 0) flat out uint vn;
void main()
{
   gl_Position = vec4(pos, 0.0, 1.0);
   vn = n;
}
