/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Every fragment stores the provoking vertex's twenty values, in capture
 * order. */
#version 460

struct S { float p; vec3 q; };
layout(location = 1) flat in S s;
layout(location = 3, component = 0) flat in float a[2];
layout(location = 3, component = 1) flat in float b[2];
layout(location = 3, component = 2) flat in vec2 c[2];
layout(location = 5) flat in mat2 m;
layout(location = 7, component = 1) flat in vec3 e;
layout(location = 7, component = 0) flat in float f;

layout(std430, set = 0, binding = 0) buffer Out { float v[20]; } o;
layout(location = 0) out vec4 o_color;

void main()
{
   o.v[0] = s.p;
   o.v[1] = s.q.x;
   o.v[2] = s.q.y;
   o.v[3] = s.q.z;
   o.v[4] = a[0];
   o.v[5] = a[1];
   o.v[6] = b[0];
   o.v[7] = b[1];
   o.v[8] = c[0].x;
   o.v[9] = c[0].y;
   o.v[10] = c[1].x;
   o.v[11] = c[1].y;
   o.v[12] = m[0].x;
   o.v[13] = m[0].y;
   o.v[14] = m[1].x;
   o.v[15] = m[1].y;
   o.v[16] = e.x;
   o.v[17] = e.y;
   o.v[18] = e.z;
   o.v[19] = f;
   o_color = vec4(0.0, 1.0, 0.0, 1.0);
}
