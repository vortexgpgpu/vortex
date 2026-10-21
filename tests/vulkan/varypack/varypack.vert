/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Varyings that share locations by component, a struct whose vec3 member takes
 * a location of its own, and a matrix, each captured and read by the fragment
 * stage. Vertex v writes 100 * v + k to the k-th captured float (k = 1..20),
 * in capture order. The struct goes through a temporary array indexed at run
 * time, which keeps a struct in the shader's own memory. */
#version 460

struct S { float p; vec3 q; };
layout(location = 1, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 80) flat out S s;
layout(location = 3, component = 0, xfb_buffer = 0, xfb_offset = 16) flat out float a[2];
layout(location = 3, component = 1, xfb_buffer = 0, xfb_offset = 24) flat out float b[2];
layout(location = 3, component = 2, xfb_buffer = 0, xfb_offset = 32) flat out vec2 c[2];
layout(location = 5, xfb_buffer = 0, xfb_offset = 48) flat out mat2 m;
layout(location = 7, component = 1, xfb_buffer = 0, xfb_offset = 64) flat out vec3 e;
layout(location = 7, component = 0, xfb_buffer = 0, xfb_offset = 76) flat out float f;

void main()
{
   const float v = 100.0 * float(gl_VertexIndex);
   gl_Position = vec4(float(gl_VertexIndex & 1) * 4.0 - 1.0,
                      float(gl_VertexIndex >> 1) * 4.0 - 1.0, 0.0, 1.0);
   S tmp[2];
   for (int i = 0; i < 2; i++) {
      tmp[i].p = v + 1.0 + float(i) * 1000.0;
      tmp[i].q = vec3(v + 2.0, v + 3.0, v + 4.0);
   }
   s = tmp[gl_VertexIndex >> 3];
   a[0] = v + 5.0;
   a[1] = v + 6.0;
   b[0] = v + 7.0;
   b[1] = v + 8.0;
   c[0] = vec2(v + 9.0, v + 10.0);
   c[1] = vec2(v + 11.0, v + 12.0);
   m = mat2(v + 13.0, v + 14.0, v + 15.0, v + 16.0);
   e = vec3(v + 17.0, v + 18.0, v + 19.0);
   f = v + 20.0;
}
