/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Every captured value is exact in fp32, so the host can rebuild the capture
 * bit for bit from (gl_VertexIndex, gl_InstanceIndex). Buffer 0 leaves holes
 * at bytes 0-3, 12-15, 20-23, 28-31, 48-51, 60-63 and 80-95 of each 96-byte
 * vertex. Everything after o_color is captured and read by no fragment input:
 * a matrix, an array, a struct, and two floats sharing a location by
 * component. */
#version 460

layout(location = 0) out vec4 o_color;
layout(location = 1, xfb_buffer = 0, xfb_offset = 4, xfb_stride = 96) out vec2 o_a;
layout(location = 2, xfb_buffer = 0, xfb_offset = 16) flat out uint o_id;
layout(location = 3, xfb_buffer = 0, xfb_offset = 24) out float o_only;
layout(location = 4, xfb_buffer = 0, xfb_offset = 32) out mat2 o_m;
layout(location = 6, xfb_buffer = 0, xfb_offset = 52) out float o_arr[2];
struct S { float p; float q; };
layout(location = 8, xfb_buffer = 0, xfb_offset = 64) out S o_s;
layout(location = 10, component = 1, xfb_buffer = 0, xfb_offset = 72) out float o_c1;
layout(location = 10, component = 3, xfb_buffer = 0, xfb_offset = 76) out float o_c3;

layout(xfb_buffer = 1, xfb_stride = 24) out gl_PerVertex {
   layout(xfb_offset = 0) vec4 gl_Position;
   layout(xfb_offset = 16) float gl_PointSize;
   layout(xfb_offset = 20) float gl_ClipDistance[1];
};

void main()
{
   uint v = uint(gl_VertexIndex);
   uint i = uint(gl_InstanceIndex);
   gl_Position = vec4(-0.75 + 0.0625 * float(v % 16u) + 0.03125 * float(i),
                      -0.75 + 0.25 * float(v / 16u), 0.5, 1.0);
   gl_PointSize = 1.0 + float(v % 4u);
   gl_ClipDistance[0] = 2.0 + float(v);
   o_color = vec4(1.0, 0.0, 0.0, 1.0);
   o_a = vec2(float(v), float(i) + 0.25);
   o_id = v * 16u + i;
   o_only = float(v) + 0.5;
   o_m = mat2(float(v), float(v) + 1.0, float(i), -float(v));
   o_arr[0] = 2.0 * float(v);
   o_arr[1] = 3.0 * float(v) + float(i);
   o_s.p = 5.0 * float(v);
   o_s.q = 5.0 * float(v) + 1.0;
   o_c1 = 7.0 * float(i);
   o_c3 = float(v) - float(i);
}
