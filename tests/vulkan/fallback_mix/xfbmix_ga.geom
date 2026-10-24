/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 460
/* xfbmix_a.vert's capture made by a geometry shader: each triangle passed on
 * into layer 0, every vertex capturing its index. */

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) flat in uint v_index[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uint o_v;
layout(location = 1, xfb_buffer = 0, xfb_offset = 4) flat out uint o_w;

void main()
{
   for (int k = 0; k < 3; k++) {
      gl_Position = gl_in[k].gl_Position;
      gl_Layer = 0;
      o_v = v_index[k];
      o_w = v_index[k] * 3u + 1u;
      EmitVertex();
   }
}
