/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 460
/* xfbmix_ga.geom, also counting its invocations in a storage buffer: a layered
 * draw with a side effect cannot replay per layer, so it runs on llvmpipe. */

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) flat in uint v_index[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uint o_v;
layout(location = 1, xfb_buffer = 0, xfb_offset = 4) flat out uint o_w;
layout(set = 0, binding = 0) buffer Count { uint n; } count;

void main()
{
   for (int k = 0; k < 3; k++) {
      gl_Position = gl_in[k].gl_Position;
      gl_Layer = 0;
      o_v = v_index[k];
      o_w = v_index[k] * 3u + 1u;
      EmitVertex();
   }
   atomicAdd(count.n, 1u);
}
