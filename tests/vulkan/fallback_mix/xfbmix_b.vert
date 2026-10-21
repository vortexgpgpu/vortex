/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * xfbmix_a.vert that also counts its invocations in a storage buffer. A
 * layered draw whose vertex shader has a side effect cannot replay per layer,
 * so the device leaves it to llvmpipe. */
#version 460
#extension GL_ARB_shader_viewport_layer_array : require

layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uint o_v;
layout(location = 1, xfb_buffer = 0, xfb_offset = 4) flat out uint o_w;
layout(set = 0, binding = 0) buffer Count { uint n; } count;

void main()
{
   uint v = uint(gl_VertexIndex);
   gl_Position = vec4(-0.5 + 0.5 * float(v % 3u), (v % 3u) == 1u ? 0.5 : -0.5, 0.0, 1.0);
   gl_Layer = 0;
   o_v = v;
   o_w = v * 3u + 1u;
   atomicAdd(count.n, 1u);
}
