/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * xfbmix_a.vert that also reads a vertex attribute, bound to no buffer
 * (nullDescriptor): it reads zeros, which leave the capture unchanged, and the
 * device, which fetches attributes from bound buffers only, leaves the draw to
 * llvmpipe. */
#version 460
#extension GL_ARB_shader_viewport_layer_array : require

layout(location = 0) in vec4 in_zero;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uint o_v;
layout(location = 1, xfb_buffer = 0, xfb_offset = 4) flat out uint o_w;

void main()
{
   uint v = uint(gl_VertexIndex);
   gl_Position = vec4(-0.5 + 0.5 * float(v % 3u), (v % 3u) == 1u ? 0.5 : -0.5, 0.0, 1.0);
   gl_Layer = 0;
   o_v = v + uint(in_zero.x);
   o_w = v * 3u + 1u;
}
