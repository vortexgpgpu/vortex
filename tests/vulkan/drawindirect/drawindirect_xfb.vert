/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * drawindirect.vert with gl_Position captured to transform-feedback buffer 0,
 * 16 bytes per vertex: the byte count a vkCmdDrawIndirectByteCountEXT draw
 * reads back. */
#version 450

layout(xfb_buffer = 0, xfb_stride = 16) out gl_PerVertex {
   layout(xfb_offset = 0) vec4 gl_Position;
};

vec2 positions[9] = vec2[](
   vec2(-0.9, -0.9), vec2(-0.1, -0.9), vec2(-0.5,  0.9),
   vec2( 0.1, -0.9), vec2( 0.9, -0.9), vec2( 0.5,  0.9),
   vec2(-0.08, -0.9), vec2( 0.08, -0.9), vec2( 0.0,  0.9)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
