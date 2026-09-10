/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Writes no depth, so the interpolated plane is the fragment's depth and the
 * driver is free to arm early-Z for this shader. */
#version 450

layout(push_constant) uniform PC {
   float z;
   float zwrite;
   vec4  color;
} pc;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = pc.color;
}
