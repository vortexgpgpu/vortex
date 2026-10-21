/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The triangle index, one byte per channel. A quarter step above each code keeps
 * the UNORM conversion exact whether the device rounds or truncates. */
#version 450

layout(location = 0) flat in uint v_tri;

layout(location = 0) out vec4 out_color;

void main()
{
   const uvec3 b = uvec3(v_tri, v_tri >> 8, v_tri >> 16) & 0xffu;
   out_color = vec4((vec3(b) + 0.25) / 255.0, 1.0);
}
