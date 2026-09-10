/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Writes gl_FragDepth to a value that contradicts the interpolated plane. A
 * depth test decided from the plane rather than from this value keeps the wrong
 * fragment, which is the whole point of the case: early-Z must not be armed for
 * a shader that supplies its own depth. */
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
   gl_FragDepth = pc.zwrite;
}
