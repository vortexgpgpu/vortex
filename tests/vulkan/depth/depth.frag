/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Depth-test fragment shader: passthrough of the interpolated colour. */
#version 450

layout(location = 0) in  vec3 v_color;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(v_color, 1.0);
}
