/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT */
#version 450

layout(location = 0) flat in vec3 v_flat;
layout(location = 1) in vec3 v_col;
layout(location = 0) out vec4 o_color;

void main()
{
   o_color = vec4(0.5 * v_flat + 0.5 * v_col, 1.0);
}
