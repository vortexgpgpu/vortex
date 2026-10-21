/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT */
#version 460

layout(location = 0) in vec4 i_color;
layout(location = 0) out vec4 o_color;

void main()
{
   o_color = i_color;
}
