/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450

layout(location = 0) flat in uint view;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(float(60u + 80u * view) / 255.0, 0.0, 0.0, 1.0);
}
