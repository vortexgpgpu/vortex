/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Red at quarter alpha. With alpha-to-one the stored alpha must be 1.0, and
 * 0.25 is far enough from it that keeping the shader's value cannot pass. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.0, 0.0, 0.25);
}
