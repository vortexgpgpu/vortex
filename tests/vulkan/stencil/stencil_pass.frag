/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Emits a constant colour; what the case measures is which fragments the
 * stencil test admits, not what the shader computes. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.0, 1.0, 0.0, 1.0);
}
