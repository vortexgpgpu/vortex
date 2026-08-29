/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * White, so the constant-colour factor alone decides the result and the
 * alpha equation is visible in the alpha channel. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 1.0, 1.0, 1.0);
}
