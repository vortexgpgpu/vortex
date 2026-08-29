/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A flat white fragment. Where the triangle lands is the whole signal here, so
 * the colour carries nothing. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 1.0, 1.0, 1.0);
}
