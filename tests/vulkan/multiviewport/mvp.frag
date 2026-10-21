/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The primitive's viewport into red (40 + 50 * index) and its layer into green
 * (10 + 60 * layer): exact 1/255 steps, none equal to the clear. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(float(40 + 50 * gl_ViewportIndex) / 255.0,
                    float(10 + 60 * gl_Layer) / 255.0, 0.0, 1.0);
}
