/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each view stamps its own index into red: 60, 140, 220 for views 0, 1 and 2.
 * The steps are exact 1/255 multiples so no rounding is involved, they are far
 * apart so a wrong view is a large error, and none of them is the clear -- a
 * view the draw never reached reads differently from one that read the wrong
 * index. */
#version 450
#extension GL_EXT_multiview : require

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(float(60 + 80 * gl_ViewIndex) / 255.0, 0.0, 0.0, 1.0);
}
