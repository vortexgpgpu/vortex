/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The input at offsets (-0.5, -0.5) and (0.4375, 0.4375) from the pixel
 * centre, which are the pixel's top-left corner and a point 1/16 short of its
 * bottom-right one. */
#version 450

layout(location = 0) noperspective in vec2 v_win;
layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 0) writeonly buffer Results {
   vec4 res[];
};

const int WIDTH = 8;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   int base = (p.y * WIDTH + p.x) * 4;
   res[base + 0] = vec4(interpolateAtOffset(v_win, vec2(-0.5, -0.5)),
                        interpolateAtOffset(v_win, vec2(0.4375, 0.4375)));
   out_color = vec4(v_win, 0.0, 1.0);
}
