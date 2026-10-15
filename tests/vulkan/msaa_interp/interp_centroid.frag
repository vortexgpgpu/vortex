/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The input at the centroid, from one pixel-rate invocation. */
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
   res[base] = vec4(interpolateAtCentroid(v_win), 0.0, 1.0);
   out_color = vec4(v_win, 0.0, 1.0);
}
