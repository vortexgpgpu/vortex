/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One invocation per pixel, recording the coverage it was given. Nothing here
 * asks for sample-rate shading. */
#version 450

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 0) writeonly buffer Results {
   vec4 res[];
};

const int WIDTH = 8;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   int base = (p.y * WIDTH + p.x) * 4;
   res[base] = vec4(float(gl_SampleMaskIn[0]), 0.0, 0.0, 1.0);
   out_color = vec4(0.0, 0.0, 0.0, 1.0);
}
