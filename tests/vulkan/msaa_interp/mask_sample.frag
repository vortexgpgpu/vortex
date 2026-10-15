/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One invocation per sample: each records the coverage it was given and the
 * sample it shades, in that sample's slot. */
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
   res[base + gl_SampleID] = vec4(float(gl_SampleMaskIn[0]), float(gl_SampleID), 0.0, 1.0);
   out_color = vec4(0.0, 0.0, 0.0, 1.0);
}
