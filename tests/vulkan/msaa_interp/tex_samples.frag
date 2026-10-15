/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The sample count of the 4x attachment, queried through a sampler2DMS. */
#version 450

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 0) writeonly buffer Results {
   vec4 res[];
};

layout(set = 0, binding = 1) uniform sampler2DMS ms_tex;

const int WIDTH = 8;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   int base = (p.y * WIDTH + p.x) * 4;
   res[base] = vec4(float(textureSamples(ms_tex)), 0.0, 0.0, 1.0);
   out_color = vec4(0.0);
}
