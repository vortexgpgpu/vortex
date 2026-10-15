/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Copies every sample of the multisample attachment into a buffer, four
 * consecutive vec4 per pixel in sample order. A multisample image cannot be the
 * source of a copy, so texelFetch on a sampler2DMS is how its samples are seen
 * individually rather than through a resolve that would average them away. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DMS ms_tex;

layout(std430, set = 0, binding = 1) writeonly buffer Samples {
   vec4 samples[];
};

layout(push_constant) uniform PC {
   int width;
};

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   int base = (p.y * width + p.x) * 4;
   for (int k = 0; k < 4; k++) {
      samples[base + k] = texelFetch(ms_tex, p, k);
   }
   out_color = vec4(0.0);
}
