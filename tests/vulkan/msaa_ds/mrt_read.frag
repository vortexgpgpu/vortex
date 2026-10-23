/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Copies every sample of the multisample colour attachment into a buffer, one
 * vec4 per pixel and sample. A multisample image cannot be the source of a
 * copy, so texelFetch on a sampler2DMS is how its samples are seen
 * individually. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DMS ms_color;

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
      samples[base + k] = texelFetch(ms_color, p, k);
   }
   out_color = vec4(0.0);
}
