/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Copies every sample of both multisample attachments into a buffer: per pixel
 * and sample, the RGBA8 colour, then the R32F value in the next vec4's x. A
 * multisample image cannot be the source of a copy, so texelFetch on a
 * sampler2DMS is how its samples are seen individually. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DMS ms_color;
layout(set = 0, binding = 1) uniform sampler2DMS ms_id;

layout(std430, set = 0, binding = 2) writeonly buffer Samples {
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
      samples[(base + k) * 2 + 0] = texelFetch(ms_color, p, k);
      samples[(base + k) * 2 + 1] = vec4(texelFetch(ms_id, p, k).x, 0.0, 0.0, 0.0);
   }
   out_color = vec4(0.0);
}
