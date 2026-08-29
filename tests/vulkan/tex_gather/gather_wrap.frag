/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * textureGather across the texture edge with CLAMP_TO_EDGE addressing: the
 * footprint centred on the corner texel has two taps outside the texture,
 * which must clamp back onto the edge texels rather than wrap. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

/* The red channel varies only with x. */
float red(int x) { return float(x * 60 + 15) / 255.0; }

void main()
{
   /* Centre on texel (0,0): taps at i0=-1 and j0=-1 clamp to column/row 0. */
   vec4 g = textureGather(tex, vec2(0.125, 0.125), 0);
   vec4 want = vec4(red(0), red(1), red(1), red(0));
   bool ok = all(lessThan(abs(g - want), vec4(0.01)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
