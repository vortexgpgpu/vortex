/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * textureGather on layer 1 of a two-layer array texture. Layer 0 is a
 * constant grey and layer 1 carries the per-texel pattern, so a gather
 * that ignores the layer returns the grey and fails. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray tex;

/* Layer 1's green channel, which varies only with x. */
float g1(int x) { return float(x * 60 + 15) / 255.0; }

void main()
{
   vec4 g = textureGather(tex, vec3(0.5, 0.5, 1.0), 1);
   vec4 want = vec4(g1(1), g1(2), g1(2), g1(1));
   bool ok = all(lessThan(abs(g - want), vec4(0.01)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
