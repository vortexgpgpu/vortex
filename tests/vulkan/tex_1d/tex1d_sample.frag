/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Samples a 4-texel 1D texture at each texel centre. A 1D sampler carries
 * a single coordinate component, which is the whole point of the case. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler1D tex;

void main()
{
   bool ok = true;
   for (int i = 0; i < 4; i++) {
      vec4 t = texture(tex, (float(i) + 0.5) / 4.0);
      vec4 want = vec4(float(60 * i + 20) / 255.0,
                       float(200 - 50 * i) / 255.0, 40.0 / 255.0, 1.0);
      ok = ok && all(lessThan(abs(t - want), vec4(0.02)));
   }
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
