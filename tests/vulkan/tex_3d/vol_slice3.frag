/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Samples the last depth slice: a volume sampled as if it were 2D, or one
 * whose r axis is mis-scaled, returns a different slice's colour. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler3D tex;

void main()
{
   vec4 t = texture(tex, vec3(v_uv, 0.875));
   vec4 want = vec4(200.0 / 255.0, 110.0 / 255.0, 80.0 / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
