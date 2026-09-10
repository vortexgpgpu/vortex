/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Samples layer 2 of a three-layer array texture. Each layer is a
 * distinct solid colour, so a dropped or mis-scaled layer index shows up
 * as the wrong colour rather than as noise. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray tex;

void main()
{
   vec4 t = texture(tex, vec3(v_uv, float(2)));
   /* layer L is (40*L+40, 20*L+10, 200-60*L) / 255. */
   vec4 want = vec4(float(40 * 2 + 40) / 255.0,
                    float(20 * 2 + 10) / 255.0,
                    float(200 - 60 * 2) / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
