/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Samples the first depth slice of a 4x4x4 volume whose slices are
 * distinct solid colours. The r coordinate sits at the slice centre so
 * nearest filtering lands squarely on it. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler3D tex;

void main()
{
   vec4 t = texture(tex, vec3(v_uv, 0.125));
   /* slice s is (50s+50, 30s+20, 200-40s) / 255. */
   vec4 want = vec4(50.0 / 255.0, 20.0 / 255.0, 200.0 / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
