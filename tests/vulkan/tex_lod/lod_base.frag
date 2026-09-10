/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Explicit textureLod at level 0 of a mip chain whose levels are distinct
 * solid colours. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   vec4 t = textureLod(tex, v_uv, 0.0);
   vec4 want = vec4(240.0 / 255.0, 20.0 / 255.0, 20.0 / 255.0, 1.0);
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
