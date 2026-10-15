/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sampler2DShadow clamped to an opaque-white border, compare LESS on the 4x4
 * D32F ramp: outside the texture the tap reads the border's depth 1, so
 * reference 0.5 passes there, while inside, at depth 0.125, it fails. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DShadow tex;

void main()
{
   float outside_r = texture(tex, vec3(1.5, 0.5, 0.5));
   float outside_l = texture(tex, vec3(-0.5, 0.5, 0.5));
   float inside    = texture(tex, vec3(0.125, 0.5, 0.5));
   bool ok = outside_r > 0.5 && outside_l > 0.5 && inside < 0.5;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
