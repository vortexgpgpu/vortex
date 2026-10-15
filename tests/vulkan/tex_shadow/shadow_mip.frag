/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sampler2DShadow with a linear mip filter, compare LESS: level 0 holds depth
 * 0.25 and level 1 depth 0.75, so reference 0.5 fails on level 0 and passes on
 * level 1, and halfway between the levels the two results blend to one half. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DShadow tex;

void main()
{
   vec3 p = vec3(0.5, 0.5, 0.5);
   float l0 = textureLod(tex, p, 0.0);
   float l1 = textureLod(tex, p, 1.0);
   float lh = textureLod(tex, p, 0.5);
   bool ok = l0 < 0.02 && l1 > 0.98 && abs(lh - 0.5) < 0.02;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
