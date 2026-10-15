/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sampler2DShadow through a linear filter on a 4x4 D16 ramp, compare LESS:
 * halfway between the x = 0 and x = 1 texel centres the four taps split two
 * and two, so a reference between their depths gives exactly one half, and
 * references below or above both give one and zero. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DShadow tex;

void main()
{
   /* Texel depth is (x + 1) / 8: 0.125 at x = 0, 0.25 at x = 1. u = 0.25 is
    * the midpoint of their centres; every row holds the same depths. */
   float half_pass = texture(tex, vec3(0.25, 0.5, 0.2));    /* one column passes */
   float all_pass  = texture(tex, vec3(0.25, 0.5, 0.05));   /* both pass         */
   float none      = texture(tex, vec3(0.25, 0.5, 0.3));    /* neither passes    */
   bool ok = abs(half_pass - 0.5) < 0.02 && all_pass > 0.98 && none < 0.02;
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
