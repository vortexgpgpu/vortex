/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A custom float border on an RGBA32F texture: (0.3, 0.6, 1.5, 0.25), off the
 * 8-bit grid and outside [0,1], so only a border carried at full precision
 * matches it. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   const vec4 border = vec4(0.3, 0.6, 1.5, 0.25);
   const vec4 texel  = vec4(2.0, -1.0, 0.5, 1.0);

   vec4 inside = texture(tex, vec2(0.5, 0.5));
   vec4 left   = texture(tex, vec2(-0.5, 0.5));
   vec4 above  = texture(tex, vec2(0.5, -0.5));

   bool ok = all(lessThan(abs(inside - texel), vec4(1e-4)))
          && all(lessThan(abs(left  - border), vec4(1e-4)))
          && all(lessThan(abs(above - border), vec4(1e-4)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
