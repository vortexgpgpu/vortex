/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A custom border on an RGBA8 texture, (0.2, 0.6, 1.0, 0.4): on the 8-bit grid
 * so the texture unit carries it exactly, and with red and blue far apart, so a
 * border stored in the wrong channel order is visible -- both filtered and
 * gathered. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   const vec4 border = vec4(0.2, 0.6, 1.0, 0.4);
   const vec4 texel  = vec4(0.0, 40.0 / 255.0, 1.0, 1.0);
   const float tol   = 1.5 / 255.0;

   vec4 inside = texture(tex, vec2(0.5, 0.5));
   vec4 left   = texture(tex, vec2(-0.5, 0.5));
   vec4 red    = textureGather(tex, vec2(-0.5, 0.5), 0);
   vec4 blue   = textureGather(tex, vec2(-0.5, 0.5), 2);

   bool ok = all(lessThan(abs(inside - texel), vec4(tol)))
          && all(lessThan(abs(left - border), vec4(tol)))
          && all(lessThan(abs(red - vec4(border.r)), vec4(tol)))
          && all(lessThan(abs(blue - vec4(border.b)), vec4(tol)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
