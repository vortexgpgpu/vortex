/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The integer opaque-white border on an RGBA8 UINT texture reads (1, 1, 1, 1):
 * an integer, not the 255 of an 8-bit normalized white. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform usampler2D tex;

void main()
{
   uvec4 inside = texture(tex, vec2(0.5, 0.5));
   uvec4 right  = texture(tex, vec2(1.5, 0.5));
   uvec4 below  = texture(tex, vec2(0.5, 1.5));

   bool ok = inside == uvec4(7u, 9u, 11u, 13u)
          && right  == uvec4(1u)
          && below  == uvec4(1u);
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
