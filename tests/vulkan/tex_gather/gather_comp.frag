/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * textureGather component select on a 4x4 RGBA8 texture. Gathers each of
 * the four channels around one interior footprint centre and checks the
 * values and their GL order: .x = (i0,j1) .y = (i1,j1) .z = (i1,j0)
 * .w = (i0,j0). */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

/* Texel (x,y) of the uploaded pattern, per channel. */
float texel(int x, int y, int c)
{
   if (c == 0) return float(x * 60 + 15) / 255.0;
   if (c == 1) return float(y * 60 + 15) / 255.0;
   if (c == 2) return 200.0 / 255.0;
   return 1.0;
}

/* Expected gather of channel c around the footprint whose lower-left
 * texel is (i0,j0), in GL order. */
vec4 want(int i0, int j0, int c)
{
   return vec4(texel(i0,     j0 + 1, c), texel(i0 + 1, j0 + 1, c),
               texel(i0 + 1, j0,     c), texel(i0,     j0,     c));
}

void main()
{
   /* Centre of the footprint spanning texels (1,1)..(2,2) of a 4x4. */
   vec2 uv = vec2(0.5, 0.5);
   bool ok = true;
   ok = ok && all(lessThan(abs(textureGather(tex, uv, 0) - want(1, 1, 0)), vec4(0.01)));
   ok = ok && all(lessThan(abs(textureGather(tex, uv, 1) - want(1, 1, 1)), vec4(0.01)));
   ok = ok && all(lessThan(abs(textureGather(tex, uv, 2) - want(1, 1, 2)), vec4(0.01)));
   ok = ok && all(lessThan(abs(textureGather(tex, uv, 3) - want(1, 1, 3)), vec4(0.01)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
