/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multiple-render-target fragment shader. Writes EIGHT colour attachments,
 * the most the output merger keeps state for. RT0..RT2 each isolate one
 * channel of the interpolated varying, so a correct render proves every
 * attachment was shaded (not just RT0):
 *
 *   RT0 = (v_color.r, 0, 0, 1)   -- red only
 *   RT1 = (0, v_color.g, 0, 1)   -- green only
 *   RT2 = (0, 0, v_color.b, 1)   -- blue only
 *
 * RT3..RT7 write exact constants, (k/8, 1 - k/8, 1/2, 1) for attachment k,
 * which the host checks texel for texel. The attachment index rides the
 * export's depth word, so RT7 sets every bit of that field: an index that
 * lost a bit would land on a lower attachment and leave RT7 at its clear. */
#version 450

layout(location = 0) in  vec3 v_color;

layout(location = 0) out vec4 rt0;
layout(location = 1) out vec4 rt1;
layout(location = 2) out vec4 rt2;
layout(location = 3) out vec4 rt3;
layout(location = 4) out vec4 rt4;
layout(location = 5) out vec4 rt5;
layout(location = 6) out vec4 rt6;
layout(location = 7) out vec4 rt7;

vec4 fixed_colour(float k)
{
   return vec4(k / 8.0, 1.0 - k / 8.0, 0.5, 1.0);
}

void main()
{
   rt0 = vec4(v_color.r, 0.0, 0.0, 1.0);
   rt1 = vec4(0.0, v_color.g, 0.0, 1.0);
   rt2 = vec4(0.0, 0.0, v_color.b, 1.0);
   rt3 = fixed_colour(3.0);
   rt4 = fixed_colour(4.0);
   rt5 = fixed_colour(5.0);
   rt6 = fixed_colour(6.0);
   rt7 = fixed_colour(7.0);
}
