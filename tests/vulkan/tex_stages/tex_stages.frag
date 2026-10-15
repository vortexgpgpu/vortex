/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two textures in one fragment shader, each through its own sampler: the
 * driver gives them a TEX stage each. The host holds the expected values. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex0;
layout(set = 0, binding = 1) uniform sampler2D tex1;

void main()
{
   /* Texel centres of tex0 (5x3, nearest, REPEAT): (3,1), the second through a
    * period to the left. The jitter stays inside a fifth of a texel. */
   float j = (v_uv.x - 0.5) * 0.05;
   float a = texture(tex0, vec2(3.5 / 5.0 + j, 1.5 / 3.0)).r;
   /* tex1 (4x2, linear, CLAMP_TO_EDGE): x = 0.35*4 - 0.5 = 0.9 blends columns
    * 0 and 1 of row 0 at 0.1 / 0.9. */
   float b = texture(tex1, vec2(0.35, 0.25)).r;
   /* Both stages feeding one result. */
   float c = texture(tex0, vec2(-1.0 + 0.5 / 5.0, 2.5 / 3.0)).r
           * texture(tex1, vec2(0.875, 0.75)).r;
   out_color = vec4(a, b, c, 1.0);
}
