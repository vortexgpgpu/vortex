/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Reports what three probes sampled from a 5x3 texture, rather than judging
 * them: the host holds the expected triple.
 *
 * Every texel carries its own flattened index, so a probe that lands on the
 * wrong row is as visible as one that lands on the wrong column. A texture
 * whose dimensions are not powers of two cannot be addressed by shifting, and
 * 5 and 3 share no factor with any power of two, so a shift-based address
 * cannot land on the right texel by accident. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   /* Each probe sits on a texel centre so NEAREST has no boundary to round.
    * The jitter stays well inside one texel on the shorter axis (a fifth of
    * the width, so 0.025 is an eighth of one) and only serves to give a warp's
    * lanes different addresses while still selecting the same texel. */
   float j = (v_uv.x - 0.5) * 0.05;

   /* Centres of (4,0) and (2,2), then (1,1) reached through REPEAT from one
    * period to the right. */
   float col = texture(tex, vec2(4.5 / 5.0 + j, 0.5 / 3.0)).r;
   float row = texture(tex, vec2(2.5 / 5.0 + j, 2.5 / 3.0)).r;
   float rep = texture(tex, vec2(1.0 + 1.5 / 5.0 + j, 1.5 / 3.0)).r;

   out_color = vec4(col, row, rep, 1.0);
}
