/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * discard: the fragment must not reach the colour attachment.
 *
 * Every odd column is discarded, so each pixel quad is left half alive. A
 * pattern coarser than one column -- a half-plane, say -- would leave whole
 * quads on one side of the split and never ask what happens when a quad is
 * partly discarded, which is the case the device's coverage fold exists for.
 *
 * The derivative is taken before the discard so that this shader tests one
 * thing: whether a discarded fragment's colour is withheld. Whether a discarded
 * lane keeps running is a separate question, and discard_demote.frag asks it. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   float dx = dFdx(v_uv.x) * 64.0;      /* 1.0 */

   if ((int(v_uv.x * 64.0) & 1) == 1) {
      discard;
   }

   o_color = vec4(dx, 1.0, 0.0, 1.0);
}
