/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * demote: the demoted lane keeps executing.
 *
 * A demoted invocation becomes a helper. It produces no output, but it must
 * still run, because a live neighbour in its quad may shuffle a value out of it
 * -- which is exactly what the derivative below does. Odd columns are demoted,
 * so every surviving lane's horizontal partner is a helper; if the device
 * implemented demote by ending the lane, the shuffle would return the live
 * lane's own value and the derivative would collapse to zero.
 *
 * The derivative is taken after the demote on purpose. Taking it first, as
 * discard_kill.frag does, cannot distinguish the two implementations. */
#version 450
#extension GL_EXT_demote_to_helper_invocation : require

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   if ((int(v_uv.x * 64.0) & 1) == 1) {
      demote;
   }

   float dx = dFdx(v_uv.x) * 64.0;      /* 1.0 */

   o_color = vec4(dx, 1.0, 0.0, 1.0);
}
