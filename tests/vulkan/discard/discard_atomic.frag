/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * demote: the demoted lane's atomic must not land either.
 *
 * An atomic is a memory write, and a helper invocation is forbidden from
 * performing one just as it is forbidden a plain store. The two travel
 * different paths through the translator, though -- a store's address is built
 * where the suppression is applied, an atomic's is built separately -- so a
 * driver can suppress one and not the other, and suppressing stores alone
 * leaves the more visible half of the feature broken.
 *
 * A single contended counter rather than one word per pixel: an atomic that
 * escaped suppression at a per-pixel address would write a word no live lane
 * touches and could be mistaken for an addressing fault, whereas here every
 * lane targets the same word and the count is exactly the number of lanes that
 * were allowed to run. */
#version 450
#extension GL_EXT_demote_to_helper_invocation : require

layout(set = 0, binding = 0) buffer Hits {
   uint hit[];
} hits;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   if ((int(v_uv.x * 64.0) & 1) == 1) {
      demote;
   }

   atomicAdd(hits.hit[0], 1u);

   o_color = vec4(1.0, 1.0, 0.0, 1.0);
}
