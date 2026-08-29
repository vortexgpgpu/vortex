/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * demote: the demoted lane's side effects must not land.
 *
 * Suppressing the colour export is not enough. A helper invocation may not
 * write memory either, and because the lane goes on executing -- which is what
 * makes its quad neighbours' derivatives work -- the store below is reached by
 * demoted and live lanes alike. Only the live ones may commit it.
 *
 * One word per pixel rather than a counter: a count says how many stores landed
 * but not which lanes wrote them, and a sink that suppressed the wrong half
 * would produce the same count. */
#version 450
#extension GL_EXT_demote_to_helper_invocation : require

layout(set = 0, binding = 0) buffer Hits {
   uint hit[];
} hits;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   int x = int(v_uv.x * 64.0);
   int y = int(v_uv.y * 64.0);

   if ((x & 1) == 1) {
      demote;
   }

   hits.hit[y * 64 + x] = 1u;

   o_color = vec4(1.0, 1.0, 0.0, 1.0);
}
