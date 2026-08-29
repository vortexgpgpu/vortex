/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The same triangle as discard.vert, with one storage-buffer write per vertex.
 *
 * A draw relocates the vertex stage's buffer descriptors exactly as it does the
 * fragment stage's, so a vertex shader's store lands in the same device copy
 * and needs the same copy back. Nothing else in the suite writes memory from a
 * vertex shader, so that half of the path was carried entirely by the fragment
 * case happening to be the one anybody looked at.
 *
 * The written value varies with the vertex so the check cannot pass on a buffer
 * that was merely filled with a constant, and starts away from zero so it
 * cannot pass on one that was never written at all. */
#version 450

layout(set = 0, binding = 0) buffer Hits {
   uint hit[];
} hits;

layout(location = 0) out vec2 v_uv;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
   v_uv = (p + 1.0) * 0.5;

   hits.hit[gl_VertexIndex] = 0xA5u + uint(gl_VertexIndex);
}
