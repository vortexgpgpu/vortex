/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One oversized triangle covering the whole target, carrying a varying that
 * spans exactly 0..1 across the visible 64 pixels on each axis.
 *
 * The fragment shader derives its image coordinate from that varying rather
 * than from gl_FragCoord: at a pixel centre v_uv * 64.0 is x + 0.5, so
 * truncating recovers the texel exactly, half a unit clear of either boundary.
 * gl_FragCoord on the device path has no passing coverage in this suite, and a
 * test that depended on it could not say which of the two was broken.
 *
 * Oversized so the primitive covers the whole target and every texel of the
 * storage image is written by exactly one fragment. */
#version 450

layout(location = 0) out vec2 v_uv;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
   /* p maps -1..1 -> 0..1 over the visible area. */
   v_uv = (p + 1.0) * 0.5;
}
