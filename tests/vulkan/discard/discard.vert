/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One oversized triangle covering the whole target, carrying a varying that
 * spans exactly 0..1 across the visible 64 pixels on each axis.
 *
 * The fragment shaders derive the pixel column from that varying rather than
 * from gl_FragCoord: at a pixel centre v_uv.x * 64.0 is x + 0.5, so truncating
 * recovers x exactly, half a unit clear of either boundary. gl_FragCoord on the
 * device path has no passing coverage in this suite, and a discard test that
 * depended on it could not say which of the two was broken.
 *
 * Oversized rather than inset so every quad is fully covered by the primitive:
 * the neighbour a derivative shuffles from is then a real fragment, not a
 * helper invocation generated at an edge. */
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
