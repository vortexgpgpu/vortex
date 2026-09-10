/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One oversized triangle covering the whole target, carrying a varying that
 * spans exactly 0..1 across the visible 64 pixels on each axis. That makes the
 * expected screen-space derivative exactly 1/64 per pixel, a number the test
 * can state rather than derive from the triangle's clipped extent.
 *
 * Oversized rather than inset on purpose: every sampled pixel is then interior
 * to the primitive, so the quad a derivative is taken over is fully covered and
 * the result does not depend on helper-invocation behaviour at an edge. */
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
