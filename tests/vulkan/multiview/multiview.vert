/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A single oversized triangle covering the whole target, so every pixel of
 * every view is either the draw's colour or the clear and a partially rendered
 * view cannot be mistaken for an edge. gl_ViewIndex is deliberately not read:
 * the per-view replay belongs to the render pass, and a shader asking for the
 * index would refuse the device path and hide the case under test. */
#version 460

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
}
