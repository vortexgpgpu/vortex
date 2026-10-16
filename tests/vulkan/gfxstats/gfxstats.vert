/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two triangles tiling the viewport exactly once, shifted by a push-constant
 * offset so the same pipeline can also place them wholly outside it. Every
 * instance lands in the same place. */
#version 460

layout(push_constant) uniform PC {
   vec2 offset;
} pc;

void main()
{
   int v = gl_VertexIndex;
   vec2 p = vec2((v == 1 || v == 4 || v == 5) ? 1.0 : -1.0,
                 (v == 2 || v == 3 || v == 5) ? 1.0 : -1.0);
   gl_Position = vec4(p + pc.offset, 0.0, 1.0);
}
