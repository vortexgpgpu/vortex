/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * An axis-aligned quad at one depth. The host places its edges on pixel
 * boundaries, so every pixel -- and every sample of a multisampled one -- is
 * wholly inside or wholly outside it and the passing count has one answer. */
#version 460

layout(push_constant) uniform PC {
   vec4 rect;     /* x0, y0, x1, y1 in clip space */
   vec4 color;
   float depth;
} pc;

void main()
{
   int v = gl_VertexIndex;
   bool right = v == 1 || v == 4 || v == 5;
   bool far_y = v == 2 || v == 3 || v == 5;
   gl_Position = vec4(right ? pc.rect.z : pc.rect.x,
                      far_y ? pc.rect.w : pc.rect.y, pc.depth, 1.0);
}
