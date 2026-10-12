/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Layer k is (k & 1, (k >> 1) & 1, (k >> 2) & 1) at half coverage. Two
 * consecutive layers always differ in red, so a pair that merges out of order
 * moves the result by a quarter of the range or more.
 *
 * Even layers spin before they export and odd layers do not, so without an
 * ordered merge a later layer's warp routinely overtakes an earlier layer's at
 * the same pixel. The spin's result is folded in through a step that is always
 * zero (fract() stays below 1), so it costs time without changing the colour. */
#version 450

layout(location = 0) flat in int v_layer;

layout(location = 0) out vec4 out_color;

void main()
{
   float acc = gl_FragCoord.x * 0.001;
   int spins = ((v_layer & 1) == 0) ? 256 : 0;
   for (int i = 0; i < spins; i++)
      acc = fract(acc * 1.37 + 0.11);
   out_color = vec4(float(v_layer & 1) + step(2.0, acc), float((v_layer >> 1) & 1),
                    float((v_layer >> 2) & 1), 0.5);
}
