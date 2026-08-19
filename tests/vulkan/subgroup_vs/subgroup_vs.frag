/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Forwards the vertex stage's subgroup report to the colour target. Each
 * component is byte-encoded so an 8-bit UNORM attachment carries it exactly:
 * v/255.0 quantises back to v with no rounding error for v in 0..255.
 */
#version 450

layout(location = 0) flat in uvec4 v_sg;
layout(location = 0) out vec4 o_color;

void main()
{
   o_color = vec4(v_sg) / 255.0;
}
