/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The source the blend combines with what is already there. Its channels are a
 * different permutation of the base's, so a blend that read the wrong operand
 * lands on a value no channel expects. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.125, 0.25, 0.1875, 1.0);
}
