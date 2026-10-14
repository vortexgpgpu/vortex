/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Black first source and a pushed second source, so with a ONE source factor
 * and a SRC1 destination factor the result is the destination scaled by the
 * second source alone. */
#version 450

layout(push_constant) uniform Push {
   vec4 src1;
} pc;

layout(location = 0, index = 0) out vec4 out_color;
layout(location = 0, index = 1) out vec4 out_src1;

void main()
{
   out_color = vec4(0.0);
   out_src1 = pc.src1;
}
