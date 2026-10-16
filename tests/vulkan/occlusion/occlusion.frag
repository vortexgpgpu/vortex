/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A constant colour per draw, each channel 0 or 1, so the footprint check reads
 * which draw last wrote a pixel without depending on how the device rounds. */
#version 460

layout(push_constant) uniform PC {
   vec4 rect;
   vec4 color;
   float depth;
} pc;

layout(location = 0) out vec4 o_color;

void main()
{
   o_color = pc.color;
}
