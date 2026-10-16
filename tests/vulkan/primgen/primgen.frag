/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT */
#version 460

layout(push_constant) uniform PC {
   uint strip;
   uint height;
} pc;

layout(location = 0) out vec4 o_color;

void main()
{
   o_color = pc.strip != 0u ? vec4(1.0, 0.0, 0.0, 1.0)
                            : vec4(0.0, 1.0, 0.0, 1.0);
}
