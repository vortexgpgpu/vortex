/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One constant colour, an exact 1/255 step so the check does not depend on
 * whether the device truncates or rounds a UNORM8. What is under test is which
 * views the fragment reached, not what it computed. */
#version 460

layout(location = 0) out vec4 o_color;

void main()
{
   o_color = vec4(1.0, 0.0, 0.0, 1.0);
}
