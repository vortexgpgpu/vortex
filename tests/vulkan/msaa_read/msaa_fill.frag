/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A constant opaque red, written into the multisample attachment. Red and the
 * pass's green clear differ in two channels at full scale, so a pixel read back
 * later is one or the other and never something a rounding could confuse. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
