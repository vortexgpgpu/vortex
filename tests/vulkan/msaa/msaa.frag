/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A constant opaque red. The fragment stage carries nothing of its own here:
 * what is under test is that the merger keeps four samples per pixel and that
 * the resolve averages them, so the shaded value has to be a constant for a
 * partial pixel's value to mean coverage and nothing else. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
