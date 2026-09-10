/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three clearly different channel values, so a swapped or rotated channel
 * order is unambiguous in the read-back bytes. Equal channels -- the usual
 * white or grey test colour -- would hide exactly the defect this looks for. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.5, 0.25, 1.0);
}
