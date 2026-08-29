/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each covered fragment contributes a fixed 0.25 and the pipeline blends
 * additively, so a pixel's final value counts how many DRAWS covered it. The
 * draws are coincident, so a plain coverage count cannot tell four draws from
 * one -- the sum is what distinguishes them. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.25, 0.25, 0.25, 1.0);
}
