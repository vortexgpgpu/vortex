/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Opaque red that keeps only samples 1 and 3 through the shader's own coverage
 * output. Writing gl_SampleMask does not make the shader run per sample, so one
 * invocation decides the whole pixel. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(1.0, 0.0, 0.0, 1.0);
   gl_SampleMask[0] = 0xA;
}
