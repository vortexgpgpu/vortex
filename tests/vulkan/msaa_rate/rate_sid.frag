/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Reading gl_SampleID makes the shader run once per sample. Each sample then
 * records which invocation wrote it and where that invocation was told it sits
 * in the pixel. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(float(gl_SampleID), gl_SamplePosition, 1.0);
}
