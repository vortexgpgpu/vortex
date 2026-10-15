/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * No per-sample built-in is read: running once per sample comes from the
 * pipeline's minSampleShading alone. The input is the window position, so each
 * sample records where its invocation evaluated it. */
#version 450

layout(location = 0) in vec2 v_win;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(v_win, 0.0, 1.0);
}
