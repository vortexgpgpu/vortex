/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One invocation per pixel, with the window position taken at the centroid:
 * the pixel centre when every sample is covered, and a point inside the
 * primitive when only some are. */
#version 450

layout(location = 0) centroid in vec2 v_win;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(v_win, 0.0, 1.0);
}
