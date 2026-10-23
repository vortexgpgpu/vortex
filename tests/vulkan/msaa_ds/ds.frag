/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each primitive writes its own colour, so a stored sample names the primitive
 * that won its depth test. */
#version 450

layout(location = 0) flat in int v_prim;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = v_prim == 0 ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
}
