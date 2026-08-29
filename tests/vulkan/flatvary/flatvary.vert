/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three vertices carrying three DIFFERENT flat values. That is the whole point:
 * with equal values an interpolated varying and a flat one agree everywhere,
 * which is exactly how the driver carried flat varyings wrongly for so long
 * without any test noticing.
 *
 * v_id is an integer, so it is flat by Vulkan rule rather than by choice, and
 * its value is small -- 1, 2, 3 -- which is what makes it vanish if it is ever
 * reinterpreted as a float and quantised: those bit patterns are denormals.
 *
 * The provoking vertex under Vulkan's default is the first, so every fragment
 * of this triangle must read v_id = 1 and v_f = 0.25. */
#version 450

layout(location = 0) flat out uint  v_id;
layout(location = 1) flat out float v_f;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  0.9 : -0.9,
                 (gl_VertexIndex == 2) ?  0.9 : -0.9);
   gl_Position = vec4(p, 0.0, 1.0);
   v_id = uint(gl_VertexIndex) + 1u;
   v_f  = 0.25 + 0.25 * float(gl_VertexIndex);
}
