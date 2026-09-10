/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Highest-descriptor-set UBO test — fragment stage. Reads one uniform buffer
 * from descriptor SET 0 and another from descriptor SET 7, and combines both
 * into the output colour:
 *
 *   out = vec4(u0.color.rgb + u7.color.rgb, 1.0)
 *
 * Set 7 is the last set maxBoundDescriptorSets allows, and lavapipe binds
 * descriptor set N at constant-buffer index N+1, so u7 lives at index 8 —
 * one past the end of the driver's fragment descriptor table. Nothing
 * refuses a shader that reaches there; the index is a host-side table bound
 * that never reaches the translator, and the device reads whatever follows
 * the table and dereferences it as a constant-buffer base.
 *
 * Set 0 is read as well so a wrong answer distinguishes "set 7 was lost"
 * from "nothing rendered at all". */
#version 450

layout(set = 0, binding = 0) uniform U0 { vec4 color; } u0;
layout(set = 7, binding = 0) uniform U7 { vec4 color; } u7;

layout(location = 0) in  vec3 v_color;   /* unused: isolates the UBO path */
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(u0.color.rgb + u7.color.rgb, 1.0);
}
