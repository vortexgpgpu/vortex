/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Counts its invocations under early fragment tests: only fragments that
 * passed the depth and stencil tests may run it. */
#version 450

layout(early_fragment_tests) in;

layout(push_constant) uniform PC {
   float z;
   vec4  color;
} pc;

layout(std430, set = 0, binding = 0) buffer counter_t { uint n; } counter;
layout(location = 0) out vec4 out_color;

void main()
{
   atomicAdd(counter.n, 1u);
   out_color = pc.color;
}
