/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Discards every fragment after early fragment tests. The tests already ran,
 * so the depth and stencil they wrote stay written. */
#version 450

layout(early_fragment_tests) in;

layout(std430, set = 0, binding = 0) buffer counter_t { uint n; } counter;
layout(location = 0) out vec4 out_color;

void main()
{
   atomicAdd(counter.n, 1u);
   discard;
}
