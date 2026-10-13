/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Counts its invocations and writes no colour: the attachment-less pass sees
 * the fragment stage only through this counter. */
#version 450

layout(std430, set = 0, binding = 0) buffer counter_t { uint n; } counter;

void main()
{
   atomicAdd(counter.n, 1u);
}
