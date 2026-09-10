/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * All four varyings are declared so the fragment stage's plane demand matches
 * the vertex stage's, but only the last one reaches the colour target: it is
 * the one that falls past the twelve-plane budget, so it is the one whose loss
 * a colour check can see. The other three are referenced with a zero weight so
 * no optimiser can drop them from the interface. */
#version 450

layout(location = 0) in vec4 v0;
layout(location = 1) in vec4 v1;
layout(location = 2) in vec4 v2;
layout(location = 3) in vec4 v3;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = v3 + 0.0 * (v0 + v1 + v2);
}
