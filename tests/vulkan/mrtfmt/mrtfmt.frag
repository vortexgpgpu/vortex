/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two colour outputs carrying the SAME three channel values in opposite
 * orders. Reversing them is what makes the second attachment's check
 * meaningful: an attachment that was never written keeps the clear, and one
 * that was written with the first attachment's colour is equally wrong, so
 * only a merger that stored each target's own fragment passes both. */
#version 450

layout(location = 0) out vec4 rt0;
layout(location = 1) out vec4 rt1;

void main()
{
   rt0 = vec4(1.0, 0.5, 0.25, 1.0);
   rt1 = vec4(0.25, 0.5, 1.0, 1.0);
}
