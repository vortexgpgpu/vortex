/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One opaque colour. The test compares whole images, so the fragment stage
 * only has to be deterministic. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.0, 1.0, 0.0, 1.0);
}
