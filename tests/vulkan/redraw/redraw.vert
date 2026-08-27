/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle out of a real vertex buffer, drawn repeatedly. The shader is
 * deliberately trivial: what this test measures is the driver's handling of the
 * SAME vertex buffer across successive draws, not anything the shader does. */
#version 450

layout(location = 0) in vec2 in_pos;

void main()
{
   gl_Position = vec4(in_pos, 0.0, 1.0);
}
