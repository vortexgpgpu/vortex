/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// The vertex as given, each instance 32 pixels further down.
layout(location = 0) in vec2 pos;
void main()
{
   gl_Position = vec4(pos + vec2(0.0, float(gl_InstanceIndex)), 0.0, 1.0);
}
