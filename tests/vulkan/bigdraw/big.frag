/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(location = 0) flat in vec4 vcol;
layout(location = 0) out vec4 o;
void main()
{
   o = vcol;
}
