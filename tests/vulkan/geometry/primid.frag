/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// The primitive's gl_PrimitiveID as its colour.
layout(location = 0) out vec4 o;
void main()
{
   uint c = uint(gl_PrimitiveID) % 15u + 1u;
   o = vec4(float(c & 1u), float((c >> 1) & 1u), float((c >> 2) & 1u), float((c >> 3) & 1u));
}
