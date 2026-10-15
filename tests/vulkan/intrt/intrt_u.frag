/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Unsigned outputs: red is wider than 8 bits and has a distinct high byte, so a
 * 16- or 32-bit texel shows its byte order and an 8-bit one shows saturation. */
#version 450

layout(location = 0) out uvec4 out_color;

void main()
{
   out_color = uvec4(0x1234u, 100u, 3u, 1u);
}
