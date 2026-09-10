/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The destination the blend reads back. Drawn with blending off, so the stored
 * bytes are the merger's own sRGB encode of these values -- which is what makes
 * the linear value behind them known exactly, without trusting the clear.
 *
 * Chosen so each channel encodes to a byte that decodes back to exactly the
 * linear value it came from: the round-trip loss is then zero and any drift the
 * test sees is the blend's, not the transfer function's. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.125, 0.1875, 0.25, 1.0);
}
