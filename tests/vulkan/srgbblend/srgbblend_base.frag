/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The destination the blend reads back. Drawn with blending off, so the stored
 * bytes are the device's own sRGB encode of these values -- which is what makes
 * the linear value behind them known, without trusting the clear.
 *
 * Chosen so each channel encodes to a byte that decodes back to within a third
 * of a count of the value it came from: any drift the test sees is the
 * blend's, not the transfer function's. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(0.125, 0.1875, 0.25, 1.0);
}
