/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Screen-space derivatives. dFdx/dFdy are core fragment features -- no feature
 * bit gates them -- and the device computes them by shuffling within a pixel
 * quad, so they exercise the quad lane packing that nothing else in the suite
 * touches.
 *
 * v_uv spans 0..1 over 64 pixels, so dFdx(v_uv.x) is 1/64 and scaling by 64
 * gives exactly 1.0. The cross terms are zero: v_uv.x does not vary along y.
 *
 * Scaling to a full-range 1.0 rather than reading the raw derivative is what
 * makes the failure legible -- a derivative that comes back zero, the way an
 * unimplemented quad shuffle would, is then 255 away rather than a fraction of
 * a quantisation step. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   float dx  = dFdx(v_uv.x) * 64.0;        /* 1.0 */
   float dy  = dFdy(v_uv.y) * 64.0;        /* 1.0 */
   float off = abs(dFdy(v_uv.x)) * 64.0    /* 0.0 */
             + abs(dFdx(v_uv.y)) * 64.0;
   o_color = vec4(dx, dy, off, 1.0);
}
