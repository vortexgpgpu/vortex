/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Encode the two flat varyings into channels the host can read back.
 *
 * v_id is scaled by 32 rather than written raw: a value of 1 in the low bits of
 * a colour channel is one quantisation step away from 0, so the defect it is
 * meant to catch -- the integer arriving as 0 -- would be within rounding of a
 * pass. At 32 per unit the two are unmistakable. */
#version 450

layout(location = 0) flat in uint  v_id;
layout(location = 1) flat in float v_f;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(float(v_id) * (32.0 / 255.0), v_f, 0.0, 1.0);
}
