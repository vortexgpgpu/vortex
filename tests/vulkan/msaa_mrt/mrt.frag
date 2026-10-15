/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each primitive writes its own identity into both attachments, so a sample in
 * which the two attachments disagree took its colours from different depth
 * decisions. */
#version 450

layout(location = 0) flat in int v_prim;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_id;

void main()
{
   out_color = v_prim == 0 ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
   out_id = v_prim == 0 ? 1.0 : 2.0;
}
