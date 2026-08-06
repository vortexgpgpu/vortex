/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multiple-render-target fragment shader (issue I4). Writes THREE
 * colour attachments, each a DISTINCT value computed from the same
 * interpolated varying so a correct render proves every attachment was
 * shaded (not just RT0). Each output isolates one channel of v_color:
 *
 *   RT0 = (v_color.r, 0, 0, 1)   -- red only
 *   RT1 = (0, v_color.g, 0, 1)   -- green only
 *   RT2 = (0, 0, v_color.b, 1)   -- blue only
 *
 * At the frame centre v_color is the barycentric blend of the three
 * vertex colours (~0.33 each), so RT0 is red-ish, RT1 green-ish and RT2
 * blue-ish -- three distinct, non-black colours. A driver that wired
 * only RT0 into the live draw path would leave RT1/RT2 at the (black)
 * clear, which the host check detects. */
#version 450

layout(location = 0) in  vec3 v_color;

layout(location = 0) out vec4 rt0;
layout(location = 1) out vec4 rt1;
layout(location = 2) out vec4 rt2;

void main()
{
   rt0 = vec4(v_color.r, 0.0, 0.0, 1.0);
   rt1 = vec4(0.0, v_color.g, 0.0, 1.0);
   rt2 = vec4(0.0, 0.0, v_color.b, 1.0);
}
