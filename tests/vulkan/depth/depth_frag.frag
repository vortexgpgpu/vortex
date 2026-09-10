/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Writes gl_FragDepth, inverting the depths the vertex stage interpolated:
 * the near (blue) triangle exports a far depth and the far (red) one a near
 * depth, so the depth test must let red win the centre. A driver that
 * ignored the shader's export would keep the plane's depth and leave blue
 * there, which is the unmodified test's expected result. */
#version 450

layout(location = 0) in  vec3 v_color;
layout(location = 0) out vec4 out_color;

void main()
{
   /* The blue triangle carries v_color.b = 1; push it to the back. */
   gl_FragDepth = v_color.b > 0.5 ? 0.9 : 0.1;
   out_color = vec4(v_color, 1.0);
}
