/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A single oversized triangle covering the whole target, handing the fragment
 * stage its normalised-device position. The ray origin is taken from that
 * varying rather than from gl_FragCoord, so the test measures the ray query
 * and not the driver's fragment-coordinate convention -- which has its own
 * cover elsewhere in this suite and a depth-convention history of its own. */
#version 460

layout(location = 0) out vec2 v_ndc;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
   v_ndc = p;
}
