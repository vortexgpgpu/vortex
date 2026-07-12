/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * W8 draw-instancing vertex shader. A self-contained gl_VertexIndex
 * triangle (no vertex buffers) offset in X by gl_InstanceIndex, so a
 * 3-instance draw renders three separated triangles — one per screen
 * third. This exercises the on-device instancing path: the VS runs
 * instance_count × 3 threads and resolves gl_InstanceIndex from the
 * device VS arg block. If gl_InstanceIndex were ignored, all three
 * instances would overlap in the centre third and the host check fails. */
#version 450

layout(location = 0) out vec3 v_color;

vec2 positions[3] = vec2[](
   vec2( 0.0,  -0.3),
   vec2( 0.18,  0.3),
   vec2(-0.18,  0.3)
);

/* one colour per instance so a correct render is also colour-distinct */
vec3 colors[3] = vec3[](
   vec3(1.0, 0.2, 0.2),
   vec3(0.2, 1.0, 0.2),
   vec3(0.2, 0.2, 1.0)
);

void main()
{
   /* instances 0,1,2 -> x offset -0.6, 0.0, +0.6 (left / centre / right) */
   float dx = (float(gl_InstanceIndex) - 1.0) * 0.6;
   gl_Position = vec4(positions[gl_VertexIndex] + vec2(dx, 0.0), 0.0, 1.0);
   v_color = colors[gl_InstanceIndex];
}
