/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Vertices come from the push constants (window-space, converted to NDC), so
 * each pass places its points and lines exactly. */
#version 450

layout(push_constant) uniform PC {
   vec4  v[4];        /* xy = window position, z = point size */
   vec2  size;        /* framebuffer extent */
   layout(offset = 80) vec2 inst;   /* window offset per instance */
} pc;

void main()
{
   vec4 v = pc.v[gl_VertexIndex];
   vec2 w = v.xy + float(gl_InstanceIndex) * pc.inst;
   gl_Position = vec4(w / pc.size * 2.0 - 1.0, 0.5, 1.0);
   gl_PointSize = v.z;
}
