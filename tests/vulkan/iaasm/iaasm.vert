/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each vertex record carries the gl_VertexIndex it must be read at. A vertex
 * read for any other index, or fetched from the wrong record, takes magenta as
 * its flat colour: the flat input then shows which vertex is provoking, and
 * the smooth one, in another channel order, every vertex of the primitive. */
#version 450

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_col;
layout(location = 2) in float a_id;

layout(location = 0) flat out vec3 v_flat;
layout(location = 1) out vec3 v_col;

layout(push_constant) uniform pc_t { float inst_dx; } pc;

void main()
{
   gl_Position = vec4(a_pos.x + pc.inst_dx * float(gl_InstanceIndex), a_pos.y, 0.0, 1.0);
   gl_PointSize = 3.0;
   v_flat = gl_VertexIndex == int(a_id) ? a_col : vec3(1.0, 0.0, 1.0);
   v_col = a_col.bgr;
}
