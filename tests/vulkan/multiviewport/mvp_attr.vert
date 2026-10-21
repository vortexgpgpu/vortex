/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Every vertex carries its clip position and the viewport index, layer and
 * point size it writes, so a case places each primitive exactly and can give
 * the vertices of one primitive different indices. */
#version 450
#extension GL_ARB_shader_viewport_layer_array : require

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 in_sel;   /* viewport index, layer, point size */

void main()
{
   gl_Position = in_pos;
   gl_PointSize = in_sel.z;
   gl_ViewportIndex = int(in_sel.x);
   gl_Layer = int(in_sel.y);
}
