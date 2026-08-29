/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * draw3d vertex shader. The CGLTrace vertices are already in clip
 * space (the trace captured post-transform geometry), so the vertex
 * stage is a passthrough -- it just fetches the per-vertex attributes
 * from the bound vertex buffer and forwards them. Running it on the
 * Vortex device exercises vortexpipe's vertex-buffer input path. */
#version 450

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

void main()
{
   gl_Position = in_pos;
   v_color     = in_color;
   v_uv        = in_uv;
}
