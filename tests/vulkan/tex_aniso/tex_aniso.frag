/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One implicit-LOD sample per pixel; the host judges what came back. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   out_color = vec4(texture(tex, v_uv).rgb, 1.0);
}
