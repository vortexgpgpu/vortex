/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Textured-quad fragment shader: samples the bound 2D texture at the
 * interpolated coordinate. vortexpipe translates `texture()` to the
 * Vortex TEX hardware unit (vx_tex); the fragment stage runs on the
 * Vortex device (Phase 6). */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   out_color = texture(tex, v_uv);
}
