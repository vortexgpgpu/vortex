/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * draw3d fragment shader. MODE is a specialization constant set per
 * draw call, so each pipeline variant constant-folds to one path:
 *   0 = colour      (gouraud, vertex colour)
 *   1 = texture     (replace)
 *   2 = modulate    (texture * vertex colour)
 * It mirrors the fixed-function fragment work in the native draw3d
 * kernel; vortexpipe lowers texture() to the Vortex TEX unit. */
#version 450

layout(constant_id = 0) const int MODE = 0;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   if (MODE == 2)
      out_color = texture(tex, v_uv) * v_color;
   else if (MODE == 1)
      out_color = texture(tex, v_uv);
   else
      out_color = v_color;
}
