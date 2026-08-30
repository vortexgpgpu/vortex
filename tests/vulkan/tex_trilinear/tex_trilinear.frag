/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three explicit LODs on a two-level mip chain whose levels are flat and
 * distinct. The endpoints say which level was selected; the midpoint says
 * whether the two were blended or one was simply picked.
 *
 * Explicit LOD rather than a derivative-derived one: the value under test is
 * the mip blend, and deriving the LOD from a quad's derivatives would make the
 * assertion depend on the interpolation as well. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   float base = textureLod(tex, v_uv, 0.0).r;
   float mid  = textureLod(tex, v_uv, 0.5).r;
   float top  = textureLod(tex, v_uv, 1.0).r;

   out_color = vec4(base, mid, top, 1.0);
}
