#version 450
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One vertex per texel of rows [H, 2H) of a sparse image: imageStore of a
 * per-texel value, then imageLoad of it back into res.v. Nothing is rasterized.
 */

layout(binding = 0, r32ui) uniform uimage2D img;

layout(binding = 1, std430) buffer Out { uint v[]; } res;

layout(push_constant) uniform Extent { uvec2 e; } ext;

void main()
{
   uint i = uint(gl_VertexIndex);
   ivec2 c = ivec2(i % ext.e.x, ext.e.y + i / ext.e.x);
   uint k = ext.e.x * ext.e.y + i;
   imageStore(img, c, uvec4(0xc0000000u ^ (k * 2654435761u), 0u, 0u, 0u));
   res.v[k] = imageLoad(img, c).x;
   gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
   gl_PointSize = 1.0;
}
