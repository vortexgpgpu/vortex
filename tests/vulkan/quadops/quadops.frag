/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each fragment's code is its pixel position. The quad swaps and the broadcast
 * must hand back the code of the quad neighbour they name, helper or not, and
 * a swap of gl_HelperInvocation tells a fragment whether that neighbour is a
 * helper. Red holds the four shuffle checks as bits, green the two helper
 * flags. */
#version 450
#extension GL_KHR_shader_subgroup_quad : require

layout(location = 0) out vec4 o_color;

uint code(uvec2 p) { return p.x | (p.y << 8); }

void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   uint c = code(p);
   uint ok = 0u;
   ok |= (subgroupQuadSwapHorizontal(c) == code(p ^ uvec2(1u, 0u))) ? 1u : 0u;
   ok |= (subgroupQuadSwapVertical(c)   == code(p ^ uvec2(0u, 1u))) ? 2u : 0u;
   ok |= (subgroupQuadSwapDiagonal(c)   == code(p ^ uvec2(1u, 1u))) ? 4u : 0u;
   ok |= (subgroupQuadBroadcast(c, 0u)  == code(p & ~uvec2(1u)))    ? 8u : 0u;
   uint h = gl_HelperInvocation ? 1u : 0u;
   uint helpers = subgroupQuadSwapHorizontal(h) | (subgroupQuadSwapVertical(h) << 1);
   o_color = (vec4(float(ok * 16u), float(helpers * 64u), 200.0, 255.0) + 0.25) / 255.0;
}
