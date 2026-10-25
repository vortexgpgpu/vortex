/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * mp_image.comp in a vertex shader: vertex i is the invocation at (i % side,
 * i / side), and its partner is the mirrored vertex, run by another core.
 */
#version 450 core
#pragma use_vulkan_memory_model
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_buffer_reference : enable

layout(constant_id = 0) const int DIM = 1;
layout(constant_id = 1) const int NUM_WG = 1;

layout(set = 0, binding = 0, r32ui) uniform devicecoherent uimage2D payload;
layout(set = 0, binding = 1) buffer Result { uint x[]; } result;
layout(buffer_reference) buffer GuardRef { uint x[]; };
layout(push_constant, std430) uniform PC { GuardRef guard; };

void main()
{
   int side      = DIM * NUM_WG;
   ivec2 id      = ivec2(gl_VertexIndex % side, gl_VertexIndex / side);
   ivec2 partner = ivec2(side - 1) - id;
   uint coord        = uint(id.y * side + id.x);
   uint partnerCoord = uint(partner.y * side + partner.x);

   imageStore(payload, id, uvec4(coord + (imageLoad(payload, partner).x >> 31), 0, 0, 0));
   atomicExchange(guard.x[coord], 1u, gl_ScopeDevice, gl_StorageSemanticsImage,
                  gl_SemanticsRelease);
   bool skip = atomicExchange(guard.x[partnerCoord], 2u, gl_ScopeDevice,
                              gl_StorageSemanticsImage, gl_SemanticsAcquire) == 0u;
   uint r = imageLoad(payload, partner).x;
   uint flags = (skip ? 0u : 2u) | ((!skip && r != partnerCoord) ? 1u : 0u);
   if (flags != 0u)
      result.x[coord] |= flags;

   gl_PointSize = 1.0;
   gl_Position  = vec4(0.0, 0.0, 0.0, 1.0);
}
