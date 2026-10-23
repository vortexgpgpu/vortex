/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Procedural hits. Reached by the recursion row's second ray, it traces a
 * third one, which misses. */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpipe.glsl"
#include "rtpipe_hit.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(location = 0) rayPayloadInEXT Payload pl;
layout(location = 2) rayPayloadEXT Payload third;
layout(shaderRecordEXT, std430) buffer Sbt { uint sbt_data; };
hitAttributeEXT vec2 bary;

void main()
{
   record_hit(pl, K_CHIT2, sbt_data, bary);
   if ((pl.w[P_LAUNCH] >> 16) == ROW_RECURSE) {
      third.w[P_KIND] = 0u;
      traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xff, 0, 1, 0, vec3(5.0, 0.0, 2.0), 0.0,
                  vec3(0.0, 0.0, -1.0), 100.0, 2);
      pl.w[P_THIRD] = third.w[P_KIND];
   }
}
