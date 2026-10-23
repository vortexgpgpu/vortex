/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Opaque triangles. On the recursion row it traces a second ray, at the
 * procedural instance, from inside the closest-hit shader; that one's
 * closest-hit traces a third (recursion depth 3). */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpipe.glsl"
#include "rtpipe_hit.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(location = 0) rayPayloadInEXT Payload pl;
layout(location = 1) rayPayloadEXT Payload sec;
layout(shaderRecordEXT, std430) buffer Sbt { uint sbt_data; };
hitAttributeEXT vec2 bary;

void main()
{
   record_hit(pl, K_CHIT0, sbt_data, bary);
   if ((pl.w[P_LAUNCH] >> 16) == ROW_RECURSE) {
      for (uint i = 0u; i < P_WORDS; i++)
         sec.w[i] = 0u;
      sec.w[P_LAUNCH] = pl.w[P_LAUNCH];
      vec3 o = gl_WorldRayOriginEXT + vec3(20.0, 0.0, 0.0);
      traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xff, 0, 1, 0, o, 0.0,
                  vec3(0.0, 0.0, -1.0), 100.0, 1);
      pl.w[P_SEC_KIND] = sec.w[P_KIND];
      pl.w[P_SEC_T]    = sec.w[P_T];
      pl.w[P_SEC_CUST] = sec.w[P_CUSTOM];
      pl.w[P_THIRD]    = sec.w[P_THIRD];
   }
}
