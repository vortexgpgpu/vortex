/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpipe.glsl"

layout(location = 0) rayPayloadInEXT Payload pl;
layout(shaderRecordEXT, std430) buffer Sbt { uint sbt_data; };

void main()
{
   pl.w[P_KIND]   = K_MISS0;
   pl.w[P_RECORD] = sbt_data;
   pl.w[P_TMIN]   = floatBitsToUint(gl_RayTminEXT);
   pl.w[P_TMAX]   = floatBitsToUint(gl_RayTmaxEXT);
   pl.w[P_FLAGS]  = gl_IncomingRayFlagsEXT;
   pl.w[P_WDIR_Z] = floatBitsToUint(gl_WorldRayDirectionEXT.z);
}
