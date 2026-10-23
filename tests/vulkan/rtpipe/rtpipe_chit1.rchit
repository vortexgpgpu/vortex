/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpipe.glsl"
#include "rtpipe_hit.glsl"

layout(location = 0) rayPayloadInEXT Payload pl;
layout(shaderRecordEXT, std430) buffer Sbt { uint sbt_data; };
hitAttributeEXT vec2 bary;

void main()
{
   record_hit(pl, K_CHIT1, sbt_data, bary);
}
