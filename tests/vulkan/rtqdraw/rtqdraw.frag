/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The ray query of tests/vulkan/rtquery, moved from a compute shader into a
 * fragment shader.
 *
 * An acceleration structure reaches a shader as a descriptor, and a draw
 * relocates a descriptor through a different loop than a dispatch does. The
 * compute loop handles all three descriptor kinds; the draw loop handles
 * buffers and images. Every ray-query test in this suite is compute, so the
 * draw path's handling of an acceleration structure has never been asked for
 * an answer.
 *
 * One orthographic ray per fragment, down -Z at the same one-triangle scene:
 * red on a hit, black on a miss. */
#version 460
#extension GL_EXT_ray_query : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(location = 0) in  vec2 v_ndc;
layout(location = 0) out vec4 o_color;

void main()
{
   vec3 origin = vec3(v_ndc, 1.0);
   vec3 dir    = vec3(0.0, 0.0, -1.0);

   rayQueryEXT rq;
   rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFFu,
                         origin, 0.0, dir, 2.0);
   while (rayQueryProceedEXT(rq)) { }

   bool hit = rayQueryGetIntersectionTypeEXT(rq, true) ==
              gl_RayQueryCommittedIntersectionTriangleEXT;

   o_color = hit ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
