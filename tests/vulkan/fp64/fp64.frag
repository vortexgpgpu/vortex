#version 450
#extension GL_ARB_gpu_shader_int64 : require
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Flat double, dvec3 and dvec4 varyings: the scalar computed on in double
 * precision, the vectors stored as they arrive. Every fragment of one triangle
 * stores the same bits, so the store order does not matter.
 */

layout(location = 0) flat in double v_d;
layout(location = 2) flat in uint v_tri;
layout(location = 3) flat in dvec3 v_d3;
layout(location = 5) flat in dvec4 v_d4;

layout(location = 0) out vec4 color;

/* Per triangle: two computed values, then the dvec3 and dvec4 components. */
#define NFS 9u

layout(std430, binding = 0) buffer Out {
   uint64_t vs_r[6];
   uint64_t fs_r[NFS * 2u];
} outr;

void main()
{
   const uint o = v_tri * NFS;
   outr.fs_r[o + 0u] = doubleBitsToUint64(v_d * v_d - 2.0lf);
   outr.fs_r[o + 1u] = doubleBitsToUint64(v_d / 9.0lf);
   for (uint c = 0u; c < 3u; c++)
      outr.fs_r[o + 2u + c] = doubleBitsToUint64(v_d3[c]);
   for (uint c = 0u; c < 4u; c++)
      outr.fs_r[o + 5u + c] = doubleBitsToUint64(v_d4[c]);
   color = vec4(0.0, float(v_tri) * 0.5 + 0.25, 0.0, 1.0);
}
