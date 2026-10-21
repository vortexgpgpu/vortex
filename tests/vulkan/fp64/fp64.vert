#version 450
#extension GL_ARB_gpu_shader_int64 : require
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Doubles through the vertex stage: a double read from a storage buffer per
 * vertex, double arithmetic in the shader, flat double, dvec3 and dvec4
 * varyings to the fragment stage, and a double stored from the vertex shader itself. (No
 * 64-bit vertex format is offered, so the value cannot be a vertex attribute.)
 */

layout(location = 0) in vec2 pos;

layout(std430, binding = 1) readonly buffer In {
   double val[6];
} inv;

layout(location = 0) flat out double v_d;
layout(location = 2) flat out uint v_tri;
/* Three and four doubles span two locations each, so their halves cross a
 * location boundary in the varying record. */
layout(location = 3) flat out dvec3 v_d3;
layout(location = 5) flat out dvec4 v_d4;

/* Per triangle: two computed values, then the dvec3 and dvec4 components. */
#define NFS 9u

layout(std430, binding = 0) buffer Out {
   uint64_t vs_r[6];
   uint64_t fs_r[NFS * 2u];
} outr;

void main()
{
   double val = inv.val[gl_VertexIndex];
   gl_Position = vec4(pos, 0.0, 1.0);
   v_d = val * 3.0lf + 1.0lf / 3.0lf;
   v_tri = uint(gl_VertexIndex) / 3u;
   v_d3 = dvec3(val * 2.0lf, val - 5.0lf, val / 3.0lf);
   v_d4 = dvec4(val + 1.0lf, -val, val * 0.5lf, val * val);
   outr.vs_r[gl_VertexIndex] = doubleBitsToUint64(val / 7.0lf + sqrt(abs(val)));
}
