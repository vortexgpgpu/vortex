/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The record every closest-hit shader writes. */

void record_hit(inout Payload p, uint kind, uint sbt_data, vec2 attr)
{
   p.w[P_KIND]    = kind;
   p.w[P_HITKIND] = gl_HitKindEXT;
   p.w[P_T]       = floatBitsToUint(gl_HitTEXT);
   p.w[P_PRIM]    = gl_PrimitiveID;
   p.w[P_INST]    = gl_InstanceID;
   p.w[P_CUSTOM]  = gl_InstanceCustomIndexEXT;
   p.w[P_GEOM]    = gl_GeometryIndexEXT;
   p.w[P_ATTR_U]  = floatBitsToUint(attr.x);
   p.w[P_ATTR_V]  = floatBitsToUint(attr.y);
   p.w[P_RECORD]  = sbt_data;
   p.w[P_OBJ_OX]  = floatBitsToUint(gl_ObjectRayOriginEXT.x);
   p.w[P_O2W_TX]  = floatBitsToUint(gl_ObjectToWorldEXT[3].x);
   p.w[P_W2O_XX]  = floatBitsToUint(gl_WorldToObjectEXT[0].x);
   p.w[P_TMIN]    = floatBitsToUint(gl_RayTminEXT);
   p.w[P_TMAX]    = floatBitsToUint(gl_RayTmaxEXT);
   p.w[P_FLAGS]   = gl_IncomingRayFlagsEXT;
   p.w[P_WDIR_Z]  = floatBitsToUint(gl_WorldRayDirectionEXT.z);
}
