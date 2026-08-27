/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Subgroup coverage for the VERTEX stage.
 *
 * The vertex stage reports what it observes about its own subgroup and the
 * fragment stage only forwards it. gl_SubgroupSize must be the device warp
 * width, not the width the host rasterizer finalized this shader at -- those
 * differ (4 against 8), and a vertex shader lowered at the host's width reports
 * the wrong one here as a wrong colour rather than as a crash.
 *
 * subgroupBroadcastFirst is included so a constant-folded intrinsic cannot pass
 * for an executed one: under MESA_VORTEX_STRICT=1 an unlowered intrinsic bails
 * the shader, the draw leaves the clear colour in place, and the host's
 * clear-colour assertion catches it.
 */
#version 450
#extension GL_KHR_shader_subgroup_basic  : require
#extension GL_KHR_shader_subgroup_ballot : require

/* flat: the provoking vertex's value must reach the fragment unchanged.
 * Interpolating would average lane ids across the triangle and destroy the
 * signal this test reads. */
layout(location = 0) flat out uvec4 v_sg;

/* One triangle covering the whole target, so any sampled pixel is fully
 * covered and carries a vertex's value rather than a partial-coverage blend. */
vec2 corners[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));

void main()
{
   gl_Position = vec4(corners[gl_VertexIndex], 0.5, 1.0);
   /* The lane byte carries a second, non-subgroup value in its high nibble:
    * gl_VertexIndex + 1, which owes nothing to subgroup lowering. Without it
    * every component of this varying is a subgroup op, and an all-zero report
    * cannot say whether the intrinsics returned zero or the varying never
    * arrived -- two very different defects. A lane id is under 16 and the
    * vertex count is 3, so both fit one byte with no loss. */
   v_sg = uvec4(gl_SubgroupSize,
                gl_SubgroupInvocationID + (gl_VertexIndex + 1u) * 16u,
                subgroupBroadcastFirst(gl_VertexIndex + 1u),
                subgroupBallot(true).x);
}
