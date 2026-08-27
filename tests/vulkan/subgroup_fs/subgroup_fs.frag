/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Subgroup coverage for the FRAGMENT stage.
 *
 * gl_SubgroupSize must be the device warp width, not the width the host
 * rasterizer finalized this shader at -- those differ (4 against 8). Subgroup
 * lowering constant-folds the value into the shader, so a fragment shader that
 * was only ever lowered for llvmpipe reports llvmpipe's width and there is no
 * later opportunity to correct it.
 *
 * subgroupAny is included so a constant-folded intrinsic cannot pass for an
 * executed one: it forces the lowering to run, and its value is deterministic
 * (some lane is always active in a shaded quad) so it does not make the test
 * depend on lane packing.
 *
 * The quarter-step bias is the same one subgroup_fs's vertex twin needs: the
 * device truncates its UNORM8 conversion where the host rounds, so an unbiased
 * byte encoding comes back one short. */
#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote  : require

layout(location = 0) out vec4 o_color;

void main()
{
   o_color = (vec4(float(gl_SubgroupSize),
                   subgroupAny(true) ? 100.0 : 0.0,
                   200.0,
                   255.0) + 0.25) / 255.0;
}
