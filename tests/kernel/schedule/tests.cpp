#include "tests.h"
#include <algorithm>
#include <stdio.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#define LOCAL_X (32)
#define LOCAL_Y (1)
#define LOCAL_Z (1)
#define LOCAL_XYZ (LOCAL_X * LOCAL_Y * LOCAL_Z)

#define GROUP_X (1)
#define GROUP_Y (1)
#define GROUP_Z (1)
#define GROUP_XYZ (GROUP_X * GROUP_Y * GROUP_Z)

#define BUF_SIZE (LOCAL_XYZ * GROUP_XYZ)

int BUF[BUF_SIZE];

void li_kernel(int* A, context_t* ctx, int gid_x, int gid_y, int gid_z)
{
  int num_local_x = ctx->local_size[0];
  int num_local_y = ctx->local_size[1];
  int num_local_z = ctx->local_size[2];
  int num_local_xy = num_local_x * num_local_y;
  int workload = num_local_xy * num_local_z;

  int num_groups_x = ctx->num_groups[0];
  int num_groups_y = ctx->num_groups[1];
  int num_groups_xy = num_groups_x * num_groups_y;
  int group_offset = (gid_x + gid_y * num_groups_x + gid_z * num_groups_xy)
      * workload;

  int wid = vx_warp_id();
  int tlid = vx_thread_id() + wid * vx_num_warps();

  int TpC = vx_num_threads() * vx_num_warps();
  int nHT = vx_num_threads();

  int remains = workload % nHT;
  int loopWorks = workload - remains;

  int lid = tlid;

  for (; lid < workload; lid += TpC) {
    if (lid >= loopWorks) {
      int thread_mask = ((1 << remains) - 1);
      vx_tmc(thread_mask);
    }

    int z = lid / num_local_xy;
    int remains_xy = (lid - z * num_local_xy);
    int y = remains_xy / num_local_x;
    int x = remains_xy - y * num_local_x;
    int tid = group_offset + lid;
    vx_printf("check tid %d, htid %d, wid %d, point(%d, %d, %d) %d %d\n", tid, tlid, wid, x, y, z, loopWorks, remains);
    A[tid] = tid;

    if (lid >= loopWorks) {
      vx_tmc(-1);
    }
  }
}

int test_thread_kernel()
{
  vx_printf("BUF_SIZE %d\n", BUF_SIZE);
  // Make context
  context_t ctx;
  ctx.num_groups[0] = GROUP_X;
  ctx.num_groups[1] = GROUP_Y;
  ctx.num_groups[2] = GROUP_Z;
  ctx.global_offset[0] = 0;
  ctx.global_offset[1] = 0;
  ctx.global_offset[2] = 0;
  ctx.local_size[0] = LOCAL_X;
  ctx.local_size[1] = LOCAL_Y;
  ctx.local_size[2] = LOCAL_Z;
  ctx.work_dim = 3;

  vx_printf("Make context\n");

  vx_spawn_kernel_cm(&ctx, (vx_spawn_kernel_cb)li_kernel, (void*)BUF);
}

/*

void li_kernel(int* A, context_t* ctx, int gid_x, int gid_y, int gid_z)
{
  int num_local_x = ctx->local_size[0];
  int num_local_y = ctx->local_size[1];
  int num_local_z = ctx->local_size[2];
  int num_local_xy = num_local_x * num_local_y;
  int workload = num_local_xy * num_local_z;

  int num_groups_x = ctx->num_groups[0];
  int num_groups_y = ctx->num_groups[1];
  int num_groups_xy = num_groups_x * num_groups_y;
  int group_offset = (gid_x + gid_y * num_groups_x + gid_z * num_groups_xy)
      * workload;

  int wid = vx_warp_id();
  int tlid = vx_thread_lid();
  int TpC = vx_num_threads() * vx_num_warps();

  int nHT = vx_num_threads();
  //int tid = vx_thread_id();
  int mask = vx_thread_mask();
  int lid = tlid;
  if (mask == ((1 << nHT) - 1)) {
    workload = workload - workload % nHT;
  } else {
    lid = workload - workload % nHT + tlid;
  }

  for (; lid < workload; lid += TpC) {
    int z = lid / num_local_xy;
    int remains_xy = (lid - z * num_local_xy);
    int y = remains_xy / num_local_x;
    int x = remains_xy - y * num_local_x;
    int tid = group_offset + lid;
    vx_printf("check tid %d, htid %d, wid %d, point(%d, %d, %d) %d\n", tid, tlid, wid, x, y, z, workload);
    A[tid] = tid;
  }
}
*/

/*

void li_kernel(int* A, context_t* ctx, int gid_x, int gid_y, int gid_z)
{
  int num_local_x = ctx->local_size[0];
  int num_local_y = ctx->local_size[1];
  int num_local_z = ctx->local_size[2];
  int num_local_xy = num_local_x * num_local_y;
  int workload = num_local_xy * num_local_z;

  int num_groups_x = ctx->num_groups[0];
  int num_groups_y = ctx->num_groups[1];
  int num_groups_xy = num_groups_x * num_groups_y;
  int group_offset = (gid_x + gid_y * num_groups_x + gid_z * num_groups_xy)
      * workload;

  int wid = vx_warp_id();
  int tlid = vx_thread_lid();
  int TpC = vx_num_threads() * vx_num_warps();

  int nHT = vx_num_threads();

  int remains = workload % nHT;
  int loopWorks = workload - remains;

  int lid = tlid;

  for (; lid < loopWorks; lid += TpC) {
    int z = lid / num_local_xy;
    int remains_xy = (lid - z * num_local_xy);
    int y = remains_xy / num_local_x;
    int x = remains_xy - y * num_local_x;
    int tid = group_offset + lid;
    vx_printf("check tid %d, htid %d, wid %d, point(%d, %d, %d) %d %d\n", tid, tlid, wid, x, y, z, loopWorks, remains);
    A[tid] = tid;
  }

  if (lid < workload) {
    int thread_mask = ((1 << remains) - 1);

    vx_tmc(thread_mask);

    int z = lid / num_local_xy;
    int remains_xy = (lid - z * num_local_xy);
    int y = remains_xy / num_local_x;
    int x = remains_xy - y * num_local_x;
    int tid = group_offset + lid;
    vx_printf("check tid %d, htid %d, wid %d, point(%d, %d, %d) %d %d\n", tid, tlid, wid, x, y, z, loopWorks, remains);
    A[tid] = tid;

    vx_tmc(-1);
  }
}
*/
