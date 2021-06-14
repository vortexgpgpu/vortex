#include <iostream>
#include <assert.h>

#define NUM_CORES_MAX 32

#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  char * printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

typedef void (*vx_pocl_workgroup_func) (
  const void * /* args */,
	const struct context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef struct {
  struct context_t * ctx;
  vx_pocl_workgroup_func pfn;
  const void * args;
  int offset; 
  int N;
  int R;
} wspawn_args_t;

void kernel_spawn_callback(int core_id, int NW, int NT, int nW, wspawn_args_t* p_wspawn_args) {
  assert(nW <= NW);
  for (int wid = 0; wid < nW; ++wid) {
    for (int tid = 0; tid < NT; ++tid) {
      int wK = (p_wspawn_args->N * wid) + MIN(p_wspawn_args->R, wid);
      int tK = p_wspawn_args->N + (wid < p_wspawn_args->R);
      int offset = p_wspawn_args->offset + (wK * NT) + (tid * tK);

      int X = p_wspawn_args->ctx->num_groups[0];
      int Y = p_wspawn_args->ctx->num_groups[1];
      int XY = X * Y;

      for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {    
        int k = wg_id / XY;
        int wg_2d = wg_id - k * XY;
        int j = wg_2d / X;
        int i = wg_2d - j * X;

        int gid0 = p_wspawn_args->ctx->global_offset[0] + i;
        int gid1 = p_wspawn_args->ctx->global_offset[1] + j;
        int gid2 = p_wspawn_args->ctx->global_offset[2] + k;

        printf("c%d w%d t%d: g={%d, %d, %d}\n", core_id, wid, tid, gid0, gid1, gid2);
      }
    }
  }
}

void kernel_spawn_remaining_callback(int core_id, int NW, int NT, int wid, int nT, wspawn_args_t* p_wspawn_args) {    
  assert(wid < NW);
  assert(nT <= NT);
  for (int t = 0; t < nT; ++t) {
    int tid = core_id * NW * NT + wid * NT + t;

    int wg_id = p_wspawn_args->offset + tid;

    int X = p_wspawn_args->ctx->num_groups[0];
    int Y = p_wspawn_args->ctx->num_groups[1];
    int XY = X * Y;
    
    int k = wg_id / XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d / X;
    int i = wg_2d - j * X;

    int gid0 = p_wspawn_args->ctx->global_offset[0] + i;
    int gid1 = p_wspawn_args->ctx->global_offset[1] + j;
    int gid2 = p_wspawn_args->ctx->global_offset[2] + k;

    printf("c%d w%d t%d: g={%d, %d, %d}\n", core_id, wid, tid, gid0, gid1, gid2);
  }
}

void kernel_run_once(context_t* ctx, int NC, int NW, int NT, int core_id) {
    // total number of WGs
    int X = ctx->num_groups[0];
    int Y = ctx->num_groups[1];
    int Z = ctx->num_groups[2];
    int Q = X * Y * Z;

    // current core id
    if (core_id >= NUM_CORES_MAX)
      return;

    // calculate necessary active cores
    int WT = NW * NT;
    int nC = (Q > WT) ? (Q / WT) : 1;
    int nc = MIN(nC, NC);
    if (core_id >= nc)
      return; // terminate extra cores

    // number of workgroups per core
    int wgs_per_core = Q / nc;
    int wgs_per_core0 = wgs_per_core;  
    if (core_id == (NC-1)) {    
      int QC_r = Q - (nc * wgs_per_core0); 
      wgs_per_core0 += QC_r; // last core executes remaining WGs
    }

    // number of workgroups per warp
    int nW = wgs_per_core0 / NT;              // total warps per core
    int rT = wgs_per_core0 - (nW * NT);       // remaining threads
    int fW = (nW >= NW) ? (nW / NW) : 0;      // full warps iterations
    int rW = (fW != 0) ? (nW - fW * NW) : 0;  // reamining full warps
    if (0 == fW)
      fW = 1;

    //--
    wspawn_args_t wspawn_args = { ctx, NULL, NULL, core_id * wgs_per_core, fW, rW };

    //--
    if (nW >= 1)	{ 
      int nw = MIN(nW, NW);
      kernel_spawn_callback(core_id, NW, NT, nw, &wspawn_args);
    }  

    //--    
    if (rT != 0) {
      wspawn_args.offset = wgs_per_core0 - rT;
      kernel_spawn_remaining_callback(core_id, NW, NT, 0, rT, &wspawn_args);
    }
  }

  void kernel_run(int X, int Y, int Z, int NC, int NW, int NT) {
    context_t ctx;

    ctx.num_groups[0] = X;
    ctx.num_groups[1] = Y;
    ctx.num_groups[2] = Z;
    ctx.global_offset[0] = 0;
    ctx.global_offset[1] = 0;
    ctx.global_offset[2] = 0;

    for (int cid = 0; cid < NC; ++cid) {
      kernel_run_once(&ctx, NC, NW, NT, cid);
    }

    exit (0);
  }