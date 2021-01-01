#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CORES_MAX 16

#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
	pfn_callback callback;
	void * args;
	int offset;
	int N;
	int R;
} wspawn_args_t;

wspawn_args_t* g_wspawn_args[NUM_CORES_MAX];

void spawn_tasks_callback() {  
  vx_tmc(vx_num_threads());

  int core_id = vx_core_id();
  int wid     = vx_warp_id();
  int tid     = vx_thread_id(); 
  int NT      = vx_num_threads();
  
  wspawn_args_t* p_wspawn_args = g_wspawn_args[core_id];

  int wK = (p_wspawn_args->N * wid) + MIN(p_wspawn_args->R, wid);
  int tK = p_wspawn_args->N + (wid < p_wspawn_args->R);
  int offset = p_wspawn_args->offset + (wK * NT) + (tid * tK);

  for (int task_id = offset, N = task_id + tK; task_id < N; ++task_id) {
    (p_wspawn_args->callback)(task_id, p_wspawn_args->args);
  }

  vx_tmc(0 == wid);
}

void spawn_remaining_tasks_callback(int nthreads) {    
  vx_tmc(nthreads);

  int core_id = vx_core_id(); 
  int tid = vx_thread_gid();

  wspawn_args_t* p_wspawn_args = g_wspawn_args[core_id];

  int task_id = p_wspawn_args->offset + tid;
  (p_wspawn_args->callback)(task_id, p_wspawn_args->args);

  vx_tmc(1);
}

void vx_spawn_tasks(int num_tasks, pfn_callback callback , void * args) {
	// device specs
  int NC = vx_num_cores();
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();  
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = NW * NT;
  int nC = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate unused cores

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core0 = tasks_per_core;  
  if (core_id == (NC-1)) {    
    int QC_r = num_tasks - (nc * tasks_per_core0); 
    tasks_per_core0 += QC_r; // last core executes remaining tasks
  }

  // number of tasks per warp
  int nW = tasks_per_core0 / NT;        		// total warps per core
  int rT = tasks_per_core0 - (nW * NT); 		// remaining threads
  int fW  = (nW >= NW) ? (nW / NW) : 0;			// full warps iterations
  int rW  = (fW != 0) ? (nW - fW * NW) : 0; // remaining warps
  if (0 == fW)
    fW = 1;

  //--
  wspawn_args_t wspawn_args = { callback, args, core_id * tasks_per_core, fW, rW };
  g_wspawn_args[core_id] = &wspawn_args;

  //--
	if (nW > 1)	{ 
    int nw = MIN(nW, NW);    
	  vx_wspawn(nw, (unsigned)&spawn_tasks_callback);
    spawn_tasks_callback();
	}  

  //--    
  if (rT != 0) {
    wspawn_args.offset = tasks_per_core0 - rT;
    spawn_remaining_tasks_callback(rT);
  }
}

#ifdef __cplusplus
}
#endif