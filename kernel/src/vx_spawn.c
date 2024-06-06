// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>
#include <vx_print.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

typedef struct {
	vx_spawn_tasks_cb callback;
	void* arg;
	int all_tasks_offset;
  int remain_tasks_offset;
	int warp_batches;
	int remaining_warps;
} wspawn_tasks_args_t;

static void __attribute__ ((noinline)) process_all_tasks() {
  wspawn_tasks_args_t* targs = (wspawn_tasks_args_t*)csr_read(VX_CSR_MSCRATCH);

  int threads_per_warp = vx_num_threads();
  int warp_id = vx_warp_id();
  int thread_id = vx_thread_id();

  int start_warp = (warp_id * targs->warp_batches) + MIN(warp_id, targs->remaining_warps);
  int iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  int start_task_id = targs->all_tasks_offset + (start_warp * threads_per_warp) + thread_id;
  int end_task_id = start_task_id + iterations * threads_per_warp;

  vx_spawn_tasks_cb callback = targs->callback;
  void* arg = targs->arg;
  for (int task_id = start_task_id; task_id < end_task_id; task_id += threads_per_warp) {
    callback(task_id, arg);
  }
}

static void __attribute__ ((noinline)) process_remaining_tasks() {
  wspawn_tasks_args_t* targs = (wspawn_tasks_args_t*)csr_read(VX_CSR_MSCRATCH);

  int thread_id = vx_thread_id();
  int task_id = targs->remain_tasks_offset + thread_id;

  (targs->callback)(task_id, targs->arg);
}

static void __attribute__ ((noinline)) process_all_tasks_stub() {
  // activate all threads
  vx_tmc(-1);

  // process all tasks
  process_all_tasks();

  // disable warp
  vx_tmc_zero();
}

void vx_spawn_tasks(int num_tasks, vx_spawn_tasks_cb callback , void * arg) {
  // device specifications
  int num_cores = vx_num_cores();
  int warps_per_core = vx_num_warps();
  int threads_per_warp = vx_num_threads();
  int core_id = vx_core_id();

  // calculate necessary active cores
  int threads_per_core = warps_per_core * threads_per_warp;
  int needed_cores = (num_tasks + threads_per_core - 1) / threads_per_core;
  int active_cores = MIN(needed_cores, num_cores);

  // only active cores participate
  if (core_id >= active_cores)
    return;

  // number of tasks per core
  int tasks_per_core = num_tasks / active_cores;
  int remaining_tasks_per_core = num_tasks - tasks_per_core * active_cores;
  if (core_id < remaining_tasks_per_core)
    tasks_per_core++;

  // calculate number of warps to activate
  int total_warps_per_core = tasks_per_core / threads_per_warp;
  int remaining_tasks = tasks_per_core - total_warps_per_core * threads_per_warp;
  int active_warps = total_warps_per_core;
  int warp_batches = 1, remaining_warps = 0;
  if (active_warps > warps_per_core) {
    active_warps = warps_per_core;
    warp_batches = total_warps_per_core / active_warps;
    remaining_warps = total_warps_per_core - warp_batches * active_warps;
  }

  // calculate offsets for task distribution
  int all_tasks_offset = core_id * tasks_per_core + MIN(core_id, remaining_tasks_per_core);
  int remain_tasks_offset = all_tasks_offset + (tasks_per_core - remaining_tasks);

  // prepare scheduler arguments
  wspawn_tasks_args_t wspawn_args = {
    callback,
    arg,
    all_tasks_offset,
    remain_tasks_offset,
    warp_batches,
    remaining_warps
  };
  csr_write(VX_CSR_MSCRATCH, &wspawn_args);

	if (active_warps >= 1) {
    // execute callback on other warps
    vx_wspawn(active_warps, process_all_tasks_stub);

    // activate all threads
    vx_tmc(-1);

    // process all tasks
    process_all_tasks();

    // back to single-threaded
    vx_tmc_one();
	}

  if (remaining_tasks != 0) {
    // activate remaining threads
    int tmask = (1 << remaining_tasks) - 1;
    vx_tmc(tmask);

    // process remaining tasks
    process_remaining_tasks();

    // back to single-threaded
    vx_tmc_one();
  }

  // wait for spawned tasks to complete
  vx_wspawn(1, 0);
}

///////////////////////////////////////////////////////////////////////////////

typedef struct {
	vx_spawn_task_groups_cb callback;
	void* arg;
	int group_offset;
	int warp_batches;
	int remaining_warps;
  int warps_per_group;
  int groups_per_core;
  int remaining_mask;
} wspawn_task_groups_args_t;

static void __attribute__ ((noinline)) process_all_task_groups() {
  wspawn_task_groups_args_t* targs = (wspawn_task_groups_args_t*)csr_read(VX_CSR_MSCRATCH);

  int warps_per_group = targs->warps_per_group;
  int groups_per_core = targs->groups_per_core;

  int threads_per_warp = vx_num_threads();
  int warp_id = vx_warp_id();
  int thread_id = vx_thread_id();

  int iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  int local_group_id = warp_id / warps_per_group;
  int group_warp_id = warp_id - local_group_id * warps_per_group;
  int local_task_id = group_warp_id * threads_per_warp + thread_id;

  int start_group = targs->group_offset + local_group_id;
  int end_group = start_group + iterations * groups_per_core;

  vx_spawn_task_groups_cb callback = targs->callback;
  void* arg = targs->arg;

  for (int group_id = start_group; group_id < end_group; group_id += groups_per_core) {
    callback(local_task_id, group_id, start_group, warps_per_group, arg);
  }
}

static void __attribute__ ((noinline)) process_all_task_groups_stub() {
  wspawn_task_groups_args_t* targs = (wspawn_task_groups_args_t*)csr_read(VX_CSR_MSCRATCH);
  int warps_per_group = targs->warps_per_group;
  int remaining_mask = targs->remaining_mask;
  int warp_id = vx_warp_id();
  int group_warp_id = warp_id % warps_per_group;
  int threads_mask = (group_warp_id == warps_per_group-1) ? remaining_mask : -1;

  // activate threads
  vx_tmc(threads_mask);

  // process all tasks
  process_all_task_groups();

  // disable all warps except warp0
  vx_tmc(0 == vx_warp_id());
}

void vx_syncthreads(int barrier_id) {
  wspawn_task_groups_args_t* targs = (wspawn_task_groups_args_t*)csr_read(VX_CSR_MSCRATCH);
  int warps_per_group = targs->warps_per_group;
  vx_barrier(barrier_id, warps_per_group);
}

void vx_spawn_task_groups(int num_groups, int group_size, vx_spawn_task_groups_cb callback, void * arg) {
  // device specifications
  int num_cores = vx_num_cores();
  int warps_per_core = vx_num_warps();
  int threads_per_warp = vx_num_threads();
  int core_id = vx_core_id();

  // check group size
  int threads_per_core = warps_per_core * threads_per_warp;
  if (threads_per_core < group_size) {
    vx_printf("error: group_size > threads_per_core (%d)\n", threads_per_core);
    return;
  }

  int warps_per_group = group_size / threads_per_warp;
  int remaining_threads = group_size - warps_per_group * threads_per_warp;
  int remaining_mask = -1;
  if (remaining_threads != 0) {
    remaining_mask = (1 << remaining_threads) - 1;
    warps_per_group++;
  }

  int needed_warps = num_groups * warps_per_group;
  int needed_cores = (needed_warps + warps_per_core-1) / warps_per_core;
  int active_cores = MIN(needed_cores, num_cores);

  // only active cores participate
  if (core_id >= active_cores)
    return;

  int total_groups_per_core = num_groups / active_cores;
  int remaining_groups_per_core = num_groups - active_cores * total_groups_per_core;
  if (core_id < remaining_groups_per_core)
    total_groups_per_core++;

  // calculate number of warps to activate
  int groups_per_core = warps_per_core / warps_per_group;
  int total_warps_per_core = total_groups_per_core * warps_per_group;
  int active_warps = total_warps_per_core;
  int warp_batches = 1, remaining_warps = 0;
  if (active_warps > warps_per_core) {
    active_warps = groups_per_core * warps_per_group;
    warp_batches = total_warps_per_core / active_warps;
    remaining_warps = total_warps_per_core - warp_batches * active_warps;
  }

  // calculate offsets for group distribution
  int group_offset = core_id * total_groups_per_core + MIN(core_id, remaining_groups_per_core);

  // prepare scheduler arguments
  wspawn_task_groups_args_t wspawn_args = {
    callback,
    arg,
    group_offset,
    warp_batches,
    remaining_warps,
    warps_per_group,
    groups_per_core,
    remaining_mask
  };
  csr_write(VX_CSR_MSCRATCH, &wspawn_args);

  // execute callback on other warps
  vx_wspawn(active_warps, process_all_task_groups_stub);

  // execute callback on warp0
  process_all_task_groups_stub();

  // wait for spawned tasks to complete
  vx_wspawn(1, 0);
}

#ifdef __cplusplus
}
#endif
