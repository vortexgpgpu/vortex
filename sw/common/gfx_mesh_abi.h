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

// Task and mesh stages: the contract between the host that builds a mesh draw
// (vortexpipe), the stage kernels that run its workgroups (gfx_mesh_k.h, linked
// with each translated task or mesh shader), and the front-end kernels that
// order what they produce (mesh_post_k, scan_k).
//
// A draw's task workgroups run as one persistent grid that walks them in order,
// each writing its payload and the mesh grid it launches; a device-wide scan of
// the mesh-workgroup counts gives every task its first mesh workgroup. The mesh
// workgroups then run, also as a persistent grid, over windows of consecutive
// workgroups sized to a memory budget. Workgroup w of a window owns fixed slots:
// vmax vertex records, pmax primitive records and pmax list entries, written in
// place by the shader. A workgroup whose primitive count is static writes its
// entries straight into the dense list setup reads; otherwise its live
// primitives are compacted inside the workgroup, scanned across the window and
// copied into the dense list in order by mesh_post_k. Every order is (task, mesh
// workgroup, primitive index), the one the specification gives.

#pragma once

#include <stdint.h>

// The shader entries the stage kernels call once per workgroup.
#define GFX_TASK_ENTRY "vp_task_main"
#define GFX_MESH_ENTRY "vp_mesh_main"

// What one task or mesh workgroup's shader body is handed. Addresses are device
// byte addresses.
typedef struct {
  uint64_t desc;           // the stage's descriptor table (constant-buffer bases)
  uint64_t payload;        // the task's payload: written by a task, read by its meshes
  uint64_t vrec;           // mesh: vertex 0 of the workgroup's vertex records
  uint64_t prec;           // mesh: primitive 0 of its primitive records
  uint64_t prims;          // mesh: primitive 0 of its list entries (pipe_list_entry_t)
  uint64_t counts;         // mesh: uint32[2] {vertices, primitives} of SetMeshOutputsEXT
  uint64_t grid;           // task: uint32[3] mesh grid of EmitMeshTasksEXT
  uint64_t count;          // task: uint32 mesh workgroups it launches (0 if any is 0)
  uint32_t wg_id[3];       // gl_WorkGroupID
  uint32_t num_wg[3];      // gl_NumWorkGroups
  uint32_t vrec_index;     // mesh: index of vrec among the window's vertex records
  uint32_t draw_id;        // gl_DrawID
  uint32_t view;           // gl_ViewIndex
  uint32_t pad;
} gfx_mesh_wg_t;

// A draw's (or a chunk of its tasks') workgroup counts on the device.
typedef struct {
  uint32_t tasks;          // task workgroups of the chunk
  uint32_t meshes;         // its mesh workgroups: the scan of the tasks' counts writes it
  uint32_t grid[3];        // the draw's workgroup grid: of tasks, or of meshes without them
  uint32_t pad[3];
} gfx_mesh_meta_t;

// task_k: one persistent grid over the chunk's task workgroups.
typedef struct {
  uint64_t desc;           // the stage's descriptor table
  uint64_t meta;           // gfx_mesh_meta_t
  uint64_t payload;        // payload_stride bytes per task of the chunk
  uint64_t grid;           // uint32[4] per task: its mesh grid
  uint64_t count;          // uint32 per task: its mesh workgroups
  uint64_t query;          // the query counter block (GFX_QC_*), 0 = none
  uint32_t first;          // the chunk's first task workgroup in the draw
  uint32_t payload_stride;
  uint32_t flags;          // GFX_MESH_F_SHARED
  uint32_t draw_id;
  uint32_t view;
  uint32_t block;          // invocations per workgroup
} gfx_task_arg_t;

// mesh_k: one persistent grid over a window of mesh workgroups.
typedef struct {
  uint64_t desc;           // the stage's descriptor table
  uint64_t meta;           // gfx_mesh_meta_t
  uint64_t payload;        // the tasks' payloads, 0 without a task stage
  uint64_t task_grid;      // uint32[4] per task: its mesh grid
  uint64_t task_base;      // uint32 per task: its first mesh workgroup
  uint64_t vrec;           // the window's vertex records, vmax per workgroup
  uint64_t prec;           // the window's primitive records, pmax per workgroup
  // The window's list entries. Static counts: the dense list, static_prims per
  // workgroup. Otherwise pmax staged entries per workgroup, raw vertex indices
  // and the cull flag in aux, which the workgroup compacts into `live`.
  uint64_t prims;
  uint64_t live;           // pipe_list_entry_t[pmax] per workgroup, compacted
  uint64_t pcnt;           // uint32 per workgroup: its live primitives
  uint64_t query;          // the query counter block (GFX_QC_*), 0 = none
  uint32_t first;          // the window: mesh workgroups [first, first + count)
  uint32_t count;
  uint32_t vmax, pmax;     // record slots per workgroup
  uint32_t vstride;        // bytes per vertex record
  uint32_t pstride;        // bytes per primitive record, 0 = none
  uint32_t prim_verts;     // vertices per primitive: 1, 2 or 3
  uint32_t static_prims;   // with GFX_MESH_F_STATIC: primitives every workgroup emits
  uint32_t fanout;         // mesh workgroups every task launches, 0 = from task_base
  uint32_t fan_grid[3];    // the grid those make
  uint32_t flags;          // GFX_MESH_F_*
  uint32_t lmem;           // the kernel's scratch in local memory, past the shader's
  uint32_t payload_stride;
  uint32_t block;          // invocations per workgroup
  uint32_t draw_id;
  uint32_t view;
} gfx_mesh_arg_t;

#define GFX_MESH_F_TASK    (1u << 0)  // workgroups come from a task stage
#define GFX_MESH_F_SHARED  (1u << 1)  // the shader uses workgroup memory
#define GFX_MESH_F_CULL    (1u << 2)  // the shader writes CullPrimitiveEXT
#define GFX_MESH_F_PRIMREC (1u << 3)  // entries name their primitive record in aux
#define GFX_MESH_F_STATIC  (1u << 4)  // every workgroup emits static_prims primitives

// The most mesh workgroups one task workgroup launches
// (maxMeshWorkGroupTotalCount); a larger grid is clamped to it.
#define GFX_MESH_MAX_WORKGROUPS 4194304u

// Bytes of local memory mesh_k needs past the shader's for a workgroup of
// `block` invocations: the two counts, then the scan of the live primitives.
#define GFX_MESH_LMEM_BYTES(block) (16u + 8u * (block))

// A window of a task chunk whose mesh workgroups only the device counts, as
// mesh_launch_k finds it from the chunk's scan: the persistent grid its
// workgroups take (0 past the chunk's last workgroup, which skips the window),
// and with static counts the primitives they emit.
typedef struct {
  uint32_t grid;
  uint32_t prims;
  uint32_t count;          // the window's workgroups: what its scan and post pass read
  uint32_t pad;
} gfx_mesh_window_t;

// mesh_launch_k: one thread turns a chunk's mesh-workgroup count into the
// words its recorded windows launch from. For an indirect mesh draw without a
// task stage it first takes the count from the draw's command: the grid of
// `record`, none when `draw_index` is past the count buffer's word; it writes
// the grid and the count to `meta` and sets `overflow` when the windows hold
// fewer workgroups than the grid has.
typedef struct {
  uint64_t meta;           // gfx_mesh_meta_t: `meshes`, which the chunk's scan wrote
  uint64_t windows;        // gfx_mesh_window_t[nwin]
  uint64_t record;         // the indirect command's uint32[3] grid, 0 = from meta
  uint64_t draw_count;     // the count buffer's uint32, 0 = none
  uint64_t overflow;       // uint32, with `record`
  uint32_t draw_index;     // the command's index, with `draw_count`
  uint32_t nwin;           // windows recorded for the chunk
  uint32_t window;         // mesh workgroups per window
  uint32_t grid_max;       // the most CTAs one window's persistent grid takes
  uint32_t static_prims;   // primitives each workgroup emits, 0 = not static
  uint32_t pad;
} gfx_mesh_launch_arg_t;

// mesh_post_k: the window's live primitives into the dense list, in order. One
// group of `lanes` threads per workgroup copies its entries.
typedef struct {
  uint64_t live;           // pipe_list_entry_t[pmax] per workgroup
  uint64_t pcnt;           // uint32 per workgroup
  uint64_t pbase;          // uint32 per workgroup: the exclusive scan of pcnt
  uint64_t list;           // the dense list
  uint64_t count_addr;     // uint32: the workgroups a device-sized window holds, 0 = count
  uint32_t count;          // workgroups in the window, or a bound on them
  uint32_t pmax;
  uint32_t lanes;
  uint32_t pad;
} gfx_mesh_post_arg_t;
