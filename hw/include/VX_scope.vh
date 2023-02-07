`ifndef VX_SCOPE
`define VX_SCOPE

`ifdef SCOPE

`include "scope-defs.vh"

`define SCOPE_ASSIGN(d,s) assign scope_``d = s

`define SCOPE_SIZE 1024

`else

`define SCOPE_IO_VX_icache_stage

`define SCOPE_IO_VX_fetch

`define SCOPE_BIND_VX_fetch_icache_stage

`define SCOPE_BIND_VX_fetch_warp_sched

`define SCOPE_IO_VX_warp_sched

`define SCOPE_IO_VX_pipeline

`define SCOPE_BIND_VX_pipeline_fetch

`define SCOPE_IO_VX_core

`define SCOPE_BIND_VX_core_pipeline

`define SCOPE_IO_VX_cluster

`define SCOPE_BIND_VX_cluster_core(__i__)

`define SCOPE_IO_Vortex

`define SCOPE_BIND_Vortex_cluster(__i__)

`define SCOPE_BIND_afu_vortex

`define SCOPE_IO_VX_lsu_unit

`define SCOPE_IO_VX_gpu_unit

`define SCOPE_IO_VX_execute

`define SCOPE_BIND_VX_execute_lsu_unit

`define SCOPE_BIND_VX_execute_gpu_unit

`define SCOPE_BIND_VX_pipeline_execute

`define SCOPE_IO_VX_issue

`define SCOPE_BIND_VX_pipeline_issue

`define SCOPE_IO_VX_bank

`define SCOPE_IO_VX_cache

`define SCOPE_BIND_VX_cache_bank(__i__)

`define SCOPE_BIND_Vortex_l3cache

`define SCOPE_BIND_VX_cluster_l2cache

`define SCOPE_IO_VX_mem_unit

`define SCOPE_BIND_VX_mem_unit_dcache

`define SCOPE_BIND_VX_core_mem_unit

`define SCOPE_BIND_VX_mem_unit_icache

`define SCOPE_BIND_VX_mem_unit_smem

`define SCOPE_DECL_SIGNALS

`define SCOPE_DATA_LIST

`define SCOPE_UPDATE_LIST

`define SCOPE_TRIGGER

`define SCOPE_ASSIGN(d,s)

`endif
`endif