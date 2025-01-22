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

`ifndef VX_TYPES_VH
`define VX_TYPES_VH

// Device configuration registers /////////////////////////////////////////////

`define VX_CSR_ADDR_BITS                12
`define VX_DCR_ADDR_BITS                12

`define VX_DCR_BASE_STATE_BEGIN         12'h001
`define VX_DCR_BASE_STARTUP_ADDR0       12'h001
`define VX_DCR_BASE_STARTUP_ADDR1       12'h002
`define VX_DCR_BASE_STARTUP_ARG0        12'h003
`define VX_DCR_BASE_STARTUP_ARG1        12'h004
`define VX_DCR_BASE_MPM_CLASS           12'h005
`define VX_DCR_BASE_STATE_END           12'h006

`define VX_DCR_BASE_STATE(addr)         ((addr) - `VX_DCR_BASE_STATE_BEGIN)
`define VX_DCR_BASE_STATE_COUNT         (`VX_DCR_BASE_STATE_END-`VX_DCR_BASE_STATE_BEGIN)

// Machine Performance-monitoring counters classes ////////////////////////////

`define VX_DCR_MPM_CLASS_NONE           0
`define VX_DCR_MPM_CLASS_CORE           1
`define VX_DCR_MPM_CLASS_MEM            2

// User Floating-Point CSRs ///////////////////////////////////////////////////

`define VX_CSR_FFLAGS                   12'h001
`define VX_CSR_FRM                      12'h002
`define VX_CSR_FCSR                     12'h003

`define VX_CSR_SATP                     12'h180

`define VX_CSR_PMPCFG0                  12'h3A0
`define VX_CSR_PMPADDR0                 12'h3B0

`define VX_CSR_MSTATUS                  12'h300
`define VX_CSR_MISA                     12'h301
`define VX_CSR_MEDELEG                  12'h302
`define VX_CSR_MIDELEG                  12'h303
`define VX_CSR_MIE                      12'h304
`define VX_CSR_MTVEC                    12'h305

`define VX_CSR_MSCRATCH                 12'h340
`define VX_CSR_MEPC                     12'h341
`define VX_CSR_MCAUSE                   12'h342

`define VX_CSR_MNSTATUS                 12'h744

`define VX_CSR_MPM_BASE                 12'hB00
`define VX_CSR_MPM_BASE_H               12'hB80
`define VX_CSR_MPM_USER                 12'hB03
`define VX_CSR_MPM_USER_H               12'hB83

// Machine Performance-monitoring core counters (Standard) ////////////////////

`define VX_CSR_MCYCLE                   12'hB00
`define VX_CSR_MCYCLE_H                 12'hB80
`define VX_CSR_MPM_RESERVED             12'hB01
`define VX_CSR_MPM_RESERVED_H           12'hB81
`define VX_CSR_MINSTRET                 12'hB02
`define VX_CSR_MINSTRET_H               12'hB82

// Machine Performance-monitoring core counters (class 1) /////////////////////

// PERF: pipeline
`define VX_CSR_MPM_SCHED_ID             12'hB03
`define VX_CSR_MPM_SCHED_ID_H           12'hB83
`define VX_CSR_MPM_SCHED_ST             12'hB04
`define VX_CSR_MPM_SCHED_ST_H           12'hB84
`define VX_CSR_MPM_IBUF_ST              12'hB05
`define VX_CSR_MPM_IBUF_ST_H            12'hB85
`define VX_CSR_MPM_SCRB_ST              12'hB06
`define VX_CSR_MPM_SCRB_ST_H            12'hB86
`define VX_CSR_MPM_OPDS_ST              12'hB07
`define VX_CSR_MPM_OPDS_ST_H            12'hB87
`define VX_CSR_MPM_SCRB_ALU             12'hB08
`define VX_CSR_MPM_SCRB_ALU_H           12'hB88
`define VX_CSR_MPM_SCRB_FPU             12'hB09
`define VX_CSR_MPM_SCRB_FPU_H           12'hB89
`define VX_CSR_MPM_SCRB_LSU             12'hB0A
`define VX_CSR_MPM_SCRB_LSU_H           12'hB8A
`define VX_CSR_MPM_SCRB_SFU             12'hB0B
`define VX_CSR_MPM_SCRB_SFU_H           12'hB8B
`define VX_CSR_MPM_SCRB_CSRS            12'hB0C
`define VX_CSR_MPM_SCRB_CSRS_H          12'hB8C
`define VX_CSR_MPM_SCRB_WCTL            12'hB0D
`define VX_CSR_MPM_SCRB_WCTL_H          12'hB8D
// PERF: memory
`define VX_CSR_MPM_IFETCHES             12'hB0E
`define VX_CSR_MPM_IFETCHES_H           12'hB8E
`define VX_CSR_MPM_LOADS                12'hB0F
`define VX_CSR_MPM_LOADS_H              12'hB8F
`define VX_CSR_MPM_STORES               12'hB10
`define VX_CSR_MPM_STORES_H             12'hB90
`define VX_CSR_MPM_IFETCH_LT            12'hB11
`define VX_CSR_MPM_IFETCH_LT_H          12'hB91
`define VX_CSR_MPM_LOAD_LT              12'hB12
`define VX_CSR_MPM_LOAD_LT_H            12'hB92

// Machine Performance-monitoring memory counters (class 2) ///////////////////

// PERF: icache
`define VX_CSR_MPM_ICACHE_READS         12'hB03     // total reads
`define VX_CSR_MPM_ICACHE_READS_H       12'hB83
`define VX_CSR_MPM_ICACHE_MISS_R        12'hB04     // read misses
`define VX_CSR_MPM_ICACHE_MISS_R_H      12'hB84
`define VX_CSR_MPM_ICACHE_MSHR_ST       12'hB05     // MSHR stalls
`define VX_CSR_MPM_ICACHE_MSHR_ST_H     12'hB85
// PERF: dcache
`define VX_CSR_MPM_DCACHE_READS         12'hB06     // total reads
`define VX_CSR_MPM_DCACHE_READS_H       12'hB86
`define VX_CSR_MPM_DCACHE_WRITES        12'hB07     // total writes
`define VX_CSR_MPM_DCACHE_WRITES_H      12'hB87
`define VX_CSR_MPM_DCACHE_MISS_R        12'hB08     // read misses
`define VX_CSR_MPM_DCACHE_MISS_R_H      12'hB88
`define VX_CSR_MPM_DCACHE_MISS_W        12'hB09     // write misses
`define VX_CSR_MPM_DCACHE_MISS_W_H      12'hB89
`define VX_CSR_MPM_DCACHE_BANK_ST       12'hB0A     // bank conflicts
`define VX_CSR_MPM_DCACHE_BANK_ST_H     12'hB8A
`define VX_CSR_MPM_DCACHE_MSHR_ST       12'hB0B     // MSHR stalls
`define VX_CSR_MPM_DCACHE_MSHR_ST_H     12'hB8B
// PERF: l2cache
`define VX_CSR_MPM_L2CACHE_READS        12'hB0C     // total reads
`define VX_CSR_MPM_L2CACHE_READS_H      12'hB8C
`define VX_CSR_MPM_L2CACHE_WRITES       12'hB0D     // total writes
`define VX_CSR_MPM_L2CACHE_WRITES_H     12'hB8D
`define VX_CSR_MPM_L2CACHE_MISS_R       12'hB0E     // read misses
`define VX_CSR_MPM_L2CACHE_MISS_R_H     12'hB8E
`define VX_CSR_MPM_L2CACHE_MISS_W       12'hB0F     // write misses
`define VX_CSR_MPM_L2CACHE_MISS_W_H     12'hB8F
`define VX_CSR_MPM_L2CACHE_BANK_ST      12'hB10     // bank conflicts
`define VX_CSR_MPM_L2CACHE_BANK_ST_H    12'hB90
`define VX_CSR_MPM_L2CACHE_MSHR_ST      12'hB11     // MSHR stalls
`define VX_CSR_MPM_L2CACHE_MSHR_ST_H    12'hB91
// PERF: l3cache
`define VX_CSR_MPM_L3CACHE_READS        12'hB12     // total reads
`define VX_CSR_MPM_L3CACHE_READS_H      12'hB92
`define VX_CSR_MPM_L3CACHE_WRITES       12'hB13     // total writes
`define VX_CSR_MPM_L3CACHE_WRITES_H     12'hB93
`define VX_CSR_MPM_L3CACHE_MISS_R       12'hB14     // read misses
`define VX_CSR_MPM_L3CACHE_MISS_R_H     12'hB94
`define VX_CSR_MPM_L3CACHE_MISS_W       12'hB15     // write misses
`define VX_CSR_MPM_L3CACHE_MISS_W_H     12'hB95
`define VX_CSR_MPM_L3CACHE_BANK_ST      12'hB16     // bank conflicts
`define VX_CSR_MPM_L3CACHE_BANK_ST_H    12'hB96
`define VX_CSR_MPM_L3CACHE_MSHR_ST      12'hB17     // MSHR stalls
`define VX_CSR_MPM_L3CACHE_MSHR_ST_H    12'hB97
// PERF: memory
`define VX_CSR_MPM_MEM_READS            12'hB18     // total reads
`define VX_CSR_MPM_MEM_READS_H          12'hB98
`define VX_CSR_MPM_MEM_WRITES           12'hB19     // total writes
`define VX_CSR_MPM_MEM_WRITES_H         12'hB99
`define VX_CSR_MPM_MEM_LT               12'hB1A     // memory latency
`define VX_CSR_MPM_MEM_LT_H             12'hB9A
`define VX_CSR_MPM_MEM_BANK_ST          12'hB1E     // bank conflicts
`define VX_CSR_MPM_MEM_BANK_ST_H        12'hB9E
// PERF: lmem
`define VX_CSR_MPM_LMEM_READS           12'hB1B     // memory reads
`define VX_CSR_MPM_LMEM_READS_H         12'hB9B
`define VX_CSR_MPM_LMEM_WRITES          12'hB1C     // memory writes
`define VX_CSR_MPM_LMEM_WRITES_H        12'hB9C
`define VX_CSR_MPM_LMEM_BANK_ST         12'hB1D     // bank conflicts
`define VX_CSR_MPM_LMEM_BANK_ST_H       12'hB9D
// PERF: coalescer
`define VX_CSR_MPM_COALESCER_MISS       12'hB1F     // coalescer misses
`define VX_CSR_MPM_COALESCER_MISS_H     12'hB9F

// Machine Performance-monitoring memory counters (class 3) ///////////////////
// <Add your own counters: use addresses hB03..B1F, hB83..hB9F>

// Machine Information Registers //////////////////////////////////////////////

`define VX_CSR_MVENDORID                12'hF11
`define VX_CSR_MARCHID                  12'hF12
`define VX_CSR_MIMPID                   12'hF13
`define VX_CSR_MHARTID                  12'hF14

// Vector CSRs

`define VX_CSR_VSTART                   12'h008
`define VX_CSR_VXSAT                    12'h009
`define VX_CSR_VXRM                     12'h00A
`define VX_CSR_VCSR                     12'h00F
`define VX_CSR_VL                       12'hC20
`define VX_CSR_VTYPE                    12'hC21
`define VX_CSR_VLENB                    12'hC22
`define VX_CSR_VCYCLE                   12'hC00
`define VX_CSR_VTIME                    12'hC01
`define VX_CSR_VINSTRET                 12'hC02

// GPGU CSRs

`define VX_CSR_THREAD_ID                12'hCC0
`define VX_CSR_WARP_ID                  12'hCC1
`define VX_CSR_CORE_ID                  12'hCC2
`define VX_CSR_ACTIVE_WARPS             12'hCC3
`define VX_CSR_ACTIVE_THREADS           12'hCC4     // warning! this value is also used in LLVM

`define VX_CSR_NUM_THREADS              12'hFC0
`define VX_CSR_NUM_WARPS                12'hFC1
`define VX_CSR_NUM_CORES                12'hFC2
`define VX_CSR_LOCAL_MEM_BASE           12'hFC3

`define VX_MAT_MUL_SIZE                 12'hFC4     // VX_MAT_MUL_SIZE = Matrix Size / TC Size
`define VX_TC_NUM                       12'hFC5
`define VX_TC_SIZE                      12'hFC6



`endif // VX_TYPES_VH
