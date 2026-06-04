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

`include "VX_define.vh"

interface VX_sched_csr_if import VX_gpu_pkg::*; ();

    wire [PERF_CTR_BITS-1:0]        cycles;
    wire [PERF_CTR_BITS-1:0]        instret;
    wire [`VX_CFG_NUM_WARPS-1:0]           active_warps;
    wire [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] thread_masks;

    // Read port: slave sends wid + cta_id, master returns selected mscratch
    // and cta_csrs. csr_rd_cta_id must carry the executing instruction's
    // cta_id so the scheduler can index its per-CTA table directly.
    logic [NW_WIDTH-1:0]            csr_rd_wid;
    logic [NCTA_WIDTH-1:0]          csr_rd_cta_id;
    logic [`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch;
    cta_csrs_t                      cta_csrs;
    logic [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_tid;

`ifdef VX_CFG_VM_ENABLE
    logic [`VX_CFG_XLEN-1:0]        csr_satp;
`endif

    // Write port: slave notifies scheduler of MSCRATCH CSR writes
    logic                           csr_wr_valid;
    logic [NW_WIDTH-1:0]            csr_wr_wid;
    logic [`VX_CFG_MEM_ADDR_WIDTH-1:0]     csr_wr_data;

    // Per-warp machine-mode trap CSRs live in the scheduler (alongside
    // mscratch) because the scheduler owns warp PC redirection. Reads are
    // returned already selected by csr_rd_wid; csrw writes are forwarded
    // here carrying the CSR address. csr_wr_wid is shared with mscratch.
    logic [`VX_CFG_XLEN-1:0]               csr_mstatus;
    logic [`VX_CFG_XLEN-1:0]               csr_mtvec;
    logic [`VX_CFG_XLEN-1:0]               csr_mepc;
    logic [`VX_CFG_XLEN-1:0]               csr_mcause;
    logic [`VX_CFG_XLEN-1:0]               csr_mtval;
    logic                           trap_csr_wr_valid;
    logic [`VX_CSR_ADDR_BITS-1:0]   trap_csr_wr_addr;
    logic [`VX_CFG_XLEN-1:0]               trap_csr_wr_data;

    modport master (
        output cycles,
        output instret,
        output active_warps,
        output thread_masks,
        output mscratch,
        output cta_csrs,
        output cta_tid,
        output csr_mstatus,
        output csr_mtvec,
        output csr_mepc,
        output csr_mcause,
        output csr_mtval,
    `ifdef VX_CFG_VM_ENABLE
        input csr_satp,
    `endif
        input  csr_rd_wid,
        input  csr_rd_cta_id,
        input  csr_wr_valid,
        input  csr_wr_wid,
        input  csr_wr_data,
        input  trap_csr_wr_valid,
        input  trap_csr_wr_addr,
        input  trap_csr_wr_data
    );

    modport slave (
        input  cycles,
        input  instret,
        input  active_warps,
        input  thread_masks,
        input  mscratch,
        input  cta_csrs,
        input  cta_tid,
        input  csr_mstatus,
        input  csr_mtvec,
        input  csr_mepc,
        input  csr_mcause,
        input  csr_mtval,
    `ifdef VX_CFG_VM_ENABLE
        output csr_satp,
    `endif
        output csr_rd_wid,
        output csr_rd_cta_id,
        output csr_wr_valid,
        output csr_wr_wid,
        output csr_wr_data,
        output trap_csr_wr_valid,
        output trap_csr_wr_addr,
        output trap_csr_wr_data
    );

endinterface
