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
    wire [`NUM_WARPS-1:0]           active_warps;
    wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;

    // Read port: slave sends wid + cta_id, master returns selected mscratch
    // and cta_csrs. csr_rd_cta_id is sourced from execute_if.data.header.cta_id
    // so the scheduler can index its per-CTA table directly without an
    // internal cta_id_per_warp_r lookup.
    logic [NW_WIDTH-1:0]            csr_rd_wid;
    logic [NCTA_WIDTH-1:0]          csr_rd_cta_id;
    logic [`MEM_ADDR_WIDTH-1:0]     mscratch;
    cta_csrs_t                      cta_csrs;

`ifdef VM_ENABLE
    logic [`XLEN-1:0]               csr_satp;
`endif

    // Write port: slave notifies scheduler of MSCRATCH CSR writes
    logic                           csr_wr_valid;
    logic [NW_WIDTH-1:0]            csr_wr_wid;
    logic [`MEM_ADDR_WIDTH-1:0]     csr_wr_data;

    modport master (
        output cycles,
        output instret,
        output active_warps,
        output thread_masks,
        output mscratch,
        output cta_csrs,
    `ifdef VM_ENABLE
        input csr_satp,
    `endif
        input  csr_rd_wid,
        input  csr_rd_cta_id,
        input  csr_wr_valid,
        input  csr_wr_wid,
        input  csr_wr_data
    );

    modport slave (
        input  cycles,
        input  instret,
        input  active_warps,
        input  thread_masks,
        input  mscratch,
        input  cta_csrs,
    `ifdef VM_ENABLE
        output csr_satp,
    `endif
        output csr_rd_wid,
        output csr_rd_cta_id,
        output csr_wr_valid,
        output csr_wr_wid,
        output csr_wr_data
    );

endinterface
