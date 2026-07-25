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

// Device MMU control surface. Assembles the page-table root from its two DCR
// halves and fans it to every cluster walker; a root write also pulses the
// TLB flush (a done-tree gates completion). The first translation fault from
// any cluster latches here and reads back over the DCR bus; a fault-info
// write drops the report. Non-MMU DCR traffic passes straight through.
module VX_mmu_dcr import VX_gpu_pkg::*; #(
    parameter NUM_CLUSTERS = `VX_CFG_NUM_CLUSTERS
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave  dcr_bus_if,      // from device DCR input
    VX_dcr_bus_if.master dcr_bus_out_if,  // to the cluster DCR arb

    output wire [`VX_CFG_XLEN-1:0] satp,

    // Flush broadcast + done-tree.
    output wire                    flush_req,
    input  wire [NUM_CLUSTERS-1:0] cluster_flush_done,

    // First-fault aggregation (index priority across clusters).
    input  wire [NUM_CLUSTERS-1:0]                   cluster_fault_valid,
    input  wire [NUM_CLUSTERS-1:0][`VX_CFG_XLEN-1:0] cluster_fault_va,
    input  wire [NUM_CLUSTERS-1:0][1:0]              cluster_fault_access,
    input  wire [NUM_CLUSTERS-1:0]                   cluster_fault_amo
);
    localparam CID_W = `UP(`CLOG2(NUM_CLUSTERS));

    wire        dcr_wr    = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw;
    wire        dcr_rd    = dcr_bus_if.req_valid && ~dcr_bus_if.req_data.rw;
    wire [31:0] dcr_wdata = dcr_bus_if.req_data.data[31:0];

    wire is_satp_lo    = dcr_wr && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_SATP_LO);
    wire is_satp_hi    = dcr_wr && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_SATP_HI);
    wire is_fault_clr  = dcr_wr && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_FAULT_INFO);

    // ---------------------------------------------------------------------
    // Page-table root
    // ---------------------------------------------------------------------
    reg [`VX_CFG_XLEN-1:0] satp_r;
    always @(posedge clk) begin
        if (reset) begin
            satp_r <= '0;
        end else begin
            if (is_satp_lo) begin
                satp_r[31:0] <= dcr_wdata;
            end
        `ifdef VX_CFG_XLEN_64
            if (is_satp_hi) begin
                satp_r[`VX_CFG_XLEN-1:32] <= dcr_wdata[`VX_CFG_XLEN-32-1:0];
            end
        `endif
        end
    end
`ifndef VX_CFG_XLEN_64
    `UNUSED_VAR (is_satp_hi)
`endif
    assign satp = satp_r;

    // ---------------------------------------------------------------------
    // Flush: a root write invalidates every TLB level; hold until all done
    // ---------------------------------------------------------------------
    reg flush_pending_r;
    wire flush_done_all = (& cluster_flush_done);
    always @(posedge clk) begin
        if (reset) begin
            flush_pending_r <= 1'b0;
        end else if (is_satp_hi) begin
            flush_pending_r <= 1'b1;
        end else if (flush_pending_r && flush_done_all) begin
            flush_pending_r <= 1'b0;
        end
    end
    assign flush_req = flush_pending_r;

    // ---------------------------------------------------------------------
    // First-fault latch
    // ---------------------------------------------------------------------
    wire any_fault = (| cluster_fault_valid);
    reg [CID_W-1:0] fidx;
    always @(*) begin
        fidx = '0;
        for (int i = NUM_CLUSTERS-1; i >= 0; --i) begin
            if (cluster_fault_valid[i]) begin
                fidx = CID_W'(i);
            end
        end
    end

    reg                    fault_valid_r;
    reg [`VX_CFG_XLEN-1:0] fault_va_r;
    reg [1:0]              fault_acc_r;
    reg                    fault_amo_r;
    always @(posedge clk) begin
        // Programming the root at launch init also arms a clean fault state,
        // discarding any report from before the address space was valid.
        if (reset || is_satp_hi) begin
            fault_valid_r <= 1'b0;
        end else if (is_fault_clr) begin
            fault_valid_r <= 1'b0;
        end else if (any_fault && ~fault_valid_r) begin
            fault_valid_r <= 1'b1;
            fault_va_r    <= cluster_fault_va[fidx];
            fault_acc_r   <= cluster_fault_access[fidx];
            fault_amo_r   <= cluster_fault_amo[fidx];
        end
    end

    wire [31:0] fault_info_word = fault_valid_r
        ? (32'(`VX_MMU_FAULT_VALID)
          | ((32'(fault_acc_r) << `VX_MMU_FAULT_ACCESS_SH) & 32'(`VX_MMU_FAULT_ACCESS))
          | (fault_amo_r ? 32'(`VX_MMU_FAULT_AMO) : 32'd0))
        : 32'd0;

    // ---------------------------------------------------------------------
    // DCR read response for the fault registers, muxed ahead of downstream
    // ---------------------------------------------------------------------
    wire is_fault_info_rd = dcr_rd && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_FAULT_INFO);
    wire is_fault_va_rd   = dcr_rd && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_FAULT_VA);
    wire is_fault_vahi_rd = dcr_rd && (dcr_bus_if.req_data.addr == `VX_DCR_MMU_FAULT_VA_HI);
    wire is_mmu_rd = is_fault_info_rd || is_fault_va_rd || is_fault_vahi_rd;

    // The DCR read handshake is pulse-then-poll: the host asserts the request
    // for one cycle, drops it, then samples rsp_valid. A response tied to the
    // request cycle is gone before the poll, so latch it when the read is seen
    // and hold it until the next DCR access retires it.
    reg       mmu_rsp_valid_r;
    reg [31:0] mmu_rsp_data_r;
    always @(posedge clk) begin
        if (reset) begin
            mmu_rsp_valid_r <= 1'b0;
        end else if (is_mmu_rd) begin
            mmu_rsp_valid_r <= 1'b1;
        end else if (dcr_bus_if.req_valid) begin
            mmu_rsp_valid_r <= 1'b0;
        end
        if (is_fault_info_rd) begin
            mmu_rsp_data_r <= fault_info_word;
        end else if (is_fault_va_rd) begin
            mmu_rsp_data_r <= fault_va_r[31:0];
        end else if (is_fault_vahi_rd) begin
        `ifdef VX_CFG_XLEN_64
            mmu_rsp_data_r <= fault_va_r[`VX_CFG_XLEN-1:32];
        `else
            mmu_rsp_data_r <= 32'd0;
        `endif
        end
    end

    // ---------------------------------------------------------------------
    // Pass-through: requests forward downstream; responses mux MMU ahead of
    // the cluster reply (DCR reads are serialized, so only one fires).
    // ---------------------------------------------------------------------
    assign dcr_bus_out_if.req_valid = dcr_bus_if.req_valid;
    assign dcr_bus_out_if.req_data  = dcr_bus_if.req_data;

    assign dcr_bus_if.rsp_valid = mmu_rsp_valid_r || dcr_bus_out_if.rsp_valid;
    assign dcr_bus_if.rsp_data  = mmu_rsp_valid_r ? dcr_rsp_t'(mmu_rsp_data_r)
                                                  : dcr_bus_out_if.rsp_data;

endmodule
