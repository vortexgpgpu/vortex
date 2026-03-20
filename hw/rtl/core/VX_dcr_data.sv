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

module VX_dcr_data import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // DCR-triggered CSR read bypass (for MPM perf counter readback)
    VX_dcr_csr_if.master    dcr_csr_if,

    // DCR-triggered cache flush
    VX_dcr_flush_if.master dcr_flush_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // MPM perf-counter readback via DCR interface ////////////////////////////

    wire [7:0]  mpm_class      = dcr_bus_if.req_data.data[22 +: 8];
    wire [5:0]  mpm_tag_idx    = dcr_bus_if.req_data.data[16 +: 6];
    wire [15:0] mpm_target_cid = dcr_bus_if.req_data.data[0  +:16];

    wire [`VX_CSR_ADDR_BITS-1:0] mpm_csr_addr = mpm_tag_idx[5]
        ? (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE_H) + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]))
        : (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE)   + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]));

    // Latch a read request when it is for this core and addr == MPM_VALUE
    reg                         dcr_csr_pending_r;
    reg [`VX_CSR_ADDR_BITS-1:0] dcr_csr_addr_r;

    wire is_mpm_read = dcr_bus_if.req_valid
                    && ~dcr_bus_if.req_data.rw
                    && (dcr_bus_if.req_data.addr == `VX_DCR_BASE_MPM_VALUE)
                    && (mpm_target_cid == 16'(CORE_ID))
                    && ~dcr_csr_pending_r;

    always @(posedge clk) begin
        if (reset) begin
            dcr_csr_pending_r <= 1'b0;
            dcr_csr_addr_r    <= '0;
        end else if (is_mpm_read) begin
            dcr_csr_pending_r <= 1'b1;
            dcr_csr_addr_r    <= mpm_csr_addr;
        end else if (dcr_csr_if.ready) begin
            dcr_csr_pending_r <= 1'b0;
        end
    end

    assign dcr_csr_if.valid = dcr_csr_pending_r;
    assign dcr_csr_if.addr  = dcr_csr_addr_r;
    assign dcr_csr_if.mpm_class = mpm_class;

    wire dcr_csr_if_fire = dcr_csr_if.valid && dcr_csr_if.ready;

    // Cache flush via DCR interface /////////////////////////////////////////

    reg flush_pending_r;

    wire is_flush_read = dcr_bus_if.req_valid
                      && ~dcr_bus_if.req_data.rw
                      && (dcr_bus_if.req_data.addr == `VX_DCR_BASE_CACHE_FLUSH)
                      && (mpm_target_cid == 16'(CORE_ID))
                      && ~flush_pending_r;

    always @(posedge clk) begin
        if (reset) begin
            flush_pending_r <= 1'b0;
        end else if (is_flush_read) begin
            flush_pending_r <= 1'b1;
        end else if (dcr_flush_if.done) begin
            flush_pending_r <= 1'b0;
        end
    end

    assign dcr_flush_if.req = flush_pending_r;

    wire flush_done = flush_pending_r && dcr_flush_if.done;

    // Return CSR data or flush-done or flush-done on DCR response bus
    always @(posedge clk) begin
        if (reset) begin
            dcr_bus_if.rsp_valid <= 1'b0;
        end else begin
            dcr_bus_if.rsp_valid <= dcr_csr_if_fire || flush_done;
            if (dcr_csr_if_fire) begin
                dcr_bus_if.rsp_data <= dcr_csr_if.value;
            end else if (flush_done) begin
                dcr_bus_if.rsp_data <= '0;
            end
        end
    end

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (dcr_bus_if.req_valid && dcr_bus_if.req_data.rw) begin
            `TRACE(2, ("%t: %s: state=", $time, INSTANCE_ID))
            VX_trace_pkg::trace_base_dcr(1, dcr_bus_if.req_data.addr);
            `TRACE(2, (", rw=%b, data=0x%h\n", dcr_bus_if.req_data.rw, dcr_bus_if.req_data.data))
        end
    end
`endif

endmodule
