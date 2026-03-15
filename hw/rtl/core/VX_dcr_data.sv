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

    // Outputs
    output base_dcrs_t      base_dcrs,

    // DCR-triggered CSR read bypass (for MPM perf counter readback)
    VX_dcr_csr_if.master    dcr_csr_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    base_dcrs_t dcrs;

    always @(posedge clk) begin
       if (dcr_bus_if.req_valid && dcr_bus_if.req_data.rw) begin
            case (dcr_bus_if.req_data.addr)
            `VX_DCR_BASE_STARTUP_ADDR0 : dcrs.startup_addr[31:0] <= dcr_bus_if.req_data.data;
        `ifdef XLEN_64
            `VX_DCR_BASE_STARTUP_ADDR1 : dcrs.startup_addr[63:32] <= dcr_bus_if.req_data.data;
        `endif
            `VX_DCR_BASE_STARTUP_ARG0 : dcrs.startup_arg[31:0] <= dcr_bus_if.req_data.data;
        `ifdef XLEN_64
            `VX_DCR_BASE_STARTUP_ARG1 : dcrs.startup_arg[63:32] <= dcr_bus_if.req_data.data;
        `endif
            `VX_DCR_BASE_MPM_CLASS : dcrs.mpm_class <= dcr_bus_if.req_data.data[7:0];
            default:;
            endcase
        end
    end

    assign base_dcrs = dcrs;

    // MPM perf-counter readback via DCR interface ////////////////////////////
    // Tag encoding (from runtime/stub/utils.cpp):
    //   req_data.data[21:16] = csr_index (0..31 = lo half, 32..63 = hi half)
    //   req_data.data[15:0]  = target core_id

    wire [5:0]  mpm_tag_idx    = dcr_bus_if.req_data.data[21:16];
    wire [15:0] mpm_target_cid = dcr_bus_if.req_data.data[15:0];

    wire [`VX_CSR_ADDR_BITS-1:0] mpm_csr_addr = mpm_tag_idx[5]
        ? (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE_H) + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]))
        : (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE)   + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]));

    // Latch a read request when it is for this core and addr == MPM_VALUE
    reg                             dcr_csr_pending_r;
    reg [`VX_CSR_ADDR_BITS-1:0]     dcr_csr_addr_r;

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

    wire dcr_csr_if_fire = dcr_csr_if.valid && dcr_csr_if.ready;

    // Return CSR data on DCR response bus
    always @(posedge clk) begin
        if (reset) begin
            dcr_bus_if.rsp_valid <= 1'b0;
        end else begin
            dcr_bus_if.rsp_valid <= dcr_csr_if_fire;
            if (dcr_csr_if_fire) begin
                dcr_bus_if.rsp_data <= dcr_csr_if.value;
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
