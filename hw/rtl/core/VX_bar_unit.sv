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

module VX_bar_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // request from SFU
    input wire              req_valid,
    input wire [NW_WIDTH-1:0] req_wid,
    input barrier_t         req_data,
    input wire [`NUM_WARPS-1:0] active_warps,

    // reading phase_r number (valid one cycle behind req_valid)
    input wire [BAR_ADDR_W-1:0] read_addr,
    output wire             read_phase,

    // global barrier interface
`ifdef GBAR_ENABLE
    output VX_gbar_bus_if.master gbar_bus_if,
`endif

    // Output to scheduler
    output wire             unlock_valid, // unlock stalled warps
    output wire [`NUM_WARPS-1:0] unlock_mask // warps to unlock
);
    `UNUSED_SPARAM (INSTANCE_ID)

    //                     warp mask + warp count_r + phase_r
    localparam BAR_DATAW = `NUM_WARPS + NW_WIDTH + 1;

    logic [`NUM_WARPS-1:0] mask_r, mask_n;
    logic [NW_WIDTH-1:0]   count_r, count_n;
    logic                  phase_r, phase_n;

    logic                  unlock_valid_n;
    logic [`NUM_WARPS-1:0] unlock_mask_n;

`ifdef GBAR_ENABLE
    logic gbar_req_valid_r, gbar_req_valid_n;
    logic [NB_WIDTH-1:0] gbar_req_id_r, gbar_req_id_n;
    logic [NC_WIDTH-1:0] gbar_req_size_m1_r, gbar_req_size_m1_n;
`endif

    wire req_data_valid = req_valid && req_data.valid;

    wire [`NUM_WARPS-1:0] wait_mask = ((`NUM_WARPS)'(1) << req_wid) | mask_r;
    wire [NW_WIDTH-1:0] next_count  = count_r + NW_WIDTH'(1);
    wire next_phase  = ~phase_r;

    always @(*) begin
        mask_n  = mask_r;
        count_n = count_r;
        phase_n = phase_r;
        unlock_valid_n = 0;
        unlock_mask_n = 'x;
    `ifdef GBAR_ENABLE
        gbar_req_valid_n = gbar_req_valid_r;
        gbar_req_id_n = gbar_req_id_r;
        gbar_req_size_m1_n = gbar_req_size_m1_r;
    `endif
        // local barrier scheduling
        if (req_data_valid && ~req_data.is_global) begin
            if (req_data.is_arrive) begin
                // barrier arrival
                if (count_r == req_data.size_m1) begin
                    count_n = '0;
                    mask_n  = '0;
                    unlock_valid_n = 1; // release waiting warps
                    unlock_mask_n = req_data.is_async ? mask_r : wait_mask;
                    phase_n = next_phase; // advance phase
                end else begin
                    count_n = next_count;
                    if (!req_data.is_async) begin
                        mask_n = wait_mask;
                    end
                end
            end else begin
                // barrier waiting
                if (req_data.phase != phase_r) begin
                    unlock_valid_n = 1; // release warp
                    unlock_mask_n = (`NUM_WARPS)'(1) << req_wid;
                end else begin
                    // add warp to wait list
                    mask_n = wait_mask;
                end
            end
        end

    `ifdef GBAR_ENABLE
        // global barrier scheduling
        if (req_data_valid && req_data.is_global) begin
            if (req_data.is_arrive) begin
                // barrier arrival
                if (wait_mask == active_warps) begin
                    mask_n  = '0;
                    gbar_req_valid_n = 1; // notify global barrier
                    gbar_req_id_n = req_data.id;
                    gbar_req_size_m1_n = NC_WIDTH'(req_data.size_m1);
                end else begin
                    mask_n = wait_mask;
                end
            end
        end
        if (gbar_bus_if.rsp_valid && (gbar_bus_if.rsp_data.id == gbar_req_id_r)) begin
            unlock_valid_n = 1; // release stalled warps
            unlock_mask_n = active_warps; // release all active warps
        end
        if (gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            gbar_req_valid_n = 0;
        end
    `endif
    end

    // Barriers store
    wire [BAR_DATAW-1:0] store_rdata;
    reg [BAR_ADDR_W-1:0] write_addr;

    VX_dp_ram #(
        .DATAW    (BAR_DATAW),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("R"),
        .RADDR_REG(1)
    ) barrier_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (req_data_valid),
        .wren  (1'b1),
        .raddr (read_addr),
        .waddr (write_addr),
        .wdata ({mask_n, count_n, phase_n}),
        .rdata (store_rdata)
    );

    // Store reset handling
    reg [(1 << BAR_ADDR_BITS)-1:0] store_valids;
    wire is_rdw_hazard = req_data_valid && (write_addr == read_addr);
    always @(posedge clk) begin
        if (reset) begin
            store_valids <= '0;
            mask_r  <= '0;
            count_r <= '0;
            phase_r <= '0;
        end else begin
            if (req_data_valid) begin
                store_valids[write_addr] <= 1'b1;
            end
            if (is_rdw_hazard) begin
                {mask_r, count_r, phase_r} <= {mask_n, count_n, phase_n};
            end else begin
                {mask_r, count_r, phase_r} <= store_valids[read_addr] ? store_rdata : '0;
            end
        end
        write_addr <= read_addr;
    end

    wire phase_with_reset = store_valids[read_addr] ? store_rdata[0] : '0;
    wire phase_async = is_rdw_hazard ? phase_n : phase_with_reset;

    reg unlock_valid_r;
    reg [`NUM_WARPS-1:0] unlock_mask_r;

    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= unlock_valid_n;
        end
        unlock_mask_r <= unlock_mask_n;
    end

    assign read_phase   = phase_async;
    assign unlock_valid = unlock_valid_r;
    assign unlock_mask  = unlock_mask_r;

`ifdef GBAR_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            gbar_req_valid_r <= 0;
        end else begin
            gbar_req_valid_r <= gbar_req_valid_n;
        end
        gbar_req_size_m1_r <= gbar_req_size_m1_n;
        gbar_req_id_r <= gbar_req_id_n;
    end

    assign gbar_bus_if.req_valid        = gbar_req_valid_r;
    assign gbar_bus_if.req_data.id      = gbar_req_id_r;
    assign gbar_bus_if.req_data.size_m1 = gbar_req_size_m1_r;
    assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % `NUM_CORES);
`endif

endmodule
