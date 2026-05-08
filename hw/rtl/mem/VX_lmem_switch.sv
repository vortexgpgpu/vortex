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

// Splits an LSU request into a global-memory part and a local-memory
// part using each lane's is_addr_local user bit. Both subsets fire
// independently with masked-out lanes; lsu_in_if.req_ready waits for
// whichever subset(s) are non-empty.
//
// AMO on the local path is unsupported (proposal §6); the assertion
// below fires if any active lane reaches this switch with both
// amo_valid and is_addr_local set. The local-path attr also has its
// AMO bits stripped so downstream LMEM banks never see amo_valid.

module VX_lmem_switch import VX_gpu_pkg::*; #(
    parameter GLOBAL_OUT_BUF = 0,
    parameter LOCAL_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,
    VX_lsu_mem_if.slave     lsu_in_if,
    VX_lsu_mem_if.master    global_out_if,
    VX_lsu_mem_if.master    local_out_if
);
    localparam REQ_DATAW = `NUM_LSU_LANES + 1 + `NUM_LSU_LANES * (LSU_WORD_SIZE + LSU_ADDR_WIDTH + MEM_ATTR_WIDTH + LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_LSU_LANES + `NUM_LSU_LANES * (LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;

    // Per-lane is_addr_local from the user bits at the fixed offset.
    wire [`NUM_LSU_LANES-1:0] is_addr_local_mask;
    for (genvar i = 0; i < `NUM_LSU_LANES; ++i) begin : g_is_addr_local_mask
        assign is_addr_local_mask[i] = lsu_in_if.req_data.user[i][MEM_ATTR_LOCAL_OFFS];
    end

    wire [`NUM_LSU_LANES-1:0] global_mask = lsu_in_if.req_data.mask & ~is_addr_local_mask;
    wire [`NUM_LSU_LANES-1:0] local_mask  = lsu_in_if.req_data.mask &  is_addr_local_mask;

    wire is_addr_global = |global_mask;
    wire is_addr_local  = |local_mask;

    wire req_global_ready;
    wire req_local_ready;

    // Both subsets must be accepted before we release the LSU input.
    assign lsu_in_if.req_ready = (!is_addr_global || req_global_ready)
                              && (!is_addr_local  || req_local_ready);

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(GLOBAL_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(GLOBAL_OUT_BUF))
    ) req_global_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_global),
        .data_in   ({
            global_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.user,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_global_ready),
        .valid_out (global_out_if.req_valid),
        .data_out  (global_out_if.req_data),
        .ready_out (global_out_if.req_ready)
    );

    // Strip per-lane AMO bits on the local path so the LMEM banks never
    // observe amo.amo_valid. Other attr fields pass through unchanged.
    wire [`NUM_LSU_LANES-1:0][MEM_ATTR_WIDTH-1:0] local_user;
    for (genvar i = 0; i < `NUM_LSU_LANES; ++i) begin : g_local_user
        mem_bus_attr_t lane_clean;
        always_comb begin
            lane_clean      = mem_bus_attr_t'(lsu_in_if.req_data.user[i]);
            lane_clean.amo  = '0;
        end
        assign local_user[i] = MEM_ATTR_WIDTH'(lane_clean);
    end

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(LOCAL_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(LOCAL_OUT_BUF))
    ) req_local_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_local),
        .data_in   ({
            local_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            local_user,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_local_ready),
        .valid_out (local_out_if.req_valid),
        .data_out  (local_out_if.req_data),
        .ready_out (local_out_if.req_ready)
    );

    // Synth-time assertion mirror of SimX's local_mem_switch guard
    // (sim/simx/mem/local_mem_switch.cpp:65). AMO on Shared/LMEM is
    // out of scope (proposal §6).
    for (genvar lane = 0; lane < `NUM_LSU_LANES; ++lane) begin : g_amo_lmem_assert
        wire amo_local_lane = lsu_in_if.req_valid
                           && lsu_in_if.req_data.mask[lane]
                           && lsu_in_if.req_data.user[lane][MEM_ATTR_AMO_OFFS]    // amo_valid
                           && lsu_in_if.req_data.user[lane][MEM_ATTR_LOCAL_OFFS]; // is_addr_local
        always_comb begin
            if (amo_local_lane) begin
                `ASSERT(0, ("AMO on Shared (LMEM) is unsupported"));
            end
        end
    end

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            local_out_if.rsp_valid,
            global_out_if.rsp_valid
        }),
        .ready_in  ({
            local_out_if.rsp_ready,
            global_out_if.rsp_ready
        }),
        .data_in   ({
            local_out_if.rsp_data,
            global_out_if.rsp_data
        }),
        .data_out  (lsu_in_if.rsp_data),
        .valid_out (lsu_in_if.rsp_valid),
        .ready_out (lsu_in_if.rsp_ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
