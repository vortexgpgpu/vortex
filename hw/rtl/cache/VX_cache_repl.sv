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

`include "VX_cache_define.vh"

// Fast PLRU encoder and decoder utility
// Adapted from BaseJump STL: http://bjump.org/data_out.html

module plru_decoder #(
    parameter NUM_WAYS      = 1,
    parameter WAY_IDX_BITS  = $clog2(NUM_WAYS),
    parameter WAY_IDX_WIDTH = `UP(WAY_IDX_BITS)
) (
    input wire [WAY_IDX_WIDTH-1:0]    way_idx,
    output wire [`UP(NUM_WAYS-1)-1:0] lru_data,
    output wire [`UP(NUM_WAYS-1)-1:0] lru_mask
);
    if (NUM_WAYS > 1) begin : g_dec
        wire [`UP(NUM_WAYS-1)-1:0] data;
    `IGNORE_UNOPTFLAT_BEGIN
        wire [`UP(NUM_WAYS-1)-1:0] mask;
    `IGNORE_UNOPTFLAT_END
        for (genvar i = 0; i < NUM_WAYS-1; ++i) begin : g_i
            if (i == 0) begin : g_i_0
                assign mask[i] = 1'b1;
            end else if (i % 2 == 1) begin : g_i_odd
                assign mask[i] = mask[(i-1)/2] & ~way_idx[WAY_IDX_BITS-$clog2(i+2)+1];
            end else begin : g_i_even
                assign mask[i] = mask[(i-2)/2] & way_idx[WAY_IDX_BITS-$clog2(i+2)+1];
            end
            assign data[i] = ~way_idx[WAY_IDX_BITS-$clog2(i+2)];
        end
        assign lru_data = data;
        assign lru_mask = mask;
    end else begin : g_no_dec
        `UNUSED_VAR (way_idx)
        assign lru_data = '0;
        assign lru_mask = '0;
    end

endmodule

module plru_encoder #(
    parameter NUM_WAYS      = 1,
    parameter WAY_IDX_BITS  = $clog2(NUM_WAYS),
    parameter WAY_IDX_WIDTH = `UP(WAY_IDX_BITS)
) (
    input wire [`UP(NUM_WAYS-1)-1:0] lru_in,
    output wire [WAY_IDX_WIDTH-1:0] way_idx
);
    if (NUM_WAYS > 1) begin : g_enc
        wire [WAY_IDX_BITS-1:0] tmp;
        for (genvar i = 0; i < WAY_IDX_BITS; ++i) begin : g_i
            if (i == 0) begin : g_i_0
                assign tmp[WAY_IDX_WIDTH-1] = lru_in[0];
            end else begin : g_i_n
                VX_mux #(
                    .N (2**i)
                ) mux (
                    .data_in  (lru_in[((2**i)-1)+:(2**i)]),
                    .sel_in   (tmp[WAY_IDX_BITS-1-:i]),
                    .data_out (tmp[WAY_IDX_BITS-1-i])
                );
            end
        end
        assign way_idx = tmp;
    end else begin : g_no_enc
        `UNUSED_VAR (lru_in)
        assign way_idx = '0;
    end

endmodule

module VX_cache_repl #(
    parameter CACHE_SIZE = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE  = 64,
    // Number of banks
    parameter NUM_BANKS  = 1,
    // Number of associative ways
    parameter NUM_WAYS   = 1,
    // replacement policy
    parameter REPL_POLICY = `CS_REPL_CYCLIC
) (
    input wire clk,
    input wire reset,
    input wire stall,
    input wire hit_valid,
    input wire [`CS_LINE_SEL_BITS-1:0] hit_line,
    input wire [`CS_WAY_SEL_WIDTH-1:0] hit_way,
    input wire repl_valid,
    input wire [`CS_LINE_SEL_BITS-1:0] repl_line,
    output wire [`CS_WAY_SEL_WIDTH-1:0] repl_way
);
    localparam WAY_SEL_WIDTH = `CS_WAY_SEL_WIDTH;
    `UNUSED_VAR (stall)

    if (NUM_WAYS > 1) begin : g_enable
        if (REPL_POLICY == `CS_REPL_PLRU) begin : g_plru
            // Pseudo Least Recently Used replacement policy
            localparam LRU_WIDTH = `UP(NUM_WAYS-1);

            wire [LRU_WIDTH-1:0] plru_rdata;
            wire [LRU_WIDTH-1:0] plru_wdata;
            wire [LRU_WIDTH-1:0] plru_wmask;

            VX_dp_ram #(
                .DATAW (LRU_WIDTH),
                .SIZE  (`CS_LINES_PER_BANK),
                .WRENW (LRU_WIDTH),
                .RDW_MODE ("R"),
                .RADDR_REG (1)
            ) plru_store (
                .clk   (clk),
                .reset (reset),
                .read  (repl_valid),
                .write (hit_valid),
                .wren  (plru_wmask),
                .waddr (hit_line),
                .raddr (repl_line),
                .wdata (plru_wdata),
                .rdata (plru_rdata)
            );

            plru_decoder #(
                .NUM_WAYS (NUM_WAYS)
            ) plru_dec (
                .way_idx  (hit_way),
                .lru_data (plru_wdata),
                .lru_mask (plru_wmask)
            );

            plru_encoder #(
                .NUM_WAYS (NUM_WAYS)
            ) plru_enc (
                .lru_in  (plru_rdata),
                .way_idx (repl_way)
            );

        end else if (REPL_POLICY == `CS_REPL_CYCLIC) begin : g_cyclic
            // Cyclic replacement policy
            `UNUSED_VAR (hit_valid)
            `UNUSED_VAR (hit_line)
            `UNUSED_VAR (hit_way)

            wire [WAY_SEL_WIDTH-1:0] ctr_rdata;
            wire [WAY_SEL_WIDTH-1:0] ctr_wdata = ctr_rdata + 1;

            VX_sp_ram #(
                .DATAW (WAY_SEL_WIDTH),
                .SIZE  (`CS_LINES_PER_BANK),
                .RDW_MODE ("R"),
                .RADDR_REG (1)
            ) ctr_store (
                .clk   (clk),
                .reset (reset),
                .read  (repl_valid),
                .write (repl_valid),
                .wren  (1'b1),
                .addr  (repl_line),
                .wdata (ctr_wdata),
                .rdata (ctr_rdata)
            );

            assign repl_way = ctr_rdata;
        end else begin : g_random
            // Random replacement policy
            `UNUSED_VAR (hit_valid)
            `UNUSED_VAR (hit_line)
            `UNUSED_VAR (hit_way)
            `UNUSED_VAR (repl_valid)
            `UNUSED_VAR (repl_line)
            reg [WAY_SEL_WIDTH-1:0] victim_idx;
            always @(posedge clk) begin
                if (reset) begin
                    victim_idx <= 0;
                end else if (~stall) begin
                    victim_idx <= victim_idx + 1;
                end
            end
            assign repl_way = victim_idx;
        end
    end else begin : g_disable
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (hit_valid)
        `UNUSED_VAR (hit_line)
        `UNUSED_VAR (hit_way)
        `UNUSED_VAR (repl_valid)
        `UNUSED_VAR (repl_line)
        assign repl_way = 1'b0;
    end

endmodule
