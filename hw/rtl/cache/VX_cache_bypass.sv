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

module VX_cache_bypass #(
    parameter NUM_REQS          = 1,
    parameter TAG_SEL_IDX       = 0,

    parameter PASSTHRU          = 0,
    parameter NC_ENABLE         = 0,

    parameter WORD_SIZE         = 1,
    parameter LINE_SIZE         = 1,

    parameter CORE_ADDR_WIDTH   = 1,

    parameter CORE_TAG_WIDTH    = 1,

    parameter MEM_ADDR_WIDTH    = 1,
    parameter MEM_TAG_IN_WIDTH  = 1,
    parameter MEM_TAG_OUT_WIDTH = 1,

    parameter UUID_WIDTH        = 0,

    parameter CORE_OUT_BUF      = 0,
    parameter MEM_OUT_BUF       = 0,

    parameter CORE_DATA_WIDTH   = WORD_SIZE * 8
 ) (
    input wire clk,
    input wire reset,

    // Core request in
    VX_mem_bus_if.slave     core_bus_in_if [NUM_REQS],

    // Core request out
    VX_mem_bus_if.master    core_bus_out_if [NUM_REQS],

    // Memory request in
    VX_mem_bus_if.slave     mem_bus_in_if,

    // Memory request out
    VX_mem_bus_if.master    mem_bus_out_if
);
    localparam DIRECT_PASSTHRU  = PASSTHRU && (`CS_WORD_SEL_BITS == 0) && (NUM_REQS == 1);

    localparam REQ_SEL_BITS     = `CLOG2(NUM_REQS);
    localparam MUX_DATAW        = 1 + WORD_SIZE + CORE_ADDR_WIDTH + `ADDR_TYPE_WIDTH + CORE_DATA_WIDTH + CORE_TAG_WIDTH;

    localparam WORDS_PER_LINE   = LINE_SIZE / WORD_SIZE;
    localparam WSEL_BITS        = `CLOG2(WORDS_PER_LINE);

    localparam CORE_TAG_ID_BITS = CORE_TAG_WIDTH - UUID_WIDTH;
    localparam MEM_TAG_ID_BITS  = REQ_SEL_BITS + WSEL_BITS + CORE_TAG_ID_BITS;
    localparam MEM_TAG_BYPASS_BITS = UUID_WIDTH + MEM_TAG_ID_BITS;

    `STATIC_ASSERT(0 == (`IO_BASE_ADDR % `MEM_BLOCK_SIZE), ("invalid parameter"))

    // handle core requests ///////////////////////////////////////////////////

    wire core_req_nc_valid;
    wire [NUM_REQS-1:0] core_req_nc_valids;
    wire [NUM_REQS-1:0] core_req_nc_idxs;
    wire [`UP(REQ_SEL_BITS)-1:0] core_req_nc_idx;
    wire [NUM_REQS-1:0] core_req_nc_sel;
    wire core_req_nc_ready;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (PASSTHRU != 0) begin
            assign core_req_nc_idxs[i] = 1'b1;
        end else if (NC_ENABLE) begin
            assign core_req_nc_idxs[i] = core_bus_in_if[i].req_data.atype[`ADDR_TYPE_IO];
        end else begin
            assign core_req_nc_idxs[i] = 1'b0;
        end
        assign core_req_nc_valids[i] = core_bus_in_if[i].req_valid && core_req_nc_idxs[i];
    end

    VX_generic_arbiter #(
        .NUM_REQS    (NUM_REQS),
        .TYPE        (PASSTHRU ? "R" : "P")
    ) core_req_nc_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (core_req_nc_valids),
        .grant_index  (core_req_nc_idx),
        .grant_onehot (core_req_nc_sel),
        .grant_valid  (core_req_nc_valid),
        .grant_ready  (core_req_nc_ready)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && ~core_req_nc_idxs[i];
        assign core_bus_out_if[i].req_data = core_bus_in_if[i].req_data;
        assign core_bus_in_if[i].req_ready = core_req_nc_valids[i] ? (core_req_nc_ready && core_req_nc_sel[i])
                                                                   : core_bus_out_if[i].req_ready;
    end

    // handle memory requests /////////////////////////////////////////////////

    wire                        mem_req_out_valid;
    wire                        mem_req_out_rw;
    wire [LINE_SIZE-1:0]        mem_req_out_byteen;
    wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_out_addr;
    wire [`ADDR_TYPE_WIDTH-1:0] mem_req_out_atype;
    wire [`CS_LINE_WIDTH-1:0]   mem_req_out_data;
    wire [MEM_TAG_OUT_WIDTH-1:0] mem_req_out_tag;
    wire                        mem_req_out_ready;

    wire                        core_req_nc_sel_rw;
    wire [WORD_SIZE-1:0]        core_req_nc_sel_byteen;
    wire [CORE_ADDR_WIDTH-1:0]  core_req_nc_sel_addr;
    wire [`ADDR_TYPE_WIDTH-1:0] core_req_nc_sel_atype;
    wire [CORE_DATA_WIDTH-1:0]  core_req_nc_sel_data;
    wire [CORE_TAG_WIDTH-1:0]   core_req_nc_sel_tag;

    wire [NUM_REQS-1:0][MUX_DATAW-1:0] core_req_nc_mux_in;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_nc_mux_in[i] = {
            core_bus_in_if[i].req_data.rw,
            core_bus_in_if[i].req_data.byteen,
            core_bus_in_if[i].req_data.addr,
            core_bus_in_if[i].req_data.atype,
            core_bus_in_if[i].req_data.data,
            core_bus_in_if[i].req_data.tag
        };
    end

    assign {
        core_req_nc_sel_rw,
        core_req_nc_sel_byteen,
        core_req_nc_sel_addr,
        core_req_nc_sel_atype,
        core_req_nc_sel_data,
        core_req_nc_sel_tag
    } = core_req_nc_mux_in[core_req_nc_idx];

    assign core_req_nc_ready = ~mem_bus_in_if.req_valid && mem_req_out_ready;

    assign mem_req_out_valid = mem_bus_in_if.req_valid || core_req_nc_valid;
    assign mem_req_out_rw    = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.rw : core_req_nc_sel_rw;
    assign mem_req_out_addr  = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.addr : core_req_nc_sel_addr[WSEL_BITS +: MEM_ADDR_WIDTH];
    assign mem_req_out_atype = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.atype : core_req_nc_sel_atype;

    wire [MEM_TAG_ID_BITS-1:0] mem_req_tag_id_bypass;

    wire [CORE_TAG_ID_BITS-1:0] core_req_in_id = core_req_nc_sel_tag[CORE_TAG_ID_BITS-1:0];

    if (WORDS_PER_LINE > 1) begin
        reg [WORDS_PER_LINE-1:0][WORD_SIZE-1:0] mem_req_byteen_in_r;
        reg [WORDS_PER_LINE-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_in_r;

        wire [WSEL_BITS-1:0] req_wsel = core_req_nc_sel_addr[WSEL_BITS-1:0];

        always @(*) begin
            mem_req_byteen_in_r = '0;
            mem_req_byteen_in_r[req_wsel] = core_req_nc_sel_byteen;

            mem_req_data_in_r = 'x;
            mem_req_data_in_r[req_wsel] = core_req_nc_sel_data;
        end

        assign mem_req_out_byteen = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.byteen : mem_req_byteen_in_r;
        assign mem_req_out_data = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.data : mem_req_data_in_r;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, req_wsel, core_req_in_id});
        end else begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({req_wsel, core_req_in_id});
        end
    end else begin
        assign mem_req_out_byteen = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.byteen : core_req_nc_sel_byteen;
        assign mem_req_out_data = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.data : core_req_nc_sel_data;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, core_req_in_id});
        end else begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_in_id});
        end
    end

    wire [MEM_TAG_BYPASS_BITS-1:0] mem_req_tag_bypass;

    if (UUID_WIDTH != 0) begin
        assign mem_req_tag_bypass = {core_req_nc_sel_tag[CORE_TAG_ID_BITS +: UUID_WIDTH], mem_req_tag_id_bypass};
    end else begin
        assign mem_req_tag_bypass = mem_req_tag_id_bypass;
    end

    if (PASSTHRU != 0) begin
        assign mem_req_out_tag = mem_req_tag_bypass;
        `UNUSED_VAR (mem_bus_in_if.req_data.tag)
    end else begin
        if (NC_ENABLE) begin
            VX_bits_insert #(
                .N   (MEM_TAG_OUT_WIDTH-1),
                .S   (1),
                .POS (TAG_SEL_IDX)
            ) mem_req_tag_in_nc_insert (
                .data_in  (mem_bus_in_if.req_valid ? (MEM_TAG_OUT_WIDTH-1)'(mem_bus_in_if.req_data.tag) : (MEM_TAG_OUT_WIDTH-1)'(mem_req_tag_bypass)),
                .ins_in   (~mem_bus_in_if.req_valid),
                .data_out (mem_req_out_tag)
            );
        end else begin
            assign mem_req_out_tag = mem_bus_in_if.req_data.tag;
        end
    end

    assign mem_bus_in_if.req_ready = mem_req_out_ready;

    `RESET_RELAY (mem_req_reset, reset);

    VX_elastic_buffer #(
        .DATAW   (1 + LINE_SIZE + `CS_MEM_ADDR_WIDTH + `ADDR_TYPE_WIDTH + `CS_LINE_WIDTH + MEM_TAG_OUT_WIDTH),
        .SIZE    ((!DIRECT_PASSTHRU) ? `TO_OUT_BUF_SIZE(MEM_OUT_BUF) : 0),
        .OUT_REG (`TO_OUT_BUF_REG(MEM_OUT_BUF))
    ) mem_req_buf (
        .clk       (clk),
        .reset     (mem_req_reset),
        .valid_in  (mem_req_out_valid),
        .ready_in  (mem_req_out_ready),
        .data_in   ({mem_req_out_rw,             mem_req_out_byteen,             mem_req_out_addr,             mem_req_out_atype,             mem_req_out_data,             mem_req_out_tag}),
        .data_out  ({mem_bus_out_if.req_data.rw, mem_bus_out_if.req_data.byteen, mem_bus_out_if.req_data.addr, mem_bus_out_if.req_data.atype, mem_bus_out_if.req_data.data, mem_bus_out_if.req_data.tag}),
        .valid_out (mem_bus_out_if.req_valid),
        .ready_out (mem_bus_out_if.req_ready)
    );

    // handle core responses //////////////////////////////////////////////////

    wire [NUM_REQS-1:0]                  core_rsp_in_valid;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_in_data;
    wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_in_tag;
    wire [NUM_REQS-1:0]                  core_rsp_in_ready;

    wire is_mem_rsp_nc;
    if (PASSTHRU != 0) begin
        assign is_mem_rsp_nc = mem_bus_out_if.rsp_valid;
    end else begin
        if (NC_ENABLE) begin
            assign is_mem_rsp_nc = mem_bus_out_if.rsp_valid && mem_bus_out_if.rsp_data.tag[TAG_SEL_IDX];
        end else begin
            assign is_mem_rsp_nc = 1'b0;
        end
    end

    wire [(MEM_TAG_OUT_WIDTH - NC_ENABLE)-1:0] mem_rsp_tag_id_nc;

    VX_bits_remove #(
        .N   (MEM_TAG_OUT_WIDTH),
        .S   (NC_ENABLE),
        .POS (TAG_SEL_IDX)
    ) mem_rsp_tag_in_nc_remove (
        .data_in  (mem_bus_out_if.rsp_data.tag),
        .data_out (mem_rsp_tag_id_nc)
    );

    wire [`UP(REQ_SEL_BITS)-1:0] rsp_idx;
    if (NUM_REQS > 1) begin
        assign rsp_idx = mem_rsp_tag_id_nc[(CORE_TAG_ID_BITS + WSEL_BITS) +: REQ_SEL_BITS];
    end else begin
        assign rsp_idx = 1'b0;
    end

    reg [NUM_REQS-1:0] rsp_nc_valid_r;
    always @(*) begin
        rsp_nc_valid_r = '0;
        rsp_nc_valid_r[rsp_idx] = is_mem_rsp_nc;
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_rsp_in_valid[i] = core_bus_out_if[i].rsp_valid || rsp_nc_valid_r[i];
        assign core_bus_out_if[i].rsp_ready = core_rsp_in_ready[i];
    end

    if (WORDS_PER_LINE > 1) begin
        wire [WSEL_BITS-1:0] rsp_wsel = mem_rsp_tag_id_nc[CORE_TAG_ID_BITS +: WSEL_BITS];
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_in_data[i] = core_bus_out_if[i].rsp_valid ?
                core_bus_out_if[i].rsp_data.data : mem_bus_out_if.rsp_data.data[rsp_wsel * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
        end
    end else begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_in_data[i] = core_bus_out_if[i].rsp_valid ? core_bus_out_if[i].rsp_data.data : mem_bus_out_if.rsp_data.data;
        end
    end

    wire [(CORE_TAG_ID_BITS + UUID_WIDTH)-1:0] mem_rsp_tag_in_nc2;
    if (UUID_WIDTH != 0) begin
        assign mem_rsp_tag_in_nc2 = {mem_rsp_tag_id_nc[(MEM_TAG_OUT_WIDTH - NC_ENABLE)-1 -: UUID_WIDTH], mem_rsp_tag_id_nc[CORE_TAG_ID_BITS-1:0]};
    end else begin
        assign mem_rsp_tag_in_nc2 = mem_rsp_tag_id_nc[CORE_TAG_ID_BITS-1:0];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (PASSTHRU) begin
            assign core_rsp_in_tag[i] = mem_rsp_tag_in_nc2;
        end else if (NC_ENABLE) begin
            assign core_rsp_in_tag[i] = core_bus_out_if[i].rsp_valid ? core_bus_out_if[i].rsp_data.tag : mem_rsp_tag_in_nc2;
        end else begin
            assign core_rsp_in_tag[i] = core_bus_out_if[i].rsp_data.tag;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin

        `RESET_RELAY (core_rsp_reset, reset);

        VX_elastic_buffer #(
            .DATAW   (`CS_WORD_WIDTH + CORE_TAG_WIDTH),
            .SIZE    ((!DIRECT_PASSTHRU) ? `TO_OUT_BUF_SIZE(CORE_OUT_BUF) : 0),
            .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
        ) core_rsp_buf (
            .clk       (clk),
            .reset     (core_rsp_reset),
            .valid_in  (core_rsp_in_valid[i]),
            .ready_in  (core_rsp_in_ready[i]),
            .data_in   ({core_rsp_in_data[i], core_rsp_in_tag[i]}),
            .data_out  ({core_bus_in_if[i].rsp_data.data, core_bus_in_if[i].rsp_data.tag}),
            .valid_out (core_bus_in_if[i].rsp_valid),
            .ready_out (core_bus_in_if[i].rsp_ready)
        );
    end

    // handle memory responses ////////////////////////////////////////////////

    if (PASSTHRU != 0) begin
        assign mem_bus_in_if.rsp_valid = 1'b0;
        assign mem_bus_in_if.rsp_data.data = '0;
        assign mem_bus_in_if.rsp_data.tag = '0;
    end else if (NC_ENABLE) begin
        assign mem_bus_in_if.rsp_valid = mem_bus_out_if.rsp_valid && ~mem_bus_out_if.rsp_data.tag[TAG_SEL_IDX];
        assign mem_bus_in_if.rsp_data.data = mem_bus_out_if.rsp_data.data;
        assign mem_bus_in_if.rsp_data.tag = mem_rsp_tag_id_nc[MEM_TAG_IN_WIDTH-1:0];
    end else begin
        assign mem_bus_in_if.rsp_valid = mem_bus_out_if.rsp_valid;
        assign mem_bus_in_if.rsp_data.data = mem_bus_out_if.rsp_data.data;
        assign mem_bus_in_if.rsp_data.tag = mem_rsp_tag_id_nc;
    end

    wire [NUM_REQS-1:0] core_rsp_out_valid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_rsp_out_valid[i] = core_bus_out_if[i].rsp_valid;
    end

    assign mem_bus_out_if.rsp_ready = is_mem_rsp_nc ? (~core_rsp_out_valid[rsp_idx] && core_rsp_in_ready[rsp_idx]) : mem_bus_in_if.rsp_ready;

endmodule
