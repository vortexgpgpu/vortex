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

`include "VX_platform.vh"

module VX_cache_bypass #(
    parameter NUM_REQS          = 1,
    parameter NC_TAG_BIT        = 0,

    parameter NC_ENABLE         = 0,
    parameter PASSTHRU          = 0,

    parameter WORD_SIZE         = 1,
    parameter LINE_SIZE         = 1, 

    parameter CORE_ADDR_WIDTH   = 1,
    
    parameter CORE_TAG_IN_WIDTH = 1,
    
    parameter MEM_ADDR_WIDTH    = 1,    
    parameter MEM_TAG_IN_WIDTH  = 1,
    parameter MEM_TAG_OUT_WIDTH = 1,

    parameter UUID_WIDTH        = 0,

    parameter CORE_OUT_REG      = 0,
    parameter MEM_OUT_REG       = 0,
 
    parameter CORE_DATA_WIDTH   = WORD_SIZE * 8,
    parameter CORE_TAG_OUT_WIDTH = CORE_TAG_IN_WIDTH - NC_ENABLE
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
    localparam MUX_DATAW        = CORE_TAG_IN_WIDTH + CORE_DATA_WIDTH + WORD_SIZE + CORE_ADDR_WIDTH + 1;

    localparam WORDS_PER_LINE   = LINE_SIZE / WORD_SIZE;
    localparam WSEL_BITS        = `CLOG2(WORDS_PER_LINE);

    localparam CORE_TAG_ID_BITS = CORE_TAG_IN_WIDTH - UUID_WIDTH;
    localparam MEM_TAG_ID_BITS  = REQ_SEL_BITS + WSEL_BITS + CORE_TAG_ID_BITS;

    localparam MEM_TAG_OUT_NC_WIDTH = MEM_TAG_OUT_WIDTH - 1 + NC_ENABLE;

    // core request handling

    wire [NUM_REQS-1:0] core_req_nc_valids;
    wire [NUM_REQS-1:0] core_req_nc_idxs;    
    wire [`UP(REQ_SEL_BITS)-1:0] core_req_nc_idx;
    wire [NUM_REQS-1:0] core_req_nc_sel;
    wire core_req_nc_valid;    
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (PASSTHRU != 0) begin
            assign core_req_nc_idxs[i] = 1'b1;
        end else begin
            assign core_req_nc_idxs[i] = core_bus_in_if[i].req_data.tag[NC_TAG_BIT];
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin    
        assign core_req_nc_valids[i] = core_bus_in_if[i].req_valid && core_req_nc_idxs[i];
    end   

    wire core_req_nc_ready = ~mem_bus_in_if.req_valid && mem_req_out_ready;

    VX_generic_arbiter #(
        .NUM_REQS    (NUM_REQS),
        .TYPE        (PASSTHRU ? "R" : "P"),
        .LOCK_ENABLE (1)
    ) core_req_nc_arb (
        .clk          (clk),
        .reset        (reset),        
        .requests     (core_req_nc_valids),        
        .grant_index  (core_req_nc_idx),
        .grant_onehot (core_req_nc_sel),
        .grant_valid  (core_req_nc_valid),
        .grant_unlock (core_req_nc_ready)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && ~core_req_nc_idxs[i];
        assign core_bus_out_if[i].req_data.rw = core_bus_in_if[i].req_data.rw;
        assign core_bus_out_if[i].req_data.addr = core_bus_in_if[i].req_data.addr;
        assign core_bus_out_if[i].req_data.byteen = core_bus_in_if[i].req_data.byteen;        
        assign core_bus_out_if[i].req_data.data = core_bus_in_if[i].req_data.data;

        VX_bits_remove #( 
            .N   (CORE_TAG_IN_WIDTH),
            .S   (NC_ENABLE),
            .POS (NC_TAG_BIT)
        ) core_req_tag_nc_remove (
            .data_in  (core_bus_in_if[i].req_data.tag),
            .data_out (core_bus_out_if[i].req_data.tag)
        );

        assign core_bus_in_if[i].req_ready = core_req_nc_valids[i] ? (core_req_nc_ready && core_req_nc_sel[i]) 
                                                                   : core_bus_out_if[i].req_ready;
    end

    // memory request handling

    wire                          mem_req_out_valid;
    wire                          mem_req_out_rw;
    wire [LINE_SIZE-1:0]          mem_req_out_byteen;
    wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_out_addr;
    wire [`CS_LINE_WIDTH-1:0]     mem_req_out_data;
    wire [MEM_TAG_OUT_WIDTH-1:0]  mem_req_out_tag;
    wire                          mem_req_out_ready;

    wire [CORE_TAG_IN_WIDTH-1:0] core_req_nc_sel_tag;
    wire [CORE_DATA_WIDTH-1:0]   core_req_nc_sel_data;
    wire [WORD_SIZE-1:0]    core_req_nc_sel_byteen;
    wire [CORE_ADDR_WIDTH-1:0]   core_req_nc_sel_addr;
    wire                         core_req_nc_sel_rw;

    wire [NUM_REQS-1:0][MUX_DATAW-1:0] core_req_nc_mux_in;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_nc_mux_in[i] = {
            core_bus_in_if[i].req_data.tag, 
            core_bus_in_if[i].req_data.data, 
            core_bus_in_if[i].req_data.byteen, 
            core_bus_in_if[i].req_data.addr, 
            core_bus_in_if[i].req_data.rw
        };
    end
    
    assign {
        core_req_nc_sel_tag, 
        core_req_nc_sel_data, 
        core_req_nc_sel_byteen, 
        core_req_nc_sel_addr, 
        core_req_nc_sel_rw
    } = core_req_nc_mux_in[core_req_nc_idx];

    assign mem_req_out_valid = mem_bus_in_if.req_valid || core_req_nc_valid;
    assign mem_req_out_rw = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.rw : core_req_nc_sel_rw;
    assign mem_req_out_addr = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.addr : core_req_nc_sel_addr[WSEL_BITS +: MEM_ADDR_WIDTH];

    wire [MEM_TAG_ID_BITS-1:0] mem_req_tag_id_bypass;

    wire [CORE_TAG_ID_BITS-1:0] core_req_in_id = core_req_nc_sel_tag[CORE_TAG_ID_BITS-1:0];
    
    if (WORDS_PER_LINE > 1) begin
        reg [WORDS_PER_LINE-1:0][WORD_SIZE-1:0]  mem_req_byteen_in_r;
        reg [WORDS_PER_LINE-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_in_r;
        
        wire [WSEL_BITS-1:0] req_wsel = core_req_nc_sel_addr[WSEL_BITS-1:0];

        always @(*) begin
            mem_req_byteen_in_r = '0;
            mem_req_byteen_in_r[req_wsel] = core_req_nc_sel_byteen;

            mem_req_data_in_r = 'x;
            mem_req_data_in_r[req_wsel] = core_req_nc_sel_data;
        end

        assign mem_req_out_byteen = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.byteen : mem_req_byteen_in_r;
        assign mem_req_out_data   = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.data : mem_req_data_in_r;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, req_wsel, core_req_in_id});
        end else begin 
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({req_wsel, core_req_in_id});
        end
    end else begin
        assign mem_req_out_byteen = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.byteen : core_req_nc_sel_byteen;
        assign mem_req_out_data   = mem_bus_in_if.req_valid ? mem_bus_in_if.req_data.data : core_req_nc_sel_data;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, core_req_in_id});
        end else begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_in_id});
        end
    end

    wire [MEM_TAG_OUT_NC_WIDTH-1:0] mem_req_tag_bypass;   

    if (UUID_WIDTH != 0) begin
        assign mem_req_tag_bypass = {core_req_nc_sel_tag[CORE_TAG_ID_BITS +: UUID_WIDTH], mem_req_tag_id_bypass};
    end else begin
        assign mem_req_tag_bypass = mem_req_tag_id_bypass;
    end

    wire [MEM_TAG_OUT_WIDTH-1:0]      mem_req_tag_bypass_nc;
    wire [(MEM_TAG_IN_WIDTH + 1)-1:0] mem_req_tag_in_nc;

    VX_bits_insert #( 
        .N   (MEM_TAG_OUT_NC_WIDTH),
        .S   (NC_ENABLE ? 0 : 1),
        .POS (NC_TAG_BIT)
    ) mem_req_tag_bypass_nc_insert (
        .data_in  (mem_req_tag_bypass),
        .sel_in   (1'b0),
        .data_out (mem_req_tag_bypass_nc)
    );    

    VX_bits_insert #( 
        .N   (MEM_TAG_IN_WIDTH),
        .POS (NC_TAG_BIT)
    ) mem_req_tag_in_nc_insert (
        .data_in  (mem_bus_in_if.req_data.tag),
        .sel_in   (1'b0),
        .data_out (mem_req_tag_in_nc)
    );

    assign mem_req_out_tag = mem_bus_in_if.req_valid ? MEM_TAG_OUT_WIDTH'(mem_req_tag_in_nc) : mem_req_tag_bypass_nc;

    assign mem_bus_in_if.req_ready = mem_req_out_ready;

    VX_elastic_buffer #(
        .DATAW   (1 + LINE_SIZE + `CS_MEM_ADDR_WIDTH + `CS_LINE_WIDTH + MEM_TAG_OUT_WIDTH),
        .SIZE    ((!DIRECT_PASSTHRU) ? `OUT_REG_TO_EB_SIZE(MEM_OUT_REG) : 0),
        .OUT_REG (`OUT_REG_TO_EB_REG(MEM_OUT_REG))
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_out_valid), 
        .ready_in  (mem_req_out_ready), 
        .data_in   ({mem_req_out_rw, mem_req_out_byteen, mem_req_out_addr, mem_req_out_data, mem_req_out_tag}),
        .data_out  ({mem_bus_out_if.req_data.rw, mem_bus_out_if.req_data.byteen, mem_bus_out_if.req_data.addr, mem_bus_out_if.req_data.data, mem_bus_out_if.req_data.tag}), 
        .valid_out (mem_bus_out_if.req_valid), 
        .ready_out (mem_bus_out_if.req_ready)
    );

    // core response handling

    wire [NUM_REQS-1:0]                  core_rsp_in_valid;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_in_data;
    wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_in_tag;
    wire [NUM_REQS-1:0]                  core_rsp_in_ready;

    wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_tag_in_nc;

    wire is_mem_rsp_nc;
    if (PASSTHRU != 0) begin
        assign is_mem_rsp_nc = mem_bus_out_if.rsp_valid;
    end else begin
        assign is_mem_rsp_nc = mem_bus_out_if.rsp_valid && mem_bus_out_if.rsp_data.tag[NC_TAG_BIT];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_bits_insert #( 
            .N   (CORE_TAG_OUT_WIDTH),
            .S   (NC_ENABLE),
            .POS (NC_TAG_BIT)
        ) core_rsp_tag_in_nc_insert (
            .data_in  (core_bus_out_if[i].rsp_data.tag),
            .sel_in   ('0),
            .data_out (core_rsp_tag_in_nc[i])
        );
    end

    wire [MEM_TAG_OUT_NC_WIDTH-1:0] mem_rsp_tag_in_nc;

    VX_bits_remove #( 
        .N   (MEM_TAG_OUT_WIDTH),
        .S   (NC_ENABLE ? 0 : 1),
        .POS (NC_TAG_BIT)
    ) mem_rsp_tag_in_nc_remove (
        .data_in  (mem_bus_out_if.rsp_data.tag),
        .data_out (mem_rsp_tag_in_nc)
    );

    wire [`UP(REQ_SEL_BITS)-1:0] rsp_idx;
    if (NUM_REQS > 1) begin
        assign rsp_idx = mem_rsp_tag_in_nc[(CORE_TAG_ID_BITS + WSEL_BITS) +: REQ_SEL_BITS];
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
        wire [WSEL_BITS-1:0] rsp_wsel = mem_rsp_tag_in_nc[CORE_TAG_ID_BITS +: WSEL_BITS];        
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_in_data[i] = core_bus_out_if[i].rsp_valid ? 
                core_bus_out_if[i].rsp_data.data : mem_bus_out_if.rsp_data.data[rsp_wsel * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
        end
    end else begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_in_data[i] = core_bus_out_if[i].rsp_valid ? core_bus_out_if[i].rsp_data.data : mem_bus_out_if.rsp_data.data;
        end
    end  

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (UUID_WIDTH != 0) begin
            assign core_rsp_in_tag[i] = core_bus_out_if[i].rsp_valid ? core_rsp_tag_in_nc[i] : {mem_rsp_tag_in_nc[MEM_TAG_OUT_NC_WIDTH-1 -: UUID_WIDTH], mem_rsp_tag_in_nc[CORE_TAG_ID_BITS-1:0]};
        end else begin
            assign core_rsp_in_tag[i] = core_bus_out_if[i].rsp_valid ? core_rsp_tag_in_nc[i] : mem_rsp_tag_in_nc[CORE_TAG_ID_BITS-1:0];
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_elastic_buffer #(
            .DATAW   (`CS_WORD_WIDTH + CORE_TAG_IN_WIDTH),
            .SIZE    ((!DIRECT_PASSTHRU) ? `OUT_REG_TO_EB_SIZE(CORE_OUT_REG) : 0),
            .OUT_REG (`OUT_REG_TO_EB_REG(CORE_OUT_REG))
        ) core_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_rsp_in_valid[i]),
            .ready_in  (core_rsp_in_ready[i]),
            .data_in   ({core_rsp_in_data[i], core_rsp_in_tag[i]}),
            .data_out  ({core_bus_in_if[i].rsp_data.data, core_bus_in_if[i].rsp_data.tag}), 
            .valid_out (core_bus_in_if[i].rsp_valid),
            .ready_out (core_bus_in_if[i].rsp_ready)
        );
    end

    // memory response handling

    if (PASSTHRU != 0) begin
        assign mem_bus_in_if.rsp_valid = 1'b0;
    end else begin
        assign mem_bus_in_if.rsp_valid = mem_bus_out_if.rsp_valid && ~mem_bus_out_if.rsp_data.tag[NC_TAG_BIT];
    end

    assign mem_bus_in_if.rsp_data.data  = mem_bus_out_if.rsp_data.data;

    VX_bits_remove #( 
        .N   (MEM_TAG_IN_WIDTH + 1),
        .POS (NC_TAG_BIT)
    ) mem_rsp_tag_out_remove (
        .data_in  (mem_bus_out_if.rsp_data.tag[(MEM_TAG_IN_WIDTH + 1)-1:0]),
        .data_out (mem_bus_in_if.rsp_data.tag)
    );

    wire [NUM_REQS-1:0] core_rsp_valid_in;
    wire [NUM_REQS-1:0] core_rsp_ready_out;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_rsp_valid_in[i] = core_bus_out_if[i].rsp_valid;
        assign core_rsp_ready_out[i] = core_bus_in_if[i].rsp_ready;
    end

    assign mem_bus_out_if.rsp_ready = is_mem_rsp_nc ? (~core_rsp_valid_in[rsp_idx] && core_rsp_ready_out[rsp_idx]) : mem_bus_in_if.rsp_ready;

endmodule
