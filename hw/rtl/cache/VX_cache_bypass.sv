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

    parameter CORE_ADDR_WIDTH   = 1,
    parameter CORE_DATA_SIZE    = 1, 
    parameter CORE_TAG_IN_WIDTH = 1,
    
    parameter MEM_ADDR_WIDTH    = 1,
    parameter MEM_DATA_SIZE     = 1,
    parameter MEM_TAG_IN_WIDTH  = 1,
    parameter MEM_TAG_OUT_WIDTH = 1,

    parameter UUID_WIDTH        = 0,
 
    parameter CORE_DATA_WIDTH   = CORE_DATA_SIZE * 8,
    parameter MEM_DATA_WIDTH    = MEM_DATA_SIZE * 8,
    parameter CORE_TAG_OUT_WIDTH= CORE_TAG_IN_WIDTH - NC_ENABLE
 ) ( 
    input wire clk,
    input wire reset,

    // Core request in   
    input wire [NUM_REQS-1:0]                       core_req_valid_in,
    input wire [NUM_REQS-1:0]                       core_req_rw_in,
    input wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0]  core_req_addr_in,
    input wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]   core_req_byteen_in,
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_req_data_in,
    input wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_req_tag_in,
    output wire [NUM_REQS-1:0]                      core_req_ready_in,

    // Core request out
    output wire [NUM_REQS-1:0]                      core_req_valid_out,
    output wire [NUM_REQS-1:0]                      core_req_rw_out,
    output wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0] core_req_addr_out,
    output wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]  core_req_byteen_out,
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_req_data_out,
    output wire [NUM_REQS-1:0][CORE_TAG_OUT_WIDTH-1:0] core_req_tag_out,
    input wire [NUM_REQS-1:0]                       core_req_ready_out,

    // Core response in
    input wire [NUM_REQS-1:0]                       core_rsp_valid_in,
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_rsp_data_in,
    input wire [NUM_REQS-1:0][CORE_TAG_OUT_WIDTH-1:0] core_rsp_tag_in,
    output  wire [NUM_REQS-1:0]                     core_rsp_ready_in,   

    // Core response out
    output wire [NUM_REQS-1:0]                      core_rsp_valid_out,
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_rsp_data_out,
    output wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_tag_out,
    input  wire [NUM_REQS-1:0]                      core_rsp_ready_out,   

    // Memory request in
    input wire                          mem_req_valid_in,
    input wire                          mem_req_rw_in,      
    input wire [MEM_ADDR_WIDTH-1:0]     mem_req_addr_in,
    input wire [MEM_DATA_SIZE-1:0]      mem_req_byteen_in,
    input wire [MEM_DATA_WIDTH-1:0]     mem_req_data_in,
    input wire [MEM_TAG_IN_WIDTH-1:0]   mem_req_tag_in,
    output  wire                        mem_req_ready_in,

    // Memory request out
    output wire                         mem_req_valid_out,
    output wire                         mem_req_rw_out,       
    output wire [MEM_ADDR_WIDTH-1:0]    mem_req_addr_out,
    output wire [MEM_DATA_SIZE-1:0]     mem_req_byteen_out, 
    output wire [MEM_DATA_WIDTH-1:0]    mem_req_data_out,
    output wire [MEM_TAG_OUT_WIDTH-1:0] mem_req_tag_out,
    input  wire                         mem_req_ready_out,
    
    // Memory response in
    input  wire                         mem_rsp_valid_in,    
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rsp_data_in,
    input  wire [MEM_TAG_OUT_WIDTH-1:0] mem_rsp_tag_in,
    output wire                         mem_rsp_ready_in,

    // Memory response out
    output  wire                        mem_rsp_valid_out,    
    output  wire [MEM_DATA_WIDTH-1:0]   mem_rsp_data_out,
    output  wire [MEM_TAG_IN_WIDTH-1:0] mem_rsp_tag_out,
    input wire                          mem_rsp_ready_out
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)    

    localparam REQ_SEL_BITS     = `CLOG2(NUM_REQS);
    localparam MUX_DATAW        = CORE_TAG_IN_WIDTH + CORE_DATA_WIDTH + CORE_DATA_SIZE + CORE_ADDR_WIDTH + 1;

    localparam WORDS_PER_LINE   = MEM_DATA_SIZE / CORE_DATA_SIZE;
    localparam WSEL_BITS        = `CLOG2(WORDS_PER_LINE);

    localparam CORE_TAG_ID_BITS = CORE_TAG_IN_WIDTH - UUID_WIDTH;
    localparam MEM_TAG_ID_BITS  = REQ_SEL_BITS + WSEL_BITS + CORE_TAG_ID_BITS;

    localparam MEM_TAG_OUT_NC_WIDTH = MEM_TAG_OUT_WIDTH - 1 + NC_ENABLE;

    // core request handling

    wire [NUM_REQS-1:0] core_req_valid_in_nc;
    wire [NUM_REQS-1:0] core_req_nc_idxs;    
    wire [`UP(REQ_SEL_BITS)-1:0] core_req_nc_idx;
    wire [NUM_REQS-1:0] core_req_nc_sel;
    wire core_req_nc_valid;    
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (PASSTHRU != 0) begin
            assign core_req_nc_idxs[i] = 1'b1;
        end else begin
            assign core_req_nc_idxs[i] = core_req_tag_in[i][NC_TAG_BIT];
        end
    end

    assign core_req_valid_in_nc = core_req_valid_in & core_req_nc_idxs;

    wire core_req_nc_ready = ~mem_req_valid_in && mem_req_ready_out;

    VX_generic_arbiter #(
        .NUM_REQS    (NUM_REQS),
        .TYPE        (PASSTHRU ? "R" : "P"),
        .LOCK_ENABLE (1)
    ) core_req_nc_arb (
        .clk          (clk),
        .reset        (reset),        
        .requests     (core_req_valid_in_nc),        
        .grant_index  (core_req_nc_idx),
        .grant_onehot (core_req_nc_sel),
        .grant_valid  (core_req_nc_valid),
        .grant_unlock (core_req_nc_ready)
    );

    assign core_req_valid_out  = core_req_valid_in & ~core_req_nc_idxs;
    assign core_req_rw_out     = core_req_rw_in;
    assign core_req_addr_out   = core_req_addr_in;
    assign core_req_byteen_out = core_req_byteen_in;
    assign core_req_data_out   = core_req_data_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_bits_remove #( 
            .N   (CORE_TAG_IN_WIDTH),
            .S   (NC_ENABLE),
            .POS (NC_TAG_BIT)
        ) core_req_tag_nc_remove (
            .data_in  (core_req_tag_in[i]),
            .data_out (core_req_tag_out[i])
        );
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_ready_in[i] = core_req_valid_in_nc[i] ? (core_req_nc_ready && core_req_nc_sel[i]) 
                                                              : core_req_ready_out[i];
    end

    // memory request handling

    assign mem_req_valid_out = mem_req_valid_in || core_req_nc_valid;
    assign mem_req_ready_in  = mem_req_ready_out;

    wire [CORE_TAG_IN_WIDTH-1:0] core_req_tag_in_sel;
    wire [CORE_DATA_WIDTH-1:0]   core_req_data_in_sel;
    wire [CORE_DATA_SIZE-1:0]    core_req_byteen_in_sel;
    wire [CORE_ADDR_WIDTH-1:0]   core_req_addr_in_sel;
    wire                         core_req_rw_in_sel;

    wire [NUM_REQS-1:0][MUX_DATAW-1:0] core_req_nc_mux_in;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_nc_mux_in[i] = {core_req_tag_in[i], core_req_data_in[i], core_req_byteen_in[i], core_req_addr_in[i], core_req_rw_in[i]};
    end
    assign {core_req_tag_in_sel, core_req_data_in_sel, core_req_byteen_in_sel, core_req_addr_in_sel, core_req_rw_in_sel} = core_req_nc_mux_in[core_req_nc_idx];

    wire [CORE_TAG_ID_BITS-1:0] core_req_in_id = core_req_tag_in_sel[CORE_TAG_ID_BITS-1:0];
      
    assign mem_req_rw_out   = mem_req_valid_in ? mem_req_rw_in : core_req_rw_in_sel;
    assign mem_req_addr_out = mem_req_valid_in ? mem_req_addr_in : core_req_addr_in_sel[WSEL_BITS +: MEM_ADDR_WIDTH];

    wire [MEM_TAG_ID_BITS-1:0] mem_req_tag_id_bypass;
    
    if (WORDS_PER_LINE > 1) begin
        reg [WORDS_PER_LINE-1:0][CORE_DATA_SIZE-1:0]  mem_req_byteen_in_r;
        reg [WORDS_PER_LINE-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_in_r;
        
        wire [WSEL_BITS-1:0] req_wsel = core_req_addr_in_sel[WSEL_BITS-1:0];

        always @(*) begin
            mem_req_byteen_in_r = '0;
            mem_req_byteen_in_r[req_wsel] = core_req_byteen_in_sel;

            mem_req_data_in_r = 'x;
            mem_req_data_in_r[req_wsel] = core_req_data_in_sel;
        end

        assign mem_req_byteen_out = mem_req_valid_in ? mem_req_byteen_in : mem_req_byteen_in_r;
        assign mem_req_data_out   = mem_req_valid_in ? mem_req_data_in : mem_req_data_in_r;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, req_wsel, core_req_in_id});
        end else begin 
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({req_wsel, core_req_in_id});
        end
    end else begin
        assign mem_req_byteen_out = mem_req_valid_in ? mem_req_byteen_in : core_req_byteen_in_sel;
        assign mem_req_data_out   = mem_req_valid_in ? mem_req_data_in : core_req_data_in_sel;
        if (NUM_REQS > 1) begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_nc_idx, core_req_in_id});
        end else begin
            assign mem_req_tag_id_bypass = MEM_TAG_ID_BITS'({core_req_in_id});
        end
    end

    wire [MEM_TAG_OUT_NC_WIDTH-1:0] mem_req_tag_bypass;   

    if (UUID_WIDTH != 0) begin
        assign mem_req_tag_bypass = {core_req_tag_in_sel[CORE_TAG_ID_BITS +: UUID_WIDTH], mem_req_tag_id_bypass};
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
        .data_in  (mem_req_tag_in),
        .sel_in   (1'b0),
        .data_out (mem_req_tag_in_nc)
    );

    assign mem_req_tag_out = mem_req_valid_in ? MEM_TAG_OUT_WIDTH'(mem_req_tag_in_nc) : mem_req_tag_bypass_nc;

    // core response handling

    wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_tag_in_nc;

    wire is_mem_rsp_nc;
    if (PASSTHRU != 0) begin
        assign is_mem_rsp_nc = mem_rsp_valid_in;
    end else begin
        assign is_mem_rsp_nc = mem_rsp_valid_in && mem_rsp_tag_in[NC_TAG_BIT];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_bits_insert #( 
            .N   (CORE_TAG_OUT_WIDTH),
            .S   (NC_ENABLE),
            .POS (NC_TAG_BIT)
        ) core_rsp_tag_in_nc_insert (
            .data_in  (core_rsp_tag_in[i]),
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
        .data_in  (mem_rsp_tag_in),
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

    assign core_rsp_valid_out = core_rsp_valid_in | rsp_nc_valid_r;
    assign core_rsp_ready_in  = core_rsp_ready_out;

    if (WORDS_PER_LINE > 1) begin
        wire [WSEL_BITS-1:0] rsp_wsel = mem_rsp_tag_in_nc[CORE_TAG_ID_BITS +: WSEL_BITS];        
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_data_out[i] = core_rsp_valid_in[i] ? 
                core_rsp_data_in[i] : mem_rsp_data_in[rsp_wsel * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
        end
    end else begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_data_out[i] = core_rsp_valid_in[i] ? core_rsp_data_in[i] : mem_rsp_data_in;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (UUID_WIDTH != 0) begin
            assign core_rsp_tag_out[i] = core_rsp_valid_in[i] ? core_rsp_tag_in_nc[i] : {mem_rsp_tag_in_nc[MEM_TAG_OUT_NC_WIDTH-1 -: UUID_WIDTH], mem_rsp_tag_in_nc[CORE_TAG_ID_BITS-1:0]};
        end else begin
            assign core_rsp_tag_out[i] = core_rsp_valid_in[i] ? core_rsp_tag_in_nc[i] : mem_rsp_tag_in_nc[CORE_TAG_ID_BITS-1:0];
        end
    end

    // memory response handling

    if (PASSTHRU != 0) begin
        assign mem_rsp_valid_out = 1'b0;
    end else begin
        assign mem_rsp_valid_out = mem_rsp_valid_in && ~mem_rsp_tag_in[NC_TAG_BIT];
    end

    assign mem_rsp_data_out  = mem_rsp_data_in;

    VX_bits_remove #( 
        .N   (MEM_TAG_IN_WIDTH + 1),
        .POS (NC_TAG_BIT)
    ) mem_rsp_tag_out_remove (
        .data_in  (mem_rsp_tag_in[(MEM_TAG_IN_WIDTH + 1)-1:0]),
        .data_out (mem_rsp_tag_out)
    );

    assign mem_rsp_ready_in = is_mem_rsp_nc ? (~core_rsp_valid_in[rsp_idx] && core_rsp_ready_out[rsp_idx]) : mem_rsp_ready_out;

endmodule
