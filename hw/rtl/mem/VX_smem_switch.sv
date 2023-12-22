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

module VX_smem_switch #(
    parameter NUM_REQS       = 1,
    parameter DATA_SIZE      = 1,
    parameter TAG_WIDTH      = 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter TAG_SEL_IDX    = 0,   
    parameter OUT_REG_REQ    = 0,
    parameter OUT_REG_RSP    = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,
    
    VX_mem_bus_if.slave     bus_in_if,
    VX_mem_bus_if.master    bus_out_if [NUM_REQS]
);  
    localparam ADDR_WIDTH    = (MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE));
    localparam DATA_WIDTH    = (8 * DATA_SIZE);
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS);
    localparam TAG_OUT_WIDTH = TAG_WIDTH - LOG_NUM_REQS;
    localparam REQ_DATAW     = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW     = TAG_OUT_WIDTH + DATA_WIDTH;

    wire [NUM_REQS-1:0]                req_valid_out;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_REQS-1:0]                req_ready_out;

    wire [REQ_DATAW-1:0]         req_data_in;
    wire [TAG_OUT_WIDTH-1:0]     req_tag_in;
    wire [`UP(LOG_NUM_REQS)-1:0] req_sel_in;
    
    VX_bits_remove #( 
        .N   (TAG_WIDTH),
        .S   (LOG_NUM_REQS),
        .POS (TAG_SEL_IDX)
    ) bits_remove (
        .data_in  (bus_in_if.req_data.tag),
        .data_out (req_tag_in)
    );            

    if (NUM_REQS > 1) begin
        assign req_sel_in = bus_in_if.req_data.tag[TAG_SEL_IDX +: LOG_NUM_REQS];
    end else begin
        assign req_sel_in = '0;
    end

    assign req_data_in = {req_tag_in, bus_in_if.req_data.addr, bus_in_if.req_data.rw, bus_in_if.req_data.byteen, bus_in_if.req_data.data};

    VX_stream_switch #(
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW),
        .OUT_REG     (OUT_REG_REQ)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (req_sel_in),
        .valid_in  (bus_in_if.req_valid),
        .ready_in  (bus_in_if.req_ready),
        .data_in   (req_data_in),
        .data_out  (req_data_out),
        .valid_out (req_valid_out),
        .ready_out (req_ready_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign {bus_out_if[i].req_data.tag, bus_out_if[i].req_data.addr, bus_out_if[i].req_data.rw, bus_out_if[i].req_data.byteen, bus_out_if[i].req_data.data} = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

    ///////////////////////////////////////////////////////////////////////        

    wire [NUM_REQS-1:0]                rsp_valid_out;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_out;
    wire [NUM_REQS-1:0]                rsp_ready_out;
    wire [RSP_DATAW-1:0]               rsp_data_in;
    wire [TAG_OUT_WIDTH-1:0]           rsp_tag_in;
    wire [`UP(LOG_NUM_REQS)-1:0]       rsp_sel_in;
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign rsp_valid_out[i] = bus_out_if[i].rsp_valid;
        assign rsp_data_out[i] = {bus_out_if[i].rsp_data.tag, bus_out_if[i].rsp_data.data};
        assign bus_out_if[i].rsp_ready = rsp_ready_out[i];
    end

    VX_stream_arb #(            
        .NUM_INPUTS (NUM_REQS),
        .DATAW      (RSP_DATAW),        
        .ARBITER    (ARBITER),
        .OUT_REG    (OUT_REG_RSP)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_out),        
        .ready_in  (rsp_ready_out),
        .data_in   (rsp_data_out),     
        .data_out  (rsp_data_in),
        .sel_out   (rsp_sel_in),
        .valid_out (bus_in_if.rsp_valid),
        .ready_out (bus_in_if.rsp_ready)
    );

    VX_bits_insert #( 
        .N   (TAG_OUT_WIDTH),
        .S   (LOG_NUM_REQS),
        .POS (TAG_SEL_IDX)
    ) bits_insert (
        .data_in  (rsp_tag_in),
        .sel_in   (rsp_sel_in),
        .data_out (bus_in_if.rsp_data.tag)
    );

    assign {rsp_tag_in, bus_in_if.rsp_data.data} = rsp_data_in;

endmodule
