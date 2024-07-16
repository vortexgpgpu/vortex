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

`TRACING_OFF
module VX_mem_coalescer #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS      = 1,
    parameter ADDR_WIDTH    = 32,
    parameter ATYPE_WIDTH   = 1,
    parameter DATA_IN_SIZE  = 4,
    parameter DATA_OUT_SIZE = 64,
    parameter TAG_WIDTH     = 8,
    parameter UUID_WIDTH    = 0, // upper section of the request tag contains the UUID
    parameter QUEUE_SIZE    = 8,
    
    parameter DATA_IN_WIDTH = DATA_IN_SIZE * 8,
    parameter DATA_OUT_WIDTH= DATA_OUT_SIZE * 8,
    parameter OUT_REQS      = (NUM_REQS * DATA_IN_WIDTH) / DATA_OUT_WIDTH,
    parameter BATCH_SIZE    = DATA_OUT_SIZE / DATA_IN_SIZE,
    parameter BATCH_SIZE_W  = `LOG2UP(BATCH_SIZE),
    parameter OUT_ADDR_WIDTH= ADDR_WIDTH - BATCH_SIZE_W,
    parameter QUEUE_ADDRW   = `CLOG2(QUEUE_SIZE),
    parameter OUT_TAG_WIDTH = UUID_WIDTH + QUEUE_ADDRW
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                          in_req_valid,
    input wire                          in_req_rw,
    input wire [NUM_REQS-1:0]           in_req_mask,
    input wire [NUM_REQS-1:0][DATA_IN_SIZE-1:0] in_req_byteen,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] in_req_addr,
    input wire [NUM_REQS-1:0][ATYPE_WIDTH-1:0] in_req_atype,
    input wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_req_data,
    input wire [TAG_WIDTH-1:0]          in_req_tag,    
    output wire                         in_req_ready,

    // Input response
    output wire                         in_rsp_valid,
    output wire [NUM_REQS-1:0]          in_rsp_mask,
    output wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_rsp_data,
    output wire [TAG_WIDTH-1:0]         in_rsp_tag,
    input wire                          in_rsp_ready,

    // Output request
    output wire                         out_req_valid,
    output wire                         out_req_rw,
    output wire [OUT_REQS-1:0]          out_req_mask,    
    output wire [OUT_REQS-1:0][DATA_OUT_SIZE-1:0] out_req_byteen,
    output wire [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr,
    output wire [OUT_REQS-1:0][ATYPE_WIDTH-1:0] out_req_atype,
    output wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_req_data,
    output wire [OUT_TAG_WIDTH-1:0]     out_req_tag,
    input wire 	                        out_req_ready,

    // Output response
    input wire                          out_rsp_valid,
    input wire [OUT_REQS-1:0]           out_rsp_mask,
    input wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_rsp_data,
    input wire [OUT_TAG_WIDTH-1:0]      out_rsp_tag,
    output wire                         out_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT (`IS_DIVISBLE(NUM_REQS * DATA_IN_WIDTH, DATA_OUT_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT ((NUM_REQS * DATA_IN_WIDTH >= DATA_OUT_WIDTH), ("invalid parameter"))
    `RUNTIME_ASSERT ((~in_req_valid || in_req_mask != 0), ("invalid request mask"));
    `RUNTIME_ASSERT ((~out_rsp_valid || out_rsp_mask != 0), ("invalid request mask"));
    
    localparam TAG_ID_WIDTH    = TAG_WIDTH - UUID_WIDTH;
    localparam NUM_REQS_W      = `LOG2UP(NUM_REQS);
    //                           tag          + mask     + offest
    localparam IBUF_DATA_WIDTH = TAG_ID_WIDTH + NUM_REQS + (NUM_REQS * BATCH_SIZE_W);

    localparam STATE_SETUP = 0;
    localparam STATE_SEND  = 1;
    
    logic state_r, state_n;
    
    logic out_req_valid_r, out_req_valid_n;
    logic out_req_rw_r, out_req_rw_n;
    logic [OUT_REQS-1:0] out_req_mask_r, out_req_mask_n;
    logic [OUT_REQS-1:0][DATA_OUT_SIZE-1:0] out_req_byteen_r, out_req_byteen_n;
    logic [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr_r, out_req_addr_n;
    logic [OUT_REQS-1:0][ATYPE_WIDTH-1:0] out_req_atype_r, out_req_atype_n;
    logic [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_req_data_r, out_req_data_n;
    logic [OUT_TAG_WIDTH-1:0] out_req_tag_r, out_req_tag_n;

    logic in_req_ready_n;    

    wire                        ibuf_push;
    wire                        ibuf_pop;
    wire [QUEUE_ADDRW-1:0]      ibuf_waddr;
    wire [QUEUE_ADDRW-1:0]      ibuf_raddr;
    wire                        ibuf_full;
    wire                        ibuf_empty;
    wire [IBUF_DATA_WIDTH-1:0]  ibuf_din;
    wire [IBUF_DATA_WIDTH-1:0]  ibuf_dout;
    
    logic [OUT_REQS-1:0] batch_valid_r, batch_valid_n;
    logic [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] seed_addr_r, seed_addr_n;
    logic [OUT_REQS-1:0][ATYPE_WIDTH-1:0] seed_atype_r, seed_atype_n;
    logic [NUM_REQS-1:0] processed_mask_r, processed_mask_n;

    wire [OUT_REQS-1:0][NUM_REQS_W-1:0] seed_idx;

    wire [NUM_REQS-1:0][OUT_ADDR_WIDTH-1:0] in_addr_base;
    wire [NUM_REQS-1:0][BATCH_SIZE_W-1:0] in_addr_offset;
    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign in_addr_base[i] = in_req_addr[i][ADDR_WIDTH-1:BATCH_SIZE_W];
        assign in_addr_offset[i] = in_req_addr[i][BATCH_SIZE_W-1:0];
    end

    for (genvar i = 0; i < OUT_REQS; ++i) begin
        wire [BATCH_SIZE-1:0] batch_mask = in_req_mask[BATCH_SIZE * i +: BATCH_SIZE] & ~processed_mask_r[BATCH_SIZE * i +: BATCH_SIZE];
        wire [BATCH_SIZE_W-1:0] batch_idx;
        VX_priority_encoder #(
            .N (BATCH_SIZE)
        ) priority_encoder (
            .data_in (batch_mask),
            .index (batch_idx),
            `UNUSED_PIN (onehot),
            .valid_out (batch_valid_n[i])
        );
        assign seed_idx[i] = NUM_REQS_W'(BATCH_SIZE * i) + NUM_REQS_W'(batch_idx);
    end

    always @(posedge clk) begin
        if (reset) begin
            state_r          <= STATE_SETUP;
            processed_mask_r <= '0;
            out_req_valid_r  <= 0;
        end else begin
            state_r          <= state_n;
            out_req_valid_r  <= out_req_valid_n;
            batch_valid_r    <= batch_valid_n;
            seed_addr_r      <= seed_addr_n;
            seed_atype_r     <= seed_atype_n;  
            out_req_rw_r     <= out_req_rw_n; 
            out_req_mask_r   <= out_req_mask_n;     
            out_req_addr_r   <= out_req_addr_n;
            out_req_atype_r  <= out_req_atype_n;
            out_req_byteen_r <= out_req_byteen_n;
            out_req_data_r   <= out_req_data_n;
            out_req_tag_r    <= out_req_tag_n;
            processed_mask_r <= processed_mask_n;
        end
    end

    logic [NUM_REQS-1:0] addr_matches;

    always @(*) begin
        addr_matches = '0;
        for (integer i = 0; i < OUT_REQS; ++i) begin
            for (integer j = 0; j < BATCH_SIZE; j++) begin
                if (in_addr_base[BATCH_SIZE * i + j] == seed_addr_r[i]) begin
                    addr_matches[BATCH_SIZE * i + j] = 1;
                end
            end
        end
    end

    wire [NUM_REQS-1:0] current_pmask = in_req_mask & addr_matches;

    always @(*) begin
        state_n          = state_r;
        out_req_valid_n  = out_req_valid_r;
        seed_addr_n      = seed_addr_r;
        seed_atype_n     = seed_atype_r;
        out_req_rw_n     = out_req_rw_r;      
        out_req_mask_n   = out_req_mask_r;     
        out_req_addr_n   = out_req_addr_r;
        out_req_atype_n  = out_req_atype_r;
        out_req_byteen_n = out_req_byteen_r;
        out_req_data_n   = out_req_data_r;
        out_req_tag_n    = out_req_tag_r;
        processed_mask_n = processed_mask_r;
        in_req_ready_n   = 0;

        case (state_r)
        STATE_SETUP: begin            
            // find the next seed address
            for (integer i = 0; i < OUT_REQS; ++i) begin
                seed_addr_n[i] = in_addr_base[seed_idx[i]];
                seed_atype_n[i] = in_req_atype[seed_idx[i]];
            end
            // wait for pending outgoing request to submit
            if (out_req_valid && out_req_ready) begin
                out_req_valid_n = 0;
            end
            if (in_req_valid && ~out_req_valid_n && ~ibuf_full) begin                
                state_n = STATE_SEND;
            end
        end
        default/*STATE_SEND*/: begin
            out_req_valid_n = 1;
            out_req_rw_n = in_req_rw;
            out_req_tag_n = {in_req_tag[TAG_WIDTH-1 -: UUID_WIDTH], ibuf_waddr};
            in_req_ready_n = 1;
            out_req_byteen_n = '0;
            out_req_data_n = 'x;
            for (integer i = 0; i < OUT_REQS; ++i) begin
                for (integer j = 0; j < BATCH_SIZE; j++) begin
                    if (in_req_mask[BATCH_SIZE * i + j]) begin
                        if (addr_matches[BATCH_SIZE * i + j]) begin
                            for (integer k = 0; k < DATA_IN_SIZE; ++k) begin
                                if (in_req_byteen[BATCH_SIZE * i + j][k]) begin
                                    out_req_byteen_n[i][in_addr_offset[BATCH_SIZE * i + j] * DATA_IN_SIZE + k +: 1] = 1'b1;
                                    out_req_data_n[i][in_addr_offset[BATCH_SIZE * i + j] * DATA_IN_WIDTH + k * 8 +: 8] = in_req_data[BATCH_SIZE * i + j][k * 8 +: 8];
                                end
                            end
                        end else begin
                            if (!processed_mask_r[BATCH_SIZE * i + j]) begin
                                in_req_ready_n = 0;    
                            end
                        end
                    end
                end
                out_req_mask_n[i] = batch_valid_r[i];
                out_req_addr_n[i] = seed_addr_r[i];
                out_req_atype_n[i]= seed_atype_r[i];
            end
            if (in_req_ready_n) begin
                processed_mask_n = '0;
            end else begin
                processed_mask_n = processed_mask_r | current_pmask;
            end
            state_n = STATE_SETUP;
        end
        endcase
    end

    wire out_rsp_fire = out_rsp_valid && out_rsp_ready;

    wire out_rsp_eop;

    assign ibuf_push  = (state_r == STATE_SEND) && ~in_req_rw;
    assign ibuf_pop   = out_rsp_fire && out_rsp_eop;
    assign ibuf_raddr = out_rsp_tag[QUEUE_ADDRW-1:0];    

    wire [TAG_ID_WIDTH-1:0] ibuf_din_tag = in_req_tag[TAG_ID_WIDTH-1:0];
    wire [NUM_REQS-1:0][BATCH_SIZE_W-1:0] ibuf_din_offset = in_addr_offset;
    wire [NUM_REQS-1:0] ibuf_din_pmask = current_pmask;    

    assign ibuf_din = {ibuf_din_tag, ibuf_din_pmask, ibuf_din_offset};

    VX_index_buffer #(
        .DATAW (IBUF_DATA_WIDTH),
        .SIZE  (QUEUE_SIZE)
    ) req_ibuf (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (ibuf_push),
        .write_addr   (ibuf_waddr),
        .write_data   (ibuf_din),
        .read_data    (ibuf_dout),
        .read_addr    (ibuf_raddr),
        .release_en   (ibuf_pop),
        .full         (ibuf_full),
        .empty        (ibuf_empty)
    );
    `UNUSED_VAR (ibuf_empty)

    assign out_req_valid  = out_req_valid_r;
    assign out_req_rw     = out_req_rw_r;
    assign out_req_mask   = out_req_mask_r;
    assign out_req_byteen = out_req_byteen_r;
    assign out_req_addr   = out_req_addr_r;
    assign out_req_atype  = out_req_atype_r;
    assign out_req_data   = out_req_data_r;
    assign out_req_tag    = out_req_tag_r;

    assign in_req_ready = in_req_ready_n;

    // unmerge responses

    reg [QUEUE_SIZE-1:0][OUT_REQS-1:0] rsp_rem_mask;    
    wire [OUT_REQS-1:0] rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr] & ~out_rsp_mask;
    assign out_rsp_eop = ~(| rsp_rem_mask_n);

    always @(posedge clk) begin
        if (ibuf_push) begin
            rsp_rem_mask[ibuf_waddr] <= batch_valid_r;
        end
        if (out_rsp_fire) begin
            rsp_rem_mask[ibuf_raddr] <= rsp_rem_mask_n;
        end
    end

    wire [NUM_REQS-1:0][BATCH_SIZE_W-1:0] ibuf_dout_offset;
    reg [NUM_REQS-1:0] ibuf_dout_pmask;
    wire [TAG_ID_WIDTH-1:0] ibuf_dout_tag;

    assign {ibuf_dout_tag, ibuf_dout_pmask, ibuf_dout_offset} = ibuf_dout;
    
    logic [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_rsp_data_n;
    logic [NUM_REQS-1:0] in_rsp_mask_n;

    always @(*) begin
        for (integer i = 0; i < OUT_REQS; ++i) begin
            for (integer j = 0; j < BATCH_SIZE; j++) begin
                in_rsp_mask_n[BATCH_SIZE * i + j] = out_rsp_mask[i] && ibuf_dout_pmask[BATCH_SIZE * i + j];
                in_rsp_data_n[BATCH_SIZE * i + j] = out_rsp_data[i][ibuf_dout_offset[BATCH_SIZE * i + j] * DATA_IN_WIDTH +: DATA_IN_WIDTH];
            end
        end
    end

    assign in_rsp_valid  = out_rsp_valid;
    assign in_rsp_mask   = in_rsp_mask_n;
    assign in_rsp_data   = in_rsp_data_n;
    assign in_rsp_tag    = {out_rsp_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH], ibuf_dout_tag};
    assign out_rsp_ready = in_rsp_ready;

`ifdef DBG_TRACE_MEM
    wire [`UP(UUID_WIDTH)-1:0] out_req_uuid;
    wire [`UP(UUID_WIDTH)-1:0] out_rsp_uuid;

    if (UUID_WIDTH != 0) begin
        assign out_req_uuid = out_req_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH];
        assign out_rsp_uuid = out_rsp_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign out_req_uuid = '0;
        assign out_rsp_uuid = '0;
    end

    reg [NUM_REQS-1:0][BATCH_SIZE_W-1:0] out_req_offset;
    reg [NUM_REQS-1:0] out_req_pmask;

    always @(posedge clk) begin
        if (ibuf_push) begin            
            out_req_offset <= ibuf_din_offset;
            out_req_pmask <= ibuf_din_pmask;
        end
    end

    wire out_req_fire = out_req_valid && out_req_ready;

    always @(posedge clk) begin
        if (out_req_fire) begin
            if (out_req_rw) begin
                `TRACE(1, ("%d: %s-out-req-wr: valid=%b, addr=", $time, INSTANCE_ID, out_req_mask));
                `TRACE_ARRAY1D(1, "0x%h", out_req_addr, OUT_REQS);     
                `TRACE(1, (", atype="));
                `TRACE_ARRAY1D(1, "%b", out_req_atype, OUT_REQS);                  
                `TRACE(1, (", byteen="));
                `TRACE_ARRAY1D(1, "0x%h", out_req_byteen, OUT_REQS);
                `TRACE(1, (", data="));
                `TRACE_ARRAY1D(1, "0x%0h", out_req_data, OUT_REQS);         
            end else begin
                `TRACE(1, ("%d: %s-out-req-rd: valid=%b, addr=", $time, INSTANCE_ID, out_req_mask));
                `TRACE_ARRAY1D(1, "0x%h", out_req_addr, OUT_REQS);
                `TRACE(1, (", atype="));
                `TRACE_ARRAY1D(1, "%b", out_req_atype, OUT_REQS);
            end
            `TRACE(1, (", offset=")); 
            `TRACE_ARRAY1D(1, "%0d", out_req_offset, NUM_REQS);
            `TRACE(1, (", pmask=%b, tag=0x%0h (#%0d)\n", out_req_pmask, out_req_tag, out_req_uuid));       
            if ($countones(out_req_pmask) > 1) begin
                `TRACE(1, ("%t: *** %s: coalescing=%b (#%0d)\n", $time, INSTANCE_ID, out_req_pmask, out_req_uuid)); 
            end    
        end
        if (out_rsp_fire) begin
            `TRACE(1, ("%d: %s-out-rsp: valid=%b, data=", $time, INSTANCE_ID, out_rsp_mask));
            `TRACE_ARRAY1D(1, "0x%0h", out_rsp_data, OUT_REQS);
            `TRACE(1, (", offset=")); 
            `TRACE_ARRAY1D(1, "%0d", ibuf_dout_offset, NUM_REQS);
            `TRACE(1, (", eop=%b, pmask=%b, tag=0x%0h (#%0d)\n", out_rsp_eop, ibuf_dout_pmask, out_rsp_tag, out_rsp_uuid));
        end
    end
`endif

endmodule
`TRACING_ON
