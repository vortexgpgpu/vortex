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

// Command-processor DMA port into the last-level cache's client slot.
// Requests in the IO aperture carry the IO attribute and bypass the cache;
// a write flagged no-allocate writes around it on a miss.
// A launch that follows DMA writes is held until those writes are inside the
// cache banks: the port sends an L1-only flush token behind them, which the
// last-level cache answers once its input crossbar has drained, and only then
// passes the start pulse on. Reads need no such fence, since the DMA waits for
// their data.
module VX_cp_llc_port import VX_gpu_pkg::*; #(
    parameter DATA_SIZE     = 64,
    parameter SLOT_TAG_WIDTH = UUID_WIDTH + 2,
    parameter TAG_WIDTH     = SLOT_TAG_WIDTH - UUID_WIDTH - 1,
    parameter ADDR_WIDTH    = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) (
    input  wire                     clk,
    input  wire                     reset,

    input  wire                     req_valid,
    input  wire                     req_rw,
    input  wire                     req_noalloc,
    input  wire [ADDR_WIDTH-1:0]    req_addr,
    input  wire [DATA_SIZE*8-1:0]   req_data,
    input  wire [DATA_SIZE-1:0]     req_byteen,
    input  wire [TAG_WIDTH-1:0]     req_tag,
    output wire                     req_ready,

    output wire                     rsp_valid,
    output wire [DATA_SIZE*8-1:0]   rsp_data,
    output wire [TAG_WIDTH-1:0]     rsp_tag,
    input  wire                     rsp_ready,

    input  wire                     start_in,
    output wire                     start_out,
    output wire                     busy,

    VX_mem_bus_if.master            mem_bus_if
);
    `STATIC_ASSERT(TAG_WIDTH >= 1, ("invalid parameter: TAG_WIDTH=%0d", TAG_WIDTH))
    `STATIC_ASSERT(0 == (`VX_MEM_IO_BASE_ADDR % DATA_SIZE), ("invalid parameter"))

    localparam ASHIFT     = `CLOG2(DATA_SIZE);
    localparam REQ_DATAW  = 1 + ADDR_WIDTH + DATA_SIZE * 8 + DATA_SIZE + `UP(MEM_ATTR_WIDTH) + SLOT_TAG_WIDTH;

    localparam STATE_IDLE  = 0;
    localparam STATE_FENCE = 1;
    localparam STATE_WAIT  = 2;

    reg [1:0] state;
    reg       wr_pending;

    wire fence_rsp = mem_bus_if.rsp_valid && mem_bus_if.rsp_data.tag.value[0];

    wire [ADDR_WIDTH-1:0] io_addr_start = ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_IO_BASE_ADDR) >> ASHIFT);
    wire [ADDR_WIDTH-1:0] io_addr_end   = ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_IO_END_ADDR) >> ASHIFT);

    mem_bus_attr_t req_attr;
    always @(*) begin
        req_attr = '0;
        req_attr.is_addr_io = (req_addr >= io_addr_start) && (req_addr < io_addr_end);
        req_attr.is_noalloc = req_rw && req_noalloc;
    end

    wire [`UP(MEM_ATTR_WIDTH)-1:0] fence_attr = `UP(MEM_ATTR_WIDTH)'(1 << MEM_ATTR_FLUSH_OFFS);

    wire [SLOT_TAG_WIDTH-1:0] req_slot_tag   = SLOT_TAG_WIDTH'({req_tag, 1'b0});
    wire [SLOT_TAG_WIDTH-1:0] fence_slot_tag = SLOT_TAG_WIDTH'(1);

    wire fence_req = (state == STATE_FENCE);

    wire                 buf_valid_in = fence_req || ((state == STATE_IDLE) && req_valid);
    wire                 buf_ready_in;
    wire [REQ_DATAW-1:0] buf_data_in  = fence_req ?
        {1'b0, ADDR_WIDTH'(0), (DATA_SIZE*8)'(0), DATA_SIZE'(1), fence_attr, fence_slot_tag} :
        {req_rw, req_addr, req_data, req_byteen, `UP(MEM_ATTR_WIDTH)'(req_attr), req_slot_tag};

    assign req_ready = (state == STATE_IDLE) && buf_ready_in;

    wire req_fire   = req_valid && req_ready;
    wire fence_fire = fence_req && buf_ready_in;

    VX_elastic_buffer #(
        .DATAW (REQ_DATAW),
        .SIZE  (1)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (buf_valid_in),
        .ready_in  (buf_ready_in),
        .data_in   (buf_data_in),
        .data_out  ({mem_bus_if.req_data.rw,
                     mem_bus_if.req_data.addr,
                     mem_bus_if.req_data.data,
                     mem_bus_if.req_data.byteen,
                     mem_bus_if.req_data.attr,
                     mem_bus_if.req_data.tag}),
        .ready_out (mem_bus_if.req_ready),
        .valid_out (mem_bus_if.req_valid)
    );

    always @(posedge clk) begin
        if (reset) begin
            state      <= STATE_IDLE;
            wr_pending <= 1'b0;
        end else begin
            case (state)
            STATE_FENCE: begin
                if (fence_fire) begin
                    state <= STATE_WAIT;
                end
            end
            STATE_WAIT: begin
                if (fence_rsp) begin
                    state      <= STATE_IDLE;
                    wr_pending <= 1'b0;
                end
            end
            default: begin
                if (start_in && wr_pending) begin
                    state <= STATE_FENCE;
                end else if (req_fire && req_rw) begin
                    wr_pending <= 1'b1;
                end
            end
            endcase
        end
    end

    assign start_out = ((state == STATE_IDLE) && start_in && ~wr_pending)
                    || ((state == STATE_WAIT) && fence_rsp);
    assign busy = (state != STATE_IDLE);

    assign rsp_valid = mem_bus_if.rsp_valid && ~fence_rsp;
    assign rsp_data  = mem_bus_if.rsp_data.data;
    assign rsp_tag   = mem_bus_if.rsp_data.tag.value[1 +: TAG_WIDTH];
    assign mem_bus_if.rsp_ready = fence_rsp || rsp_ready;

    `RUNTIME_ASSERT (~(start_in && (state != STATE_IDLE)), ("%t: launch while a DMA fence is pending", $time))

endmodule
