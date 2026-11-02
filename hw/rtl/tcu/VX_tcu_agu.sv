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

`ifdef TCU_META_ENABLE

// A TCU block's TCU_LD unit: one warp-wide load in flight through an LSU
// scheduler client port. Response fragments are written straight into the
// block's metadata RAM, lane by lane, so nothing is buffered here. The load
// cannot overwrite metadata an older MMA still needs: TCU_LD enters the block
// in order behind that MMA, which reads the RAM when it issues.

module VX_tcu_agu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES           = `VX_CFG_NUM_TCU_LANES
) (
    input wire              clk,
    input wire              reset,

    input wire              ld_valid,
    input tcu_execute_t     ld_data,
    output wire             ld_ready,

    VX_lsu_sched_if.master  client_if,

    output wire             meta_wr_en,
    output wire [NW_WIDTH-1:0] meta_wr_wid,
    output wire [4:0]       meta_wr_idx,
    output wire [TCU_BLOCK_CAP-1:0] meta_wr_mask,
    output wire [TCU_BLOCK_CAP-1:0][31:0] meta_wr_data,

    // releases the TCU_LD's metadata scoreboard bit
    output wire             result_valid,
    output tcu_result_t     result_data,
    input wire              result_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_LSU_LANES, ("TCU_LD requires NUM_TCU_LANES == NUM_LSU_LANES"))

    // An LSU word holds XLEN/32 metadata words; lane i reads word i.
    localparam LSU_W32_RATIO = LSU_WORD_SIZE / 4;
    localparam LG_W32_RATIO  = (LSU_W32_RATIO > 1) ? $clog2(LSU_W32_RATIO) : 0;

    reg                     busy_r, issued_r, done_r;
    tcu_header_t            header_r;
    reg [4:0]               slot_r;
    reg [`VX_CFG_XLEN-1:0]  addr_r;

    wire ld_fire     = ld_valid && ld_ready;
    wire req_fire    = client_if.req_valid && client_if.req_ready;
    wire rsp_fire    = client_if.rsp_valid && client_if.rsp_ready;
    wire result_fire = result_valid && result_ready;

    always @(posedge clk) begin
        if (reset) begin
            busy_r   <= 1'b0;
            issued_r <= 1'b0;
            done_r   <= 1'b0;
        end else begin
            if (ld_fire) begin
                busy_r <= 1'b1;
            end
            if (req_fire) begin
                issued_r <= 1'b1;
            end
            if (rsp_fire && client_if.rsp_data.eop) begin
                done_r <= 1'b1;
            end
            if (result_fire) begin
                busy_r   <= 1'b0;
                issued_r <= 1'b0;
                done_r   <= 1'b0;
            end
        end
        if (ld_fire) begin
            header_r <= ld_data.header;
            slot_r   <= ld_data.op_args.tcu.fmt_d;
            addr_r   <= ld_data.rs1_data[0];
        end
    end

    assign ld_ready = ~busy_r;
    `UNUSED_VAR (ld_data)

    // -----------------------------------------------------------------------
    // Request: one warp-wide load of NUM_LANES consecutive 32-bit words.
    // -----------------------------------------------------------------------
    wire [LSU_ADDR_WIDTH-1:0] base_word_addr = LSU_ADDR_WIDTH'(addr_r >> `CLOG2(LSU_WORD_SIZE));
    `UNUSED_VAR (addr_r[`CLOG2(LSU_WORD_SIZE)-1:0])

`ifdef VX_CFG_LMEM_ENABLE
    // Equality on the word-address bits above the window: the base is asserted
    // aligned to the capacity, so the prefix compare is exact and needs no end
    // address -- which is not representable when the window ends at the top of
    // the address space.
    localparam LMEM_WORD_LSB = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);
    localparam [LSU_ADDR_WIDTH-1:0] LMEM_WORD_BASE =
        LSU_ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> `CLOG2(LSU_WORD_SIZE));
    `STATIC_ASSERT (LSU_ADDR_WIDTH > LMEM_WORD_LSB, ("LMEM window wider than the word address space"))
    wire base_is_local = (base_word_addr[LSU_ADDR_WIDTH-1:LMEM_WORD_LSB]
                       == LMEM_WORD_BASE[LSU_ADDR_WIDTH-1:LMEM_WORD_LSB]);
`else
    wire base_is_local = 1'b0;
`endif

    mem_bus_attr_t req_attr;
    always_comb begin
        req_attr = '0;
        req_attr.is_addr_local = base_is_local;
    end

    lsu_req_data_t req_data;
    always_comb begin
        req_data      = '0;
        req_data.rw   = 1'b0;
        req_data.mask = {NUM_LANES{1'b1}};
        for (int i = 0; i < NUM_LANES; ++i) begin
            req_data.addr[i]   = base_word_addr + (LSU_ADDR_WIDTH'(i) >> LG_W32_RATIO);
            req_data.byteen[i] = {LSU_WORD_SIZE{1'b1}};
            req_data.attr[i]   = req_attr;
        end
    end

    assign client_if.req_valid = busy_r && ~issued_r;
    assign client_if.req_data  = req_data;
    assign client_if.rsp_ready = 1'b1;
    `UNUSED_VAR (client_if.rsp_data.sop)
    `UNUSED_VAR (client_if.rsp_data.tag)

    `RUNTIME_ASSERT(~client_if.rsp_valid || (issued_r && ~done_r),
        ("%t: *** %s unexpected TCU_LD response", $time, INSTANCE_ID))

`ifdef VX_CFG_LSU_STACK_INTERLEAVE_ENABLE
    // Metadata is read as a linear warp-wide tile, which an interleaved
    // per-thread stack cannot provide.
    if (`VX_CFG_NUM_THREADS > 1) begin : g_meta_stack_check
        wire [`VX_CFG_XLEN-1:0] meta_end_addr = addr_r + `VX_CFG_XLEN'(NUM_LANES * 4 - 1);
        `RUNTIME_ASSERT(~req_fire || (meta_end_addr < STACK_WINDOW_BOTTOM) || (addr_r >= STACK_WINDOW_TOP),
            ("%t: *** %s TCU metadata inside the thread stack window: addr=0x%0h", $time, INSTANCE_ID, addr_r))
    end
`endif

    // -----------------------------------------------------------------------
    // Metadata RAM write: each response fragment writes its own lanes.
    // -----------------------------------------------------------------------
    assign meta_wr_en  = rsp_fire;
    assign meta_wr_wid = header_r.wid;
    assign meta_wr_idx = slot_r;

    for (genvar i = 0; i < TCU_BLOCK_CAP; ++i) begin : g_meta_wr
        if (i < NUM_LANES) begin : g_lane
            wire [LSU_W32_RATIO-1:0][31:0] rsp_w32 = client_if.rsp_data.data[i];
            assign meta_wr_mask[i] = client_if.rsp_data.mask[i];
            assign meta_wr_data[i] = rsp_w32[i % LSU_W32_RATIO];
            `UNUSED_VAR (rsp_w32)
        end else begin : g_pad
            assign meta_wr_mask[i] = 1'b0;
            assign meta_wr_data[i] = '0;
        end
    end

    // -----------------------------------------------------------------------
    // Result: no register writeback; its header releases the scoreboard bit.
    // -----------------------------------------------------------------------
    assign result_valid = done_r;
    always_comb begin
        result_data        = '0;
        result_data.header = header_r;
    end

endmodule

`endif // TCU_META_ENABLE
