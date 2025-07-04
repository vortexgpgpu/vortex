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

module VX_lsu_slice import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_result_if.master     result_if,
    VX_lsu_mem_if.master    lsu_mem_if
);
    localparam NUM_LANES    = `NUM_LSU_LANES;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam RSP_ARB_DATAW= UUID_WIDTH + NW_WIDTH + NUM_LANES + PC_BITS + 1 + NUM_REGS_BITS + NUM_LANES * `XLEN + PID_WIDTH +  1 + 1;
    localparam LSUQ_SIZEW   = `LOG2UP(`LSUQ_IN_SIZE);
    localparam REQ_ASHIFT   = `CLOG2(LSU_WORD_SIZE);
    localparam MEM_ASHIFT   = `CLOG2(`MEM_BLOCK_SIZE);
    localparam MEM_ADDRW    = `MEM_ADDR_WIDTH - MEM_ASHIFT;

    // tag_id = wid + PC + wb + rd + op_type + align + pid + pkt_addr + fence
    localparam TAG_ID_WIDTH = NW_WIDTH + PC_BITS + 1 + NUM_REGS_BITS + INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + PID_WIDTH + LSUQ_SIZEW + 1;

    // tag = uuid + tag_id
    localparam TAG_WIDTH = UUID_WIDTH + TAG_ID_WIDTH;

    VX_result_if #(
        .NUM_LANES (NUM_LANES)
    ) result_rsp_if();

    VX_result_if #(
        .NUM_LANES (NUM_LANES)
    ) result_no_rsp_if();

    `UNUSED_VAR (execute_if.data.rs3_data)

    // full address calculation

    wire req_is_fence, rsp_is_fence;

    wire [NUM_LANES-1:0][`XLEN-1:0] full_addr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_full_addr
        assign full_addr[i] = execute_if.data.rs1_data[i] + `SEXT(`XLEN, execute_if.data.op_args.lsu.offset);
    end

    // address type calculation

    wire [NUM_LANES-1:0][MEM_FLAGS_WIDTH-1:0] mem_req_flags;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_flags
        wire [MEM_ADDRW-1:0] block_addr = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        // is I/O address
        wire [MEM_ADDRW-1:0] io_addr_start = MEM_ADDRW'(`XLEN'(`IO_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] io_addr_end = MEM_ADDRW'(`XLEN'(`IO_END_ADDR) >> MEM_ASHIFT);
        assign mem_req_flags[i][MEM_REQ_FLAG_FLUSH] = req_is_fence;
        assign mem_req_flags[i][MEM_REQ_FLAG_IO] = (block_addr >= io_addr_start) && (block_addr < io_addr_end);
    `ifdef LMEM_ENABLE
        // is local memory address
        wire [MEM_ADDRW-1:0] lmem_addr_start = MEM_ADDRW'(`XLEN'(`LMEM_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] lmem_addr_end = MEM_ADDRW'((`XLEN'(`LMEM_BASE_ADDR) + `XLEN'(1 << `LMEM_LOG_SIZE)) >> MEM_ASHIFT);
        assign mem_req_flags[i][MEM_REQ_FLAG_LOCAL] = (block_addr >= lmem_addr_start) && (block_addr < lmem_addr_end);
    `endif
    end

    // schedule memory request

    wire                            mem_req_valid;
    wire [NUM_LANES-1:0]            mem_req_mask;
    wire                            mem_req_rw;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] mem_req_addr;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0] mem_req_byteen;
    reg  [NUM_LANES-1:0][LSU_WORD_SIZE*8-1:0] mem_req_data;
    wire [TAG_WIDTH-1:0]            mem_req_tag;
    wire                            mem_req_ready;

    wire                            mem_rsp_valid;
    wire [NUM_LANES-1:0]            mem_rsp_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE*8-1:0] mem_rsp_data;
    wire [TAG_WIDTH-1:0]            mem_rsp_tag;
    wire                            mem_rsp_sop;
    wire                            mem_rsp_eop;
    wire                            mem_rsp_ready;

    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;

    wire mem_rsp_sop_pkt, mem_rsp_eop_pkt;
    wire no_rsp_buf_valid, no_rsp_buf_ready;

    wire [LSUQ_SIZEW-1:0] pkt_waddr, pkt_raddr;

    // fence handling

    reg fence_lock;

    assign req_is_fence = inst_lsu_is_fence(execute_if.data.op_type);

    always @(posedge clk) begin
        if (reset) begin
            fence_lock <= 0;
        end else begin
            if (mem_req_fire && req_is_fence && execute_if.data.eop) begin
                fence_lock <= 1;
            end
            if (mem_rsp_fire && rsp_is_fence && mem_rsp_eop_pkt) begin
                fence_lock <= 0;
            end
        end
    end

    wire req_skip = req_is_fence && ~execute_if.data.eop;
    wire no_rsp_buf_enable = (mem_req_rw && ~execute_if.data.wb) || req_skip;

    assign mem_req_valid = execute_if.valid
                        && ~req_skip
                        && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                        && ~fence_lock;

    assign no_rsp_buf_valid = execute_if.valid
                           && no_rsp_buf_enable
                           && (req_skip || mem_req_ready)
                           && ~fence_lock;

    assign execute_if.ready = (mem_req_ready || req_skip)
                           && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                           && ~fence_lock;

    assign mem_req_mask = execute_if.data.tmask;
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store;

    // address formatting

    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] req_align;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_addr
        assign req_align[i] = full_addr[i][REQ_ASHIFT-1:0];
        assign mem_req_addr[i] = full_addr[i][`MEM_ADDR_WIDTH-1:REQ_ASHIFT];
    end

    // byte enable formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_byteen_w
        reg [LSU_WORD_SIZE-1:0] mem_req_byteen_w;
        always @(*) begin
            mem_req_byteen_w = '0;
            case (inst_lsu_wsize(execute_if.data.op_type))
                0: begin // 8-bit
                    mem_req_byteen_w[req_align[i]] = 1'b1;
                end
                1: begin // 16 bit
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:1], 1'b0}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:1], 1'b1}] = 1'b1;
                end
            `ifdef XLEN_64
                2: begin // 32 bit
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b00}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b01}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b10}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b11}] = 1'b1;
                end
            `endif
                // 3: 64 bit
                default : mem_req_byteen_w = {LSU_WORD_SIZE{1'b1}};
            endcase
        end
        assign mem_req_byteen[i] = mem_req_byteen_w;
    end

    // memory misalignment not supported!
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_missalign
        wire lsu_req_fire = execute_if.valid && execute_if.ready;
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if.data.tmask[i] || req_is_fence || (full_addr[i] % (1 << inst_lsu_wsize(execute_if.data.op_type))) == 0),
            ("%t: misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)",
                $time, execute_if.data.wid, to_fullPC(execute_if.data.PC), full_addr[i], inst_lsu_wsize(execute_if.data.op_type), execute_if.data.uuid))
    end

    // store data formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_data
        always @(*) begin
            mem_req_data[i] = execute_if.data.rs2_data[i];
            case (req_align[i])
                1: mem_req_data[i][`XLEN-1:8]  = execute_if.data.rs2_data[i][`XLEN-9:0];
                2: mem_req_data[i][`XLEN-1:16] = execute_if.data.rs2_data[i][`XLEN-17:0];
                3: mem_req_data[i][`XLEN-1:24] = execute_if.data.rs2_data[i][`XLEN-25:0];
            `ifdef XLEN_64
                4: mem_req_data[i][`XLEN-1:32] = execute_if.data.rs2_data[i][`XLEN-33:0];
                5: mem_req_data[i][`XLEN-1:40] = execute_if.data.rs2_data[i][`XLEN-41:0];
                6: mem_req_data[i][`XLEN-1:48] = execute_if.data.rs2_data[i][`XLEN-49:0];
                7: mem_req_data[i][`XLEN-1:56] = execute_if.data.rs2_data[i][`XLEN-57:0];
            `endif
                default:;
            endcase
        end
    end

    // multi-packet load responses could return out-of-order.
    // we should track and flag SOP and EOP responses.

    if (PID_BITS != 0) begin : g_pid
        reg [`LSUQ_IN_SIZE-1:0][PID_BITS:0] pkt_ctr;
        reg [`LSUQ_IN_SIZE-1:0] pkt_sop, pkt_eop;

        wire mem_req_rd_fire     = mem_req_fire && ~mem_req_rw;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && execute_if.data.sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && execute_if.data.eop;
        wire mem_rsp_eop_fire    = mem_rsp_fire && mem_rsp_eop;
        wire mem_rsp_eop_pkt_fire= mem_rsp_fire && mem_rsp_eop_pkt;
        wire full;

        VX_allocator #(
            .SIZE (`LSUQ_IN_SIZE)
        ) pkt_allocator (
            .clk        (clk),
            .reset      (reset),
            .acquire_en (mem_req_rd_eop_fire),
            .acquire_addr(pkt_waddr),
            .release_en (mem_rsp_eop_pkt_fire),
            .release_addr(pkt_raddr),
            `UNUSED_PIN (empty),
            .full       (full)
        );

        wire rw_collision = mem_req_rd_fire && mem_rsp_eop_fire && (pkt_raddr == pkt_waddr);

        always @(posedge clk) begin
            if (reset) begin
                pkt_ctr <= '0;
                pkt_sop <= '0;
                pkt_eop <= '0;
            end else begin
                if (mem_req_rd_sop_fire) begin
                    pkt_sop[pkt_waddr] <= 1;
                end
                if (mem_req_rd_eop_fire) begin
                    pkt_eop[pkt_waddr] <= 1;
                end
                if (mem_rsp_fire) begin
                    pkt_sop[pkt_raddr] <= 0;
                end
                if (mem_rsp_eop_pkt_fire) begin
                    pkt_eop[pkt_raddr] <= 0;
                end
                if (~rw_collision) begin
                    if (mem_req_rd_fire) begin
                        pkt_ctr[pkt_waddr] <= pkt_ctr[pkt_waddr] + PID_BITS'(1);
                    end
                    if (mem_rsp_eop_fire) begin
                        pkt_ctr[pkt_raddr] <= pkt_ctr[pkt_raddr] - PID_BITS'(1);
                    end
                end
            end
        end

        assign mem_rsp_sop_pkt = pkt_sop[pkt_raddr];
        assign mem_rsp_eop_pkt = mem_rsp_eop && pkt_eop[pkt_raddr] && (pkt_ctr[pkt_raddr] == 1);
        `RUNTIME_ASSERT(~(mem_req_rd_fire && full), ("%t: allocator full!", $time))
        `RUNTIME_ASSERT(~(mem_req_rd_sop_fire && pkt_ctr[pkt_waddr] != 0), ("%t: oops! broken sop request!", $time))
        `UNUSED_VAR (mem_rsp_sop)
    end else begin : g_no_pid
        assign pkt_waddr = 0;
        assign mem_rsp_sop_pkt = mem_rsp_sop;
        assign mem_rsp_eop_pkt = mem_rsp_eop;
        `UNUSED_VAR (pkt_raddr)
    end

    // pack memory request tag
    assign mem_req_tag = {
        execute_if.data.uuid,
        execute_if.data.wid,
        execute_if.data.PC,
        execute_if.data.wb,
        execute_if.data.rd,
        execute_if.data.op_type,
        req_align,
        execute_if.data.pid,
        pkt_waddr,
        req_is_fence
    };

    wire                                    lsu_mem_req_valid;
    wire                                    lsu_mem_req_rw;
    wire [NUM_LANES-1:0]                    lsu_mem_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0] lsu_mem_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] lsu_mem_req_addr;
    wire [NUM_LANES-1:0][MEM_FLAGS_WIDTH-1:0] lsu_mem_req_flags;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_req_data;
    wire [LSU_TAG_WIDTH-1:0]                lsu_mem_req_tag;
    wire                                    lsu_mem_req_ready;

    wire                                    lsu_mem_rsp_valid;
    wire [NUM_LANES-1:0]                    lsu_mem_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_rsp_data;
    wire [LSU_TAG_WIDTH-1:0]                lsu_mem_rsp_tag;
    wire                                    lsu_mem_rsp_ready;

    VX_mem_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-memsched", INSTANCE_ID))),
        .CORE_REQS   (NUM_LANES),
        .MEM_CHANNELS(NUM_LANES),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .LINE_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LSU_ADDR_WIDTH),
        .FLAGS_WIDTH (MEM_FLAGS_WIDTH),
        .TAG_WIDTH   (TAG_WIDTH),
        .CORE_QUEUE_SIZE (`LSUQ_IN_SIZE),
        .MEM_QUEUE_SIZE (`LSUQ_OUT_SIZE),
        .UUID_WIDTH  (UUID_WIDTH),
        .RSP_PARTIAL (1),
        .MEM_OUT_BUF (0),
        .CORE_OUT_BUF(0)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .core_req_valid (mem_req_valid),
        .core_req_rw    (mem_req_rw),
        .core_req_mask  (mem_req_mask),
        .core_req_byteen(mem_req_byteen),
        .core_req_addr  (mem_req_addr),
        .core_req_flags (mem_req_flags),
        .core_req_data  (mem_req_data),
        .core_req_tag   (mem_req_tag),
        .core_req_ready (mem_req_ready),

        // request queue info
        `UNUSED_PIN (req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        // Output response
        .core_rsp_valid (mem_rsp_valid),
        .core_rsp_mask  (mem_rsp_mask),
        .core_rsp_data  (mem_rsp_data),
        .core_rsp_tag   (mem_rsp_tag),
        .core_rsp_sop   (mem_rsp_sop),
        .core_rsp_eop   (mem_rsp_eop),
        .core_rsp_ready (mem_rsp_ready),

        // Memory request
        .mem_req_valid  (lsu_mem_req_valid),
        .mem_req_rw     (lsu_mem_req_rw),
        .mem_req_mask   (lsu_mem_req_mask),
        .mem_req_byteen (lsu_mem_req_byteen),
        .mem_req_addr   (lsu_mem_req_addr),
        .mem_req_flags  (lsu_mem_req_flags),
        .mem_req_data   (lsu_mem_req_data),
        .mem_req_tag    (lsu_mem_req_tag),
        .mem_req_ready  (lsu_mem_req_ready),

        // Memory response
        .mem_rsp_valid  (lsu_mem_rsp_valid),
        .mem_rsp_mask   (lsu_mem_rsp_mask),
        .mem_rsp_data   (lsu_mem_rsp_data),
        .mem_rsp_tag    (lsu_mem_rsp_tag),
        .mem_rsp_ready  (lsu_mem_rsp_ready)
    );

    assign lsu_mem_if.req_valid = lsu_mem_req_valid;
    assign lsu_mem_if.req_data.mask = lsu_mem_req_mask;
    assign lsu_mem_if.req_data.rw = lsu_mem_req_rw;
    assign lsu_mem_if.req_data.byteen = lsu_mem_req_byteen;
    assign lsu_mem_if.req_data.addr = lsu_mem_req_addr;
    assign lsu_mem_if.req_data.flags = lsu_mem_req_flags;
    assign lsu_mem_if.req_data.data = lsu_mem_req_data;
    assign lsu_mem_if.req_data.tag = lsu_mem_req_tag;
    assign lsu_mem_req_ready = lsu_mem_if.req_ready;

    assign lsu_mem_rsp_valid = lsu_mem_if.rsp_valid;
    assign lsu_mem_rsp_mask = lsu_mem_if.rsp_data.mask;
    assign lsu_mem_rsp_data = lsu_mem_if.rsp_data.data;
    assign lsu_mem_rsp_tag = lsu_mem_if.rsp_data.tag;
    assign lsu_mem_if.rsp_ready = lsu_mem_rsp_ready;

    wire [UUID_WIDTH-1:0] rsp_uuid;
    wire [NW_WIDTH-1:0] rsp_wid;
    wire [PC_BITS-1:0] rsp_pc;
    wire rsp_wb;
    wire [NUM_REGS_BITS-1:0] rsp_rd;
    wire [INST_LSU_BITS-1:0] rsp_op_type;
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] rsp_align;
    wire [PID_WIDTH-1:0] rsp_pid;
    `UNUSED_VAR (rsp_op_type)

    // unpack memory response tag
    assign {
        rsp_uuid,
        rsp_wid,
        rsp_pc,
        rsp_wb,
        rsp_rd,
        rsp_op_type,
        rsp_align,
        rsp_pid,
        pkt_raddr,
        rsp_is_fence
    } = mem_rsp_tag;

    // load response formatting

    reg [NUM_LANES-1:0][`XLEN-1:0] rsp_data;

`ifdef XLEN_64
`ifdef EXT_F_ENABLE
    // apply nan-boxing to flw outputs
    wire rsp_is_float = rsp_rd[5];
`else
    wire rsp_is_float = 0;
`endif
`endif

    for (genvar i = 0; i < NUM_LANES; i++) begin : g_rsp_data
    `ifdef XLEN_64
        wire [63:0] rsp_data64 = mem_rsp_data[i];
        wire [31:0] rsp_data32 = (rsp_align[i][2] ? mem_rsp_data[i][63:32] : mem_rsp_data[i][31:0]);
    `else
        wire [31:0] rsp_data32 = mem_rsp_data[i];
    `endif
        wire [15:0] rsp_data16 = rsp_align[i][1] ? rsp_data32[31:16] : rsp_data32[15:0];
        wire [7:0]  rsp_data8  = rsp_align[i][0] ? rsp_data16[15:8] : rsp_data16[7:0];

        always @(*) begin
            case (inst_lsu_fmt(rsp_op_type))
            LSU_FMT_B:  rsp_data[i] = `XLEN'(signed'(rsp_data8));
            LSU_FMT_H:  rsp_data[i] = `XLEN'(signed'(rsp_data16));
            LSU_FMT_BU: rsp_data[i] = `XLEN'(unsigned'(rsp_data8));
            LSU_FMT_HU: rsp_data[i] = `XLEN'(unsigned'(rsp_data16));
        `ifdef XLEN_64
            LSU_FMT_W:  rsp_data[i] = rsp_is_float ? (`XLEN'(rsp_data32) | 64'hffffffff00000000) : `XLEN'(signed'(rsp_data32));
            LSU_FMT_WU: rsp_data[i] = `XLEN'(unsigned'(rsp_data32));
            LSU_FMT_D:  rsp_data[i] = `XLEN'(signed'(rsp_data64));
        `else
            LSU_FMT_W:  rsp_data[i] = `XLEN'(signed'(rsp_data32));
        `endif
            default: rsp_data[i] = 'x;
            endcase
        end
    end

    // result

    VX_elastic_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + NUM_LANES + PC_BITS + 1 + NUM_REGS_BITS + (NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid),
        .ready_in  (mem_rsp_ready),
        .data_in   ({rsp_uuid,                rsp_wid,                mem_rsp_mask,             rsp_pc,                rsp_wb,                rsp_rd,                rsp_data,                rsp_pid,                mem_rsp_sop_pkt,        mem_rsp_eop_pkt}),
        .data_out  ({result_rsp_if.data.uuid, result_rsp_if.data.wid, result_rsp_if.data.tmask, result_rsp_if.data.PC, result_rsp_if.data.wb, result_rsp_if.data.rd, result_rsp_if.data.data, result_rsp_if.data.pid, result_rsp_if.data.sop, result_rsp_if.data.eop}),
        .valid_out (result_rsp_if.valid),
        .ready_out (result_rsp_if.ready)
    );

    VX_elastic_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + NUM_LANES + PC_BITS + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) no_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (no_rsp_buf_valid),
        .ready_in  (no_rsp_buf_ready),
        .data_in   ({execute_if.data.uuid,       execute_if.data.wid,       execute_if.data.tmask,       execute_if.data.PC,       execute_if.data.pid,       execute_if.data.sop,       execute_if.data.eop}),
        .data_out  ({result_no_rsp_if.data.uuid, result_no_rsp_if.data.wid, result_no_rsp_if.data.tmask, result_no_rsp_if.data.PC, result_no_rsp_if.data.pid, result_no_rsp_if.data.sop, result_no_rsp_if.data.eop}),
        .valid_out (result_no_rsp_if.valid),
        .ready_out (result_no_rsp_if.ready)
    );

    assign result_no_rsp_if.data.rd   = '0;
    assign result_no_rsp_if.data.wb   = 1'b0;
    assign result_no_rsp_if.data.data = result_rsp_if.data.data; // arbiter MUX optimization

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_ARB_DATAW),
        .ARBITER    ("P"), // prioritize result_rsp_if
        .OUT_BUF    (3)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({result_no_rsp_if.valid, result_rsp_if.valid}),
        .ready_in  ({result_no_rsp_if.ready, result_rsp_if.ready}),
        .data_in   ({result_no_rsp_if.data, result_rsp_if.data}),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready),
        `UNUSED_PIN (sel_out)
    );

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (execute_if.valid && fence_lock) begin
            `TRACE(2, ("%t: *** %s fence wait\n", $time, INSTANCE_ID))
        end
        if (mem_req_fire) begin
            if (mem_req_rw) begin
                `TRACE(2, ("%t: %s Wr Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.wid, to_fullPC(execute_if.data.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", flags="))
                `TRACE_ARRAY1D(2, "%b", mem_req_flags, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, data=", mem_req_byteen))
                `TRACE_ARRAY1D(2, "0x%0h", mem_req_data, NUM_LANES)
                `TRACE(2, (", sop=%b, pid=%0d, eop=%b, tag=0x%0h (#%0d)\n", execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, mem_req_tag, execute_if.data.uuid))
            end else begin
                `TRACE(2, ("%t: %s Rd Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.wid, to_fullPC(execute_if.data.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", flags="))
                `TRACE_ARRAY1D(2, "%b", mem_req_flags, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, rd=%0d, pid=%0d, sop=%b, eop=%b, tag=0x%0h (#%0d)\n", mem_req_byteen, execute_if.data.rd, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, mem_req_tag, execute_if.data.uuid))
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%t: %s Rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, pid=%0d, sop=%b, eop=%b, data=",
                $time, INSTANCE_ID, rsp_wid, to_fullPC(rsp_pc), mem_rsp_mask, rsp_rd, rsp_pid, mem_rsp_sop_pkt, mem_rsp_eop_pkt))
            `TRACE_ARRAY1D(2, "0x%0h", mem_rsp_data, NUM_LANES)
            `TRACE(2, (", tag=0x%0h (#%0d)\n", mem_rsp_tag, rsp_uuid))
        end
    end
`endif

`ifdef SCOPE
`ifdef DBG_SCOPE_LSU
    `SCOPE_IO_SWITCH (1);
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 3, 4, 2, (
            1 + NUM_LANES * (`XLEN + LSU_WORD_SIZE + LSU_WORD_SIZE * 8) + UUID_WIDTH + NUM_LANES * LSU_WORD_SIZE * 8 + UUID_WIDTH
        ), {
            mem_req_valid,
            mem_req_ready,
            mem_rsp_valid,
            mem_rsp_ready
        }, {
            mem_req_fire,
            mem_rsp_fire
        }, {
            mem_req_rw,
            full_addr,
            mem_req_byteen,
            mem_req_data,
            execute_if.data.uuid,
            rsp_data,
            rsp_uuid
        },
        reset_negedge, 1'b0,	4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_LSU
    ila_lsu ila_lsu_inst (
        .clk    (clk),
        .probe0 ({execute_if.valid, execute_if.data, execute_if.ready}),
        .probe1 ({lsu_mem_if.req_valid, lsu_mem_if.req_data, lsu_mem_if.req_ready}),
        .probe2 ({lsu_mem_if.rsp_valid, lsu_mem_if.rsp_data, lsu_mem_if.rsp_ready})
    );
`endif
`endif

endmodule
