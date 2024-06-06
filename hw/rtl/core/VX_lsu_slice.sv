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
    parameter CORE_ID = 0,
    parameter BLOCK_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_commit_if.master     commit_if,
    VX_lsu_mem_if.master    lsu_mem_if
);
    localparam NUM_LANES    = `NUM_LSU_LANES;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam RSP_ARB_DATAW= `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + NUM_LANES * `XLEN + PID_WIDTH + 1 + 1;
    localparam LSUQ_SIZEW   = `LOG2UP(`LSUQ_IN_SIZE);
    localparam REQ_ASHIFT   = `CLOG2(LSU_WORD_SIZE);
    localparam MEM_ASHIFT   = `CLOG2(`MEM_BLOCK_SIZE);
    localparam MEM_ADDRW    = `MEM_ADDR_WIDTH - MEM_ASHIFT;

    // tag_id = wid + PC + rd + op_type + align + pid + pkt_addr
    localparam TAG_ID_WIDTH = `NW_WIDTH + `PC_BITS + `NR_BITS + `INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + PID_WIDTH + LSUQ_SIZEW;

    // tag = uuid + tag_id
    localparam TAG_WIDTH = `UUID_WIDTH + TAG_ID_WIDTH;

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_st_if();

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_ld_if();

    `UNUSED_VAR (execute_if.data.op_mod)
    `UNUSED_VAR (execute_if.data.rs3_data)
    `UNUSED_VAR (execute_if.data.tid)

    // full address calculation

    wire [NUM_LANES-1:0][`XLEN-1:0] full_addr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign full_addr[i] = execute_if.data.rs1_data[i] + `SEXT(`XLEN, execute_if.data.op_mod.lsu.offset);
    end

    // address type calculation

    wire [NUM_LANES-1:0][`ADDR_TYPE_WIDTH-1:0] mem_req_atype;
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [MEM_ADDRW-1:0] block_addr = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        // is I/O address
        wire [MEM_ADDRW-1:0] io_addr_start = MEM_ADDRW'(`XLEN'(`IO_BASE_ADDR) >> MEM_ASHIFT);
        assign mem_req_atype[i][`ADDR_TYPE_IO] = (block_addr >= io_addr_start);
    `ifdef LMEM_ENABLE
        // is local memory address
        wire [MEM_ADDRW-1:0] lmem_addr_start = MEM_ADDRW'(`XLEN'(`LMEM_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] lmem_addr_end = MEM_ADDRW'((`XLEN'(`LMEM_BASE_ADDR) + `XLEN'(1 << `LMEM_LOG_SIZE)) >> MEM_ASHIFT);
        assign mem_req_atype[i][`ADDR_TYPE_LOCAL] = (block_addr >= lmem_addr_start) && (block_addr < lmem_addr_end);
    `endif
    end

    wire mem_req_empty;
    wire st_rsp_ready;
    wire lsu_valid, lsu_ready;

    // fence: stall the pipeline until all pending requests are sent
    wire is_fence = `INST_LSU_IS_FENCE(execute_if.data.op_type);
    wire fence_wait = is_fence && ~mem_req_empty;

    assign lsu_valid = execute_if.valid && ~fence_wait;
    assign execute_if.ready = lsu_ready && ~fence_wait;

    // schedule memory request

    wire                            mem_req_valid;
    wire [NUM_LANES-1:0]            mem_req_mask;
    wire                            mem_req_rw;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] mem_req_addr;
    reg  [NUM_LANES-1:0][LSU_WORD_SIZE-1:0] mem_req_byteen;
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

    assign mem_req_valid = lsu_valid;
    assign lsu_ready = mem_req_ready
                    && (~mem_req_rw || st_rsp_ready); // writes commit directly

    assign mem_req_mask = execute_if.data.tmask;
    assign mem_req_rw = ~execute_if.data.wb;

    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
    `UNUSED_VAR (mem_req_fire)
    `UNUSED_VAR (mem_rsp_fire)

    // address formatting

    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] req_align;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign req_align[i] = full_addr[i][REQ_ASHIFT-1:0];
        assign mem_req_addr[i] = full_addr[i][`MEM_ADDR_WIDTH-1:REQ_ASHIFT];
    end

    // byte enable formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            mem_req_byteen[i] = '0;
            case (`INST_LSU_WSIZE(execute_if.data.op_type))
                0: begin // 8-bit
                    mem_req_byteen[i][req_align[i]] = 1'b1;
                end
                1: begin // 16 bit
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:1], 1'b0}] = 1'b1;
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:1], 1'b1}] = 1'b1;
                end
            `ifdef XLEN_64
                2: begin // 32 bit
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:2], 2'b00}] = 1'b1;
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:2], 2'b01}] = 1'b1;
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:2], 2'b10}] = 1'b1;
                    mem_req_byteen[i][{req_align[i][REQ_ASHIFT-1:2], 2'b11}] = 1'b1;
                end
            `endif
                default : mem_req_byteen[i] = {LSU_WORD_SIZE{1'b1}};
            endcase
        end
    end

    // memory misalignment not supported!
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire lsu_req_fire = execute_if.valid && execute_if.ready;
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if.data.tmask[i] || is_fence || (full_addr[i] % (1 << `INST_LSU_WSIZE(execute_if.data.op_type))) == 0),
            ("misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)",
                execute_if.data.wid, {execute_if.data.PC, 1'b0}, full_addr[i], `INST_LSU_WSIZE(execute_if.data.op_type), execute_if.data.uuid));
    end

    // store data formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin
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

    // track SOP/EOP for out-of-order memory responses

    wire [LSUQ_SIZEW-1:0] pkt_waddr, pkt_raddr;
    wire mem_rsp_sop_pkt, mem_rsp_eop_pkt;

    if (PID_BITS != 0) begin
        reg [`LSUQ_IN_SIZE-1:0][PID_BITS:0] pkt_ctr;
        reg [`LSUQ_IN_SIZE-1:0] pkt_sop, pkt_eop;

        wire mem_req_rd_fire     = mem_req_fire && execute_if.data.wb;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && execute_if.data.sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && execute_if.data.eop;
        wire mem_rsp_eop_fire    = mem_rsp_fire && mem_rsp_eop;
        wire full;

        VX_allocator #(
            .SIZE (`LSUQ_IN_SIZE)
        ) pkt_allocator (
            .clk        (clk),
            .reset      (reset),
            .acquire_en (mem_req_rd_eop_fire),
            .acquire_addr(pkt_waddr),
            .release_en (mem_rsp_eop_pkt),
            .release_addr(pkt_raddr),
            `UNUSED_PIN (empty),
            .full       (full)
        );

        wire rd_during_wr = mem_req_rd_fire && mem_rsp_eop_fire && (pkt_raddr == pkt_waddr);

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
                if (mem_rsp_eop_pkt) begin
                    pkt_eop[pkt_raddr] <= 0;
                end
                if (~rd_during_wr) begin
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
        assign mem_rsp_eop_pkt = mem_rsp_eop_fire && pkt_eop[pkt_raddr] && (pkt_ctr[pkt_raddr] == 1);
        `RUNTIME_ASSERT(~(mem_req_rd_fire && full), ("allocator full!"))
        `RUNTIME_ASSERT(~mem_req_rd_sop_fire || 0 == pkt_ctr[pkt_waddr], ("Oops!"))
        `UNUSED_VAR (mem_rsp_sop)
    end else begin
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
        execute_if.data.rd,
        execute_if.data.op_type,
        req_align, execute_if.data.pid,
        pkt_waddr
    };

    wire                                    lsu_mem_req_valid;
    wire                                    lsu_mem_req_rw;
    wire [NUM_LANES-1:0]                    lsu_mem_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0] lsu_mem_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] lsu_mem_req_addr;
    wire [NUM_LANES-1:0][`ADDR_TYPE_WIDTH-1:0] lsu_mem_req_atype;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_req_data;
    wire [LSU_TAG_WIDTH-1:0]                lsu_mem_req_tag;
    wire                                    lsu_mem_req_ready;

    wire                                    lsu_mem_rsp_valid;
    wire [NUM_LANES-1:0]                    lsu_mem_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_rsp_data;
    wire [LSU_TAG_WIDTH-1:0]                lsu_mem_rsp_tag;
    wire                                    lsu_mem_rsp_ready;

    `RESET_RELAY (mem_scheduler_reset, reset);

    VX_mem_scheduler #(
        .INSTANCE_ID ($sformatf("core%0d-lsu-memsched%0d", CORE_ID, BLOCK_ID)),
        .CORE_REQS   (NUM_LANES),
        .MEM_CHANNELS(NUM_LANES),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .LINE_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LSU_ADDR_WIDTH),
        .ATYPE_WIDTH (`ADDR_TYPE_WIDTH),
        .TAG_WIDTH   (TAG_WIDTH),
        .CORE_QUEUE_SIZE (`LSUQ_IN_SIZE),
        .MEM_QUEUE_SIZE (`LSUQ_OUT_SIZE),
        .UUID_WIDTH  (`UUID_WIDTH),
        .RSP_PARTIAL (1),
        .MEM_OUT_BUF (0),
        .CORE_OUT_BUF(0)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (mem_scheduler_reset),

        // Input request
        .core_req_valid (mem_req_valid),
        .core_req_rw    (mem_req_rw),
        .core_req_mask  (mem_req_mask),
        .core_req_byteen(mem_req_byteen),
        .core_req_addr  (mem_req_addr),
        .core_req_atype (mem_req_atype),
        .core_req_data  (mem_req_data),
        .core_req_tag   (mem_req_tag),
        .core_req_ready (mem_req_ready),
        .core_req_empty (mem_req_empty),
        `UNUSED_PIN (core_req_sent),

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
        .mem_req_atype  (lsu_mem_req_atype),
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
    assign lsu_mem_if.req_data.atype = lsu_mem_req_atype;
    assign lsu_mem_if.req_data.data = lsu_mem_req_data;
    assign lsu_mem_if.req_data.tag = lsu_mem_req_tag;
    assign lsu_mem_req_ready = lsu_mem_if.req_ready;

    assign lsu_mem_rsp_valid = lsu_mem_if.rsp_valid;
    assign lsu_mem_rsp_mask = lsu_mem_if.rsp_data.mask;
    assign lsu_mem_rsp_data = lsu_mem_if.rsp_data.data;
    assign lsu_mem_rsp_tag = lsu_mem_if.rsp_data.tag;
    assign lsu_mem_if.rsp_ready = lsu_mem_rsp_ready;

    wire [`UUID_WIDTH-1:0] rsp_uuid;
    wire [`NW_WIDTH-1:0] rsp_wid;
    wire [`PC_BITS-1:0] rsp_pc;
    wire [`NR_BITS-1:0] rsp_rd;
    wire [`INST_LSU_BITS-1:0] rsp_op_type;
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] rsp_align;
    wire [PID_WIDTH-1:0] rsp_pid;
    `UNUSED_VAR (rsp_op_type)

    // unpack memory response tag
    assign {
        rsp_uuid,
        rsp_wid,
        rsp_pc, rsp_rd,
        rsp_op_type,
        rsp_align,
        rsp_pid,
        pkt_raddr
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

    for (genvar i = 0; i < NUM_LANES; i++) begin
    `ifdef XLEN_64
        wire [63:0] rsp_data64 = mem_rsp_data[i];
        wire [31:0] rsp_data32 = (rsp_align[i][2] ? mem_rsp_data[i][63:32] : mem_rsp_data[i][31:0]);
    `else
        wire [31:0] rsp_data32 = mem_rsp_data[i];
    `endif
        wire [15:0] rsp_data16 = rsp_align[i][1] ? rsp_data32[31:16] : rsp_data32[15:0];
        wire [7:0]  rsp_data8  = rsp_align[i][0] ? rsp_data16[15:8] : rsp_data16[7:0];

        always @(*) begin
            case (`INST_LSU_FMT(rsp_op_type))
            `INST_FMT_B:  rsp_data[i] = `XLEN'(signed'(rsp_data8));
            `INST_FMT_H:  rsp_data[i] = `XLEN'(signed'(rsp_data16));
            `INST_FMT_BU: rsp_data[i] = `XLEN'(unsigned'(rsp_data8));
            `INST_FMT_HU: rsp_data[i] = `XLEN'(unsigned'(rsp_data16));
        `ifdef XLEN_64
            `INST_FMT_W:  rsp_data[i] = rsp_is_float ? (`XLEN'(rsp_data32) | 64'hffffffff00000000) : `XLEN'(signed'(rsp_data32));
            `INST_FMT_WU: rsp_data[i] = `XLEN'(unsigned'(rsp_data32));
            `INST_FMT_D:  rsp_data[i] = `XLEN'(signed'(rsp_data64));
        `else
            `INST_FMT_W:  rsp_data[i] = `XLEN'(signed'(rsp_data32));
        `endif
            default: rsp_data[i] = 'x;
            endcase
        end
    end

    // load commit

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + (NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) ld_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid),
        .ready_in  (mem_rsp_ready),
        .data_in   ({rsp_uuid, rsp_wid, mem_rsp_mask, rsp_pc, rsp_rd, rsp_data, rsp_pid, mem_rsp_sop_pkt, mem_rsp_eop_pkt}),
        .data_out  ({commit_ld_if.data.uuid, commit_ld_if.data.wid, commit_ld_if.data.tmask, commit_ld_if.data.PC, commit_ld_if.data.rd, commit_ld_if.data.data, commit_ld_if.data.pid, commit_ld_if.data.sop, commit_ld_if.data.eop}),
        .valid_out (commit_ld_if.valid),
        .ready_out (commit_ld_if.ready)
    );

    assign commit_ld_if.data.wb = 1'b1;

    // store commit

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) st_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_fire && mem_req_rw),
        .ready_in  (st_rsp_ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
        .data_out  ({commit_st_if.data.uuid, commit_st_if.data.wid, commit_st_if.data.tmask, commit_st_if.data.PC, commit_st_if.data.pid, commit_st_if.data.sop, commit_st_if.data.eop}),
        .valid_out (commit_st_if.valid),
        .ready_out (commit_st_if.ready)
    );
    assign commit_st_if.data.rd   = '0;
    assign commit_st_if.data.wb   = 1'b0;
    assign commit_st_if.data.data = commit_ld_if.data.data; // force arbiter passthru

    // lsu commit

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_ARB_DATAW),
        .OUT_BUF    (3)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({commit_st_if.valid, commit_ld_if.valid}),
        .ready_in  ({commit_st_if.ready, commit_ld_if.ready}),
        .data_in   ({commit_st_if.data, commit_ld_if.data}),
        .data_out  (commit_if.data),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready),
        `UNUSED_PIN (sel_out)
    );

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (execute_if.valid && fence_wait) begin
            `TRACE(1, ("%d: *** D$%0d fence wait\n", $time, CORE_ID));
        end
        if (mem_req_fire) begin
            if (mem_req_rw) begin
                `TRACE(1, ("%d: D$%0d Wr Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, CORE_ID, execute_if.data.wid, {execute_if.data.PC, 1'b0}, mem_req_mask));
                `TRACE_ARRAY1D(1, "0x%h", full_addr, NUM_LANES);
                `TRACE(1, (", atype="));
                `TRACE_ARRAY1D(1, "%b", mem_req_atype, NUM_LANES);
                `TRACE(1, (", byteen=0x%0h, data=", mem_req_byteen));
                `TRACE_ARRAY1D(1, "0x%0h", mem_req_data, NUM_LANES);
                `TRACE(1, (", tag=0x%0h (#%0d)\n", mem_req_tag, execute_if.data.uuid));
            end else begin
                `TRACE(1, ("%d: D$%0d Rd Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, CORE_ID, execute_if.data.wid, {execute_if.data.PC, 1'b0}, mem_req_mask));
                `TRACE_ARRAY1D(1, "0x%h", full_addr, NUM_LANES);
                `TRACE(1, (", atype="));
                `TRACE_ARRAY1D(1, "%b", mem_req_atype, NUM_LANES);
                `TRACE(1, (", byteen=0x%0h, rd=%0d, tag=0x%0h (#%0d)\n", mem_req_byteen, execute_if.data.rd, mem_req_tag, execute_if.data.uuid));
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(1, ("%d: D$%0d Rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, sop=%b, eop=%b, data=",
                $time, CORE_ID, rsp_wid, {rsp_pc, 1'b0}, mem_rsp_mask, rsp_rd, mem_rsp_sop, mem_rsp_eop));
            `TRACE_ARRAY1D(1, "0x%0h", mem_rsp_data, NUM_LANES);
            `TRACE(1, (", tag=0x%0h (#%0d)\n", mem_rsp_tag, rsp_uuid));
        end
    end
`endif

`ifdef DBG_SCOPE_LSU
    if (CORE_ID == 0 && BLOCK_ID == 0) begin
    `ifdef SCOPE
        VX_scope_tap #(
            .SCOPE_ID (3),
            .TRIGGERW (3),
            .PROBEW   (`UUID_WIDTH+NUM_LANES*(`XLEN+4+`XLEN)+1+`UUID_WIDTH+NUM_LANES*`XLEN)
        ) scope_tap (
            .clk(clk),
            .reset(scope_reset),
            .start(1'b0),
            .stop(1'b0),
            .triggers({reset, mem_req_fire, mem_rsp_fire}),
            .probes({execute_if.data.uuid, full_addr, mem_req_rw, mem_req_byteen, mem_req_data, rsp_uuid, rsp_data}),
            .bus_in(scope_bus_in),
            .bus_out(scope_bus_out)
        );
    `endif
    `ifdef CHIPSCOPE
        wire [31:0] full_addr_0 = full_addr[0];
        wire [31:0] mem_req_data_0 = mem_req_data[0];
        wire [31:0] rsp_data_0 = rsp_data[0];
        ila_lsu ila_lsu_inst (
            .clk    (clk),
            .probe0 ({mem_req_data_0, execute_if.data.uuid, execute_if.data.wid, execute_if.data.PC, mem_req_mask, full_addr_0, mem_req_byteen, mem_req_rw, mem_req_ready, mem_req_valid}),
            .probe1 ({rsp_data_0, rsp_uuid, mem_rsp_eop, rsp_pc, rsp_rd, mem_rsp_mask, rsp_wid, mem_rsp_ready, mem_rsp_valid}),
            .probe2 ({lsu_mem_if.req_data.data, lsu_mem_if.req_data.tag, lsu_mem_if.req_data.byteen, lsu_mem_if.req_data.addr, lsu_mem_if.req_data.rw, lsu_mem_if.req_ready, lsu_mem_if.req_valid}),
            .probe3 ({lsu_mem_if.rsp_data.data, lsu_mem_if.rsp_data.tag, lsu_mem_if.rsp_ready, lsu_mem_if.rsp_valid})
        );
    `endif
    end
`else
    `SCOPE_IO_UNUSED()
`endif

endmodule
