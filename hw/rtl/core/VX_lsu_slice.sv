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
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_result_if.master     result_if,

    // The slice exposes the memory-client side as a VX_lsu_sched_if master;
    // the parent wires it into VX_lsu_scheduler.
    VX_lsu_sched_if.master  client_if
);
    localparam NUM_LANES    = `VX_CFG_NUM_LSU_LANES;
    localparam PID_BITS     = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam LSUQ_SIZEW   = `LOG2UP(`VX_CFG_LSU_PENDING_SIZE);
    localparam REQ_ASHIFT   = `CLOG2(LSU_WORD_SIZE);
    localparam MEM_ASHIFT   = `CLOG2(`VX_CFG_MEM_BLOCK_SIZE);
    localparam MEM_ADDRW    = `VX_CFG_MEM_ADDR_WIDTH - MEM_ASHIFT;
    // Window membership is an equality on the block-address bits above the
    // window: the LMEM base is asserted aligned to the capacity, so the prefix
    // compare is exact and needs no end address -- which is not representable
    // when the window ends at the top of the address space.
    localparam LMEM_BLK_LSB = `VX_CFG_LMEM_LOG_SIZE - MEM_ASHIFT;
    localparam [MEM_ADDRW-1:0] LMEM_BLK_BASE = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> MEM_ASHIFT);
    `STATIC_ASSERT (MEM_ADDRW > LMEM_BLK_LSB, ("LMEM window wider than the block address space"))
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_SPARAM (INSTANCE_ID)

    // tag_width = header + op_type + align + pkt_addr + fence
    localparam TAG_WIDTH = $bits(lsu_header_t) + INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + LSUQ_SIZEW + 1;
    `STATIC_ASSERT (TAG_WIDTH == LSU_CLIENT_TAG_WIDTH, ("LSU client tag width mismatch — pkg/slice out of sync"))

    VX_result_if #(
        .data_t (lsu_result_t)
    ) result_rsp_if();

    VX_result_if #(
        .data_t (lsu_result_t)
    ) result_no_rsp_if();

    // The third operand slot is read only by compare-and-swap, which needs
    // the atomic extension to exist at all.
`ifdef VX_CFG_EXT_A_ENABLE
`ifndef VX_CFG_EXT_ZACAS_ENABLE
    `UNUSED_VAR (execute_if.data.rs3_data)
`endif
`else
    `UNUSED_VAR (execute_if.data.rs3_data)
`endif

    // Register-pair atomics of a 32-bit hart (AMO64; VX_uop_packld splits them):
    //  - the AMO (a .D atomic) issues one thread per request, its 64-bit
    //    operand on the even/odd lane pair holding the thread's own lane (low
    //    word on the even lane), which the dcache coalescer fuses into one
    //    8-byte atomic and the local memory serves as one. Staying on the
    //    thread's lane pair keeps the atomic behind the thread's older accesses
    //    to the same word, which travel the same lane. Its response retires
    //    twice, the old low word into rd, then the old high word into rd+1 as
    //    the HI uop's retirement;
    //  - the CMP uop stages the rd pair here (the compare-and-swap comparand)
    //    and retires like a store;
    //  - the HI uop only reserved rd+1 and is dropped.
    // On a 32-bit hart op_type LD is only ever a pair atomic: the decoder
    // refuses the RV64 load and store encodings.
`ifdef VX_CFG_EXT_AMO64_ENABLE
    wire req_is_pair     = (`VX_CFG_XLEN == 32)
                        && execute_if.data.op_args.lsu.amo_valid
                        && (execute_if.data.op_type == INST_OP_BITS'(INST_LSU_LD));
    wire req_is_pair_cmp = execute_if.data.op_args.lsu.amo_pair && execute_if.data.op_args.lsu.is_store;
    wire req_is_pair_hi  = execute_if.data.op_args.lsu.amo_pair && ~execute_if.data.op_args.lsu.is_store;
    `STATIC_ASSERT (((NUM_LANES % 2) == 0), ("AMO64 needs even/odd LSU lane pairs"))
`else
    wire req_is_pair     = 1'b0;
    wire req_is_pair_cmp = 1'b0;
    wire req_is_pair_hi  = 1'b0;
`endif

    // full address calculation — per-lane AGU (all address forms live in
    // VX_lsu_agu; this slice contains no address arithmetic).

    wire req_is_fence, rsp_is_fence;

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] full_addr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_full_addr
        VX_lsu_agu lsu_agu (
            .base   (execute_if.data.rs1_data[i]),
            .stride (execute_if.data.rs2_data[i]),
            .offset (execute_if.data.op_args.lsu.offset),
            .pack   (execute_if.data.op_args.lsu.pack),
            .addr   (full_addr[i])
        );
    end

    // Pair atomic: one thread per request, lowest pending lane first.
    wire [NUM_LANES-1:0] pair_pending;
    wire [NUM_LANES-1:0] pair_sel_onehot;
    wire [`UP(`CLOG2(NUM_LANES))-1:0] pair_sel;
    wire pair_first, pair_last;
`ifdef VX_CFG_EXT_AMO64_ENABLE
    reg [NUM_LANES-1:0] pair_issued_r;
    assign pair_pending = execute_if.data.header.tmask & ~pair_issued_r;
    VX_priority_encoder #(
        .N (NUM_LANES)
    ) pair_sel_enc (
        .data_in    (pair_pending),
        .onehot_out (pair_sel_onehot),
        .index_out  (pair_sel),
        `UNUSED_PIN (valid_out)
    );
    assign pair_first = (pair_issued_r == '0);
    assign pair_last  = ((pair_pending & ~pair_sel_onehot) == '0);
`else
    assign pair_pending    = '0;
    assign pair_sel_onehot = '0;
    assign pair_sel        = '0;
    assign pair_first      = 1'b1;
    assign pair_last       = 1'b1;
    `UNUSED_VAR ({pair_pending, pair_sel_onehot, pair_sel})
`endif

    // address type + AMO classification — per-lane mem_bus_attr_t
    // carried through the scheduler's USER channel and out via the
    // lsu_mem_if's per-lane req_data.user field.

    mem_bus_attr_t [NUM_LANES-1:0] mem_req_attr_struct;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] mem_req_attr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_attr
        wire [MEM_ADDRW-1:0] block_addr = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        // is I/O address
        wire [MEM_ADDRW-1:0] io_addr_start = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] io_addr_end = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_END_ADDR) >> MEM_ASHIFT);
        assign mem_req_attr_struct[i].is_flush  = req_is_fence;
    `ifdef VX_CFG_EXT_OM_ENABLE
        // OM fragment-export aperture.
        //
        // An aperture store is presented to the cache hierarchy as an ORDINARY IO
        // STORE: is_addr_io is asserted alongside is_addr_om. That is not a hack —
        // uncached, bypassed and posted is exactly the behaviour an export wants,
        // and it means no cache level needs to know the aperture exists. The
        // caches carry req_data.attr through verbatim without decoding it, so the
        // is_addr_om bit rides along untouched and the cluster's OM steer peels
        // the store off the L1->L2 trunk on that bit alone.
        wire [MEM_ADDRW-1:0] om_addr_start = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_OM_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] om_addr_end = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_OM_END_ADDR) >> MEM_ASHIFT);
        wire lane_is_om = (block_addr >= om_addr_start) && (block_addr < om_addr_end);
        assign mem_req_attr_struct[i].is_addr_om = lane_is_om;
    `else
        wire lane_is_om = 1'b0;
        assign mem_req_attr_struct[i].is_addr_om = 1'b0;
    `endif
        assign mem_req_attr_struct[i].is_addr_io = ((block_addr >= io_addr_start) && (block_addr < io_addr_end))
                                                 || lane_is_om;
    `ifdef VX_CFG_LMEM_ENABLE
        // is local memory address
        assign mem_req_attr_struct[i].is_addr_local =
            (block_addr[MEM_ADDRW-1:LMEM_BLK_LSB] == LMEM_BLK_BASE[MEM_ADDRW-1:LMEM_BLK_LSB]);
    `else
        assign mem_req_attr_struct[i].is_addr_local = 1'b0;
    `endif
    `ifdef VX_CFG_EXT_A_ENABLE
        amo_req_t lane_amo;
        assign lane_amo.amo_valid    = execute_if.data.op_args.lsu.amo_valid;
        assign lane_amo.amo_op       = execute_if.data.op_args.lsu.amo_op;
        // amo_unsigned distinguishes signed/unsigned AMOMIN/MAX variants
        // (decoder collapses MINU/MAXU into MIN/MAX + this bit).
        assign lane_amo.amo_unsigned = execute_if.data.op_args.lsu.amo_unsigned;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        // Compare-and-swap comparand. The decoder routes rd into the third
        // operand slot for it; every other atomic leaves the slot unread.
        assign lane_amo.amo_cmp      = AMO_CMP_BITS'(execute_if.data.rs3_data[i]);
    `endif
    `ifdef VX_CFG_EXT_AMO64_ENABLE
        assign lane_amo.amo_wide     = 1'b0;
    `endif
        // make_hart_id(cid, wid, tid) — packed concatenation, low bits = tid.
        assign lane_amo.hart_id      = HART_ID_WIDTH'(
            (HART_ID_WIDTH'(CORE_ID) << (NW_BITS + NT_BITS))
          | (HART_ID_WIDTH'(execute_if.data.header.wid) << NT_BITS)
          |  HART_ID_WIDTH'(i)
        );
        assign mem_req_attr_struct[i].amo = lane_amo;
    `else
        // EXT_A disabled: tie the AMO sideband to zero so the bits don't
        // propagate as 'x under --x-assign unique.
        assign mem_req_attr_struct[i].amo = '0;
    `endif
        assign mem_req_attr[i] = mem_req_attr_struct[i];
    end

    // The selected thread's attribute rides both lanes of a pair request,
    // marked wide and carrying the staged 64-bit comparand.
    wire [MEM_ATTR_WIDTH-1:0] pair_attr;
    // the even/odd lanes a pair request occupies: the selected thread's own
    wire [NUM_LANES-1:0] pair_lanes;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_pair_lanes
        assign pair_lanes[i] = req_is_pair && ((i / 2) == (32'(pair_sel) / 2));
    end
`ifdef VX_CFG_EXT_AMO64_ENABLE
    wire [63:0] pair_cmp;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    // One comparand per (issue slot within this block, operand collector,
    // packet). A warp's uops leave its collector in order and the CMP..AMO
    // burst holds the slot's issue lock, so no other CMP of the same slot and
    // collector can land between a CMP and its AMO.
    localparam BLK_BITS   = `CLOG2(`VX_CFG_NUM_LSU_BLOCKS);
    localparam STG_WBITS  = ISSUE_ISW_BITS + NUM_OPCS_BITS - BLK_BITS;
    localparam STG_ADDRW  = STG_WBITS + PID_BITS;
    localparam STG_SIZE   = 1 << STG_ADDRW;
    wire [`UP(STG_ADDRW)-1:0] stg_addr;
    if (STG_ADDRW != 0) begin : g_stg_addr
        wire [`UP(STG_WBITS)-1:0] stg_wsel;
        if (STG_WBITS != 0) begin : g_wsel
            assign stg_wsel = execute_if.data.header.wid[BLK_BITS +: STG_WBITS];
        end else begin : g_no_wsel
            assign stg_wsel = '0;
        end
        if (PID_BITS != 0) begin : g_pid
            assign stg_addr = STG_ADDRW'({stg_wsel, execute_if.data.header.pid[PID_BITS-1:0]});
        end else begin : g_no_pid
            assign stg_addr = STG_ADDRW'(stg_wsel);
        end
    end else begin : g_stg_addr0
        assign stg_addr = '0;
    end
    wire [NUM_LANES-1:0][63:0] stg_wdata;
    wire [NUM_LANES-1:0][63:0] stg_rdata;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_stg_wdata
        assign stg_wdata[i] = {execute_if.data.rs3_data[i], execute_if.data.rs2_data[i]};
    end
    VX_dp_ram #(
        .DATAW   (NUM_LANES * 64),
        .SIZE    (STG_SIZE),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) pair_cmp_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (execute_if.valid && execute_if.ready && req_is_pair_cmp),
        .wren  (1'b1),
        .waddr (stg_addr),
        .wdata (stg_wdata),
        .raddr (stg_addr),
        .rdata (stg_rdata)
    );
    assign pair_cmp = stg_rdata[pair_sel];
`else
    assign pair_cmp = '0;
    `UNUSED_VAR (pair_cmp)
`endif
    mem_bus_attr_t pair_attr_struct;
    always @(*) begin
        pair_attr_struct = mem_req_attr_struct[pair_sel];
        pair_attr_struct.amo.amo_wide = 1'b1;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        pair_attr_struct.amo.amo_cmp  = pair_cmp;
    `endif
    end
    assign pair_attr = pair_attr_struct;
`else
    assign pair_attr = '0;
`endif

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
            if (mem_req_fire && req_is_fence && execute_if.data.header.eop) begin
                fence_lock <= 1;
            end
            if (mem_rsp_fire && rsp_is_fence && mem_rsp_eop_pkt) begin
                fence_lock <= 0;
            end
        end
    end

    wire req_skip = req_is_fence && ~execute_if.data.header.eop;
    // retires through the no-response path without a memory request
    wire req_bypass = req_skip || req_is_pair_cmp;
    wire no_rsp_buf_enable = (mem_req_rw && ~execute_if.data.header.wb) || req_bypass;
    // a pair atomic holds the packet until its last thread's request
    wire req_done = ~req_is_pair || pair_last;

    assign mem_req_valid = execute_if.valid
                        && ~req_bypass
                        && ~req_is_pair_hi
                        && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                        && ~fence_lock;

    assign no_rsp_buf_valid = execute_if.valid
                           && no_rsp_buf_enable
                           && (req_bypass || mem_req_ready)
                           && ~fence_lock;

    assign execute_if.ready = (req_is_pair_hi
                            || (((mem_req_ready && req_done) || req_bypass)
                             && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)))
                           && ~fence_lock;

    assign mem_req_mask = req_is_pair ? pair_lanes : execute_if.data.header.tmask;

`ifdef VX_CFG_EXT_AMO64_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            pair_issued_r <= '0;
        end else if (mem_req_fire && req_is_pair) begin
            pair_issued_r <= pair_last ? '0 : (pair_issued_r | pair_sel_onehot);
        end
    end
`endif

    // a pair atomic's requests carry one packet's sop/eop between them
    wire req_sop = execute_if.data.header.sop && (~req_is_pair || pair_first);
    wire req_eop = execute_if.data.header.eop && req_done;
`ifdef VX_CFG_EXT_A_ENABLE
    // AMO MemReq.rw is unconditionally 0: a missing line under SC must
    // miss-and-return-failure, not write-and-succeed.
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store
                     && ~execute_if.data.op_args.lsu.amo_valid;
`else
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store;
`endif

    // address formatting

    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] req_align;

    wire [LSU_ADDR_WIDTH-1:0] pair_addr = full_addr[pair_sel][`VX_CFG_MEM_ADDR_WIDTH-1:REQ_ASHIFT];
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_addr
        assign req_align[i] = full_addr[i][REQ_ASHIFT-1:0];
        // the even lane carries the low word, the odd lane the high word
        assign mem_req_addr[i] = pair_lanes[i] ? (pair_addr + LSU_ADDR_WIDTH'(i % 2))
                                               : full_addr[i][`VX_CFG_MEM_ADDR_WIDTH-1:REQ_ASHIFT];
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
            `ifdef VX_CFG_XLEN_64
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
        assign mem_req_byteen[i] = req_is_pair ? {LSU_WORD_SIZE{1'b1}} : mem_req_byteen_w;
    end

    // memory misalignment not supported!
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_missalign
        wire lsu_req_fire = execute_if.valid && execute_if.ready;
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if.data.header.tmask[i] || req_is_fence || (full_addr[i] % (1 << inst_lsu_wsize(execute_if.data.op_type))) == 0),
            ("%t: misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), full_addr[i], inst_lsu_wsize(execute_if.data.op_type), execute_if.data.header.uuid))
    end

    // store data formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_data
        always @(*) begin
            mem_req_data[i] = execute_if.data.rs2_data[i];
            case (req_align[i])
                1: mem_req_data[i][`VX_CFG_XLEN-1:8]  = execute_if.data.rs2_data[i][`VX_CFG_XLEN-9:0];
                2: mem_req_data[i][`VX_CFG_XLEN-1:16] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-17:0];
                3: mem_req_data[i][`VX_CFG_XLEN-1:24] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-25:0];
            `ifdef VX_CFG_XLEN_64
                4: mem_req_data[i][`VX_CFG_XLEN-1:32] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-33:0];
                5: mem_req_data[i][`VX_CFG_XLEN-1:40] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-41:0];
                6: mem_req_data[i][`VX_CFG_XLEN-1:48] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-49:0];
                7: mem_req_data[i][`VX_CFG_XLEN-1:56] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-57:0];
            `endif
                default:;
            endcase
            if (pair_lanes[i]) begin
                mem_req_data[i] = ((i % 2) == 0) ? execute_if.data.rs2_data[pair_sel]
                                                 : execute_if.data.rs3_data[pair_sel];
            end
        end
    end

    // multi-packet load responses could return out-of-order.
    // we should track and flag SOP and EOP responses.
    // A pair atomic sends one request per thread, so it needs the tracking
    // even for a single packet, and its count spans every thread.
`ifdef VX_CFG_EXT_AMO64_ENABLE
    localparam PKT_TRACK    = 1;
    localparam PKT_CTR_BITS = `CLOG2(`VX_CFG_NUM_THREADS + 1);
`else
    localparam PKT_TRACK    = (PID_BITS != 0);
    localparam PKT_CTR_BITS = PID_BITS + 1;
`endif

    if (PKT_TRACK) begin : g_pid
        reg [`VX_CFG_LSU_PENDING_SIZE-1:0][PKT_CTR_BITS-1:0] pkt_ctr;
        reg [`VX_CFG_LSU_PENDING_SIZE-1:0] pkt_sop, pkt_eop;

        wire mem_req_rd_fire     = mem_req_fire && ~mem_req_rw;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && req_sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && req_eop;
        wire mem_rsp_eop_fire    = mem_rsp_fire && mem_rsp_eop;
        wire mem_rsp_eop_pkt_fire= mem_rsp_fire && mem_rsp_eop_pkt;
        wire full;

        VX_allocator #(
            .SIZE (`VX_CFG_LSU_PENDING_SIZE)
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
                        pkt_ctr[pkt_waddr] <= pkt_ctr[pkt_waddr] + PKT_CTR_BITS'(1);
                    end
                    if (mem_rsp_eop_fire) begin
                        pkt_ctr[pkt_raddr] <= pkt_ctr[pkt_raddr] - PKT_CTR_BITS'(1);
                    end
                end
            end
        end

        assign mem_rsp_sop_pkt = pkt_sop[pkt_raddr];
        assign mem_rsp_eop_pkt = mem_rsp_eop && pkt_eop[pkt_raddr] && (pkt_ctr[pkt_raddr] == 1);
        `RUNTIME_ASSERT(~(mem_req_rd_fire && full), ("allocator full!"))
        `RUNTIME_ASSERT(~(mem_req_rd_sop_fire && pkt_ctr[pkt_waddr] != 0), ("oops! broken sop request!"))
        // pkt_ctr is guarded against a same-cycle same-slot request/response by
        // rw_collision; pkt_eop and pkt_sop are not, so the release wins and the
        // incoming packet loses the flag that would ever complete it.
        `RUNTIME_ASSERT(~(mem_req_rd_eop_fire && mem_rsp_eop_pkt_fire && (pkt_waddr == pkt_raddr)),
            ("%t: *** pkt_eop collision on slot %0d: incoming packet loses its eop flag", $time, pkt_waddr))
        `UNUSED_VAR (mem_rsp_sop)
    end else begin : g_no_pid
        assign pkt_waddr = 0;
        assign mem_rsp_sop_pkt = mem_rsp_sop;
        assign mem_rsp_eop_pkt = mem_rsp_eop;
        `UNUSED_VAR (pkt_raddr)
        `UNUSED_VAR ({req_sop, req_eop})
    end

    // A pair request's lanes are 8-byte aligned, so its alignment field
    // carries the thread the request serves instead.
    localparam SEL_W = `UP(`CLOG2(NUM_LANES));
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] req_align_tag = req_is_pair ? (NUM_LANES * REQ_ASHIFT)'(pair_sel) : req_align;

    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] mem_req_attr_out;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_attr_out
        // every lane, not only the pair's: the masked lanes are ignored, and
        // the coalescer reads its merge decision from lane 0's attribute
        assign mem_req_attr_out[i] = req_is_pair ? pair_attr : mem_req_attr[i];
    end

    // pack memory request tag
    assign mem_req_tag = {
        execute_if.data.header,
        execute_if.data.op_type,
        req_align_tag,
        pkt_waddr,
        req_is_fence
    };

    // Pack per-lane mem_req signals into the client_if request payload;
    // unpack the response payload into per-lane mem_rsp signals.
    assign client_if.req_valid       = mem_req_valid;
    assign client_if.req_data.rw     = mem_req_rw;
    assign client_if.req_data.mask   = mem_req_mask;
    assign client_if.req_data.byteen = mem_req_byteen;
    assign client_if.req_data.addr   = mem_req_addr;
    assign client_if.req_data.attr   = mem_req_attr_out;
    assign client_if.req_data.data   = mem_req_data;
    assign client_if.req_data.tag    = mem_req_tag;
    assign mem_req_ready             = client_if.req_ready;

    assign mem_rsp_valid             = client_if.rsp_valid;
    assign mem_rsp_mask              = client_if.rsp_data.mask;
    assign mem_rsp_data              = client_if.rsp_data.data;
    assign mem_rsp_tag               = client_if.rsp_data.tag;
    assign mem_rsp_sop               = client_if.rsp_data.sop;
    assign mem_rsp_eop               = client_if.rsp_data.eop;
    assign client_if.rsp_ready       = mem_rsp_ready;

    lsu_header_t rsp_hdr;
    wire [INST_LSU_BITS-1:0] rsp_op_type;
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] rsp_align;
    `UNUSED_VAR (rsp_op_type)

    // unpack memory response tag
    assign {
        rsp_hdr,
        rsp_op_type,
        rsp_align,
        pkt_raddr,
        rsp_is_fence
    } = mem_rsp_tag;

    wire [SEL_W-1:0] rsp_pair_sel = SEL_W'(rsp_align);
    wire [SEL_W-1:0] rsp_pair_lo  = rsp_pair_sel & ~SEL_W'(1);

    // A pair response retires in two beats: the old low word into rd, then the
    // old high word into rd+1, which is the HI uop's retirement.
    wire rsp_is_pair;
    wire rsp_hi;
    wire rsp_buf_ready;
`ifdef VX_CFG_EXT_AMO64_ENABLE
    reg rsp_hi_r;
    assign rsp_is_pair = (`VX_CFG_XLEN == 32) && (rsp_op_type == INST_LSU_BITS'(INST_LSU_LD));
    assign rsp_hi = rsp_hi_r;
    always @(posedge clk) begin
        if (reset) begin
            rsp_hi_r <= 1'b0;
        end else if (mem_rsp_valid && rsp_buf_ready && rsp_is_pair) begin
            rsp_hi_r <= ~rsp_hi_r;
        end
    end
    `RUNTIME_ASSERT(~(mem_rsp_valid && rsp_is_pair) || (mem_rsp_mask[rsp_pair_lo +: 2] == 2'b11),
        ("%t: pair atomic response without both halves (mask=%b)", $time, mem_rsp_mask))
`else
    assign rsp_is_pair = 1'b0;
    assign rsp_hi = 1'b0;
`endif
    assign mem_rsp_ready = rsp_buf_ready && (~rsp_is_pair || rsp_hi);

    // load response formatting

    reg [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rsp_data;

`ifdef VX_CFG_XLEN_64
`ifdef VX_CFG_EXT_F_ENABLE
    // apply nan-boxing to flw outputs
    wire rsp_is_float = (get_reg_type(rsp_hdr.rd) == REG_TYPE_F);
`else
    wire rsp_is_float = 0;
`endif
`endif

    for (genvar i = 0; i < NUM_LANES; i++) begin : g_rsp_data
    `ifdef VX_CFG_XLEN_64
        wire [63:0] rsp_data64 = mem_rsp_data[i];
        wire [31:0] rsp_data32 = (rsp_align[i][2] ? mem_rsp_data[i][63:32] : mem_rsp_data[i][31:0]);
    `else
        wire [31:0] rsp_data32 = mem_rsp_data[i];
    `endif
        wire [15:0] rsp_data16 = rsp_align[i][1] ? rsp_data32[31:16] : rsp_data32[15:0];
        wire [7:0]  rsp_data8  = rsp_align[i][0] ? rsp_data16[15:8] : rsp_data16[7:0];

        always @(*) begin
            case (inst_lsu_fmt(rsp_op_type))
            LSU_FMT_B:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data8));
            LSU_FMT_H:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data16));
            LSU_FMT_BU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data8));
            LSU_FMT_HU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data16));
        `ifdef VX_CFG_XLEN_64
            LSU_FMT_W:  rsp_data[i] = rsp_is_float ? (`VX_CFG_XLEN'(rsp_data32) | 64'hffffffff00000000) : `VX_CFG_XLEN'(signed'(rsp_data32));
            LSU_FMT_WU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data32));
            LSU_FMT_D:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data64));
        `else
            LSU_FMT_W:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data32));
        `endif
            default: rsp_data[i] = 'x;
            endcase
        end
    end

    // result

    lsu_header_t rsp_hdr2;
    reg [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rsp_data2;
    wire [RV_REGS_BITS-1:0] rsp_rd_idx = get_reg_idx(rsp_hdr.rd);
    always @(*) begin
        rsp_hdr2 = rsp_hdr;
        rsp_hdr2.tmask = mem_rsp_mask;
        rsp_hdr2.sop = mem_rsp_sop_pkt;
        rsp_hdr2.eop = mem_rsp_eop_pkt;
        rsp_data2 = rsp_data;
        if (rsp_is_pair) begin
            rsp_hdr2.tmask = NUM_LANES'(1) << rsp_pair_sel;
            rsp_data2[rsp_pair_sel] = rsp_hi ? `VX_CFG_XLEN'(mem_rsp_data[rsp_pair_lo + 1])
                                             : `VX_CFG_XLEN'(mem_rsp_data[rsp_pair_lo]);
            if (rsp_hi && (rsp_rd_idx != 0)) begin
                rsp_hdr2.rd = make_reg_num(REG_TYPE_I, rsp_rd_idx | RV_REGS_BITS'(1));
            end
        end
    end

    VX_elastic_buffer #(
        .DATAW ($bits(lsu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid),
        .ready_in  (rsp_buf_ready),
        .data_in   ({rsp_hdr2, rsp_data2}),
        .data_out  (result_rsp_if.data),
        .valid_out (result_rsp_if.valid),
        .ready_out (result_rsp_if.ready)
    );

    VX_elastic_buffer #(
        .DATAW ($bits(lsu_header_t)),
        .SIZE  (2)
    ) no_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (no_rsp_buf_valid),
        .ready_in  (no_rsp_buf_ready),
        .data_in   (execute_if.data.header),
        .data_out  (result_no_rsp_if.data.header),
        .valid_out (result_no_rsp_if.valid),
        .ready_out (result_no_rsp_if.ready)
    );
    assign result_no_rsp_if.data.data = result_rsp_if.data.data; // arbiter MUX optimization

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      ($bits(lsu_result_t)),
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
                `TRACE(2, ("%t: %s Wr Req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, to_fullPC(execute_if.data.header.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", attr="))
                `TRACE_ARRAY1D(2, "%b", mem_req_attr, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, data=", mem_req_byteen))
                `TRACE_ARRAY1D(2, "0x%0h", mem_req_data, NUM_LANES)
                `TRACE(2, (", sop=%b, pid=%0d, eop=%b, tag=0x%0h (#%0d)\n", execute_if.data.header.pid, execute_if.data.header.sop, execute_if.data.header.eop, mem_req_tag, execute_if.data.header.uuid))
            end else begin
                `TRACE(2, ("%t: %s Rd Req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, to_fullPC(execute_if.data.header.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", attr="))
                `TRACE_ARRAY1D(2, "%b", mem_req_attr, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, rd=%0d, pid=%0d, sop=%b, eop=%b, tag=0x%0h (#%0d)\n", mem_req_byteen, execute_if.data.header.rd, execute_if.data.header.pid, execute_if.data.header.sop, execute_if.data.header.eop, mem_req_tag, execute_if.data.header.uuid))
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%t: %s Rsp: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, rd=%0d, pid=%0d, sop=%b, eop=%b, data=",
                $time, INSTANCE_ID, rsp_hdr.wid, rsp_hdr.cta_id, to_fullPC(rsp_hdr.PC), mem_rsp_mask, rsp_hdr.rd, rsp_hdr.pid, mem_rsp_sop_pkt, mem_rsp_eop_pkt))
            `TRACE_ARRAY1D(2, "0x%0h", mem_rsp_data, NUM_LANES)
            `TRACE(2, (", tag=0x%0h (#%0d)\n", mem_rsp_tag, rsp_hdr.uuid))
        end
    end
`endif


`ifdef SCOPE
`ifdef DBG_SCOPE_LSU
    `SCOPE_IO_SWITCH (1);
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 3, 4, 2, (
            1 + NUM_LANES * (`VX_CFG_XLEN + LSU_WORD_SIZE + LSU_WORD_SIZE * 8) + UUID_WIDTH + NUM_LANES * LSU_WORD_SIZE * 8 + UUID_WIDTH
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
            execute_if.data.header.uuid,
            rsp_data,
            rsp_hdr.uuid
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
        .probe1 ({client_if.req_valid, client_if.req_data, client_if.req_ready}),
        .probe2 ({client_if.rsp_valid, client_if.rsp_data, client_if.rsp_ready})
    );
`endif
`endif

endmodule
