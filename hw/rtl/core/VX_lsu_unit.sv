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

module VX_lsu_unit import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (    
    `SCOPE_IO_DECL

   input wire               clk,
    input wire              reset,

   // Dcache interface
    VX_mem_bus_if.master    cache_bus_if [DCACHE_NUM_REQS],

    // inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // outputs    
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE   = 1;
    localparam NUM_LANES    = `NUM_LSU_LANES;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam RSP_ARB_DATAW= `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + `NR_BITS + 1 + NUM_LANES * `XLEN + PID_WIDTH + 1 + 1;
    localparam LSUQ_SIZEW   = `LOG2UP(`LSUQ_SIZE);
    localparam MEM_ASHIFT   = `CLOG2(`MEM_BLOCK_SIZE);    
    localparam MEM_ADDRW    = `XLEN - MEM_ASHIFT;
    localparam REQ_ASHIFT   = `CLOG2(DCACHE_WORD_SIZE);
    localparam CACHE_TAG_WIDTH = `UUID_WIDTH + (NUM_LANES * `CACHE_ADDR_TYPE_BITS) + LSUQ_TAG_BITS;

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) execute_if[BLOCK_SIZE]();

    `RESET_RELAY (dispatch_reset, reset);

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (1)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (dispatch_reset),
        .dispatch_if(dispatch_if),
        .execute_if (execute_if)
    );

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_st_if();
    
    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_ld_if();
    
    `UNUSED_VAR (execute_if[0].data.op_mod)     
    `UNUSED_VAR (execute_if[0].data.use_PC)
    `UNUSED_VAR (execute_if[0].data.use_imm)
    `UNUSED_VAR (execute_if[0].data.rs3_data)
    `UNUSED_VAR (execute_if[0].data.tid)

`ifdef SM_ENABLE
    `STATIC_ASSERT((1 << `SMEM_LOG_SIZE) == `MEM_BLOCK_SIZE * ((1 << `SMEM_LOG_SIZE) / `MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`SMEM_BASE_ADDR % (1 << `SMEM_LOG_SIZE)), ("invalid parameter"))
    localparam SMEM_START_B = MEM_ADDRW'(`XLEN'(`SMEM_BASE_ADDR) >> MEM_ASHIFT);
    localparam SMEM_END_B = MEM_ADDRW'((`XLEN'(`SMEM_BASE_ADDR) + (1 << `SMEM_LOG_SIZE)) >> MEM_ASHIFT);
`endif

    // tag = uuid + addr_type + wid + PC + tmask + rd + op_type + align + is_dup + pid + pkt_addr 
    localparam TAG_WIDTH = `UUID_WIDTH + (NUM_LANES * `CACHE_ADDR_TYPE_BITS) + `NW_WIDTH + `XLEN + NUM_LANES + `NR_BITS + `INST_LSU_BITS + (NUM_LANES * (REQ_ASHIFT)) + `LSU_DUP_ENABLED + PID_WIDTH + LSUQ_SIZEW;

    `STATIC_ASSERT(0 == (`IO_BASE_ADDR % `MEM_BLOCK_SIZE), ("invalid parameter"))

    wire [NUM_LANES-1:0][`CACHE_ADDR_TYPE_BITS-1:0] lsu_addr_type;

    // full address calculation

    wire [NUM_LANES-1:0][`XLEN-1:0] full_addr;    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign full_addr[i] = execute_if[0].data.rs1_data[i][`XLEN-1:0] + execute_if[0].data.imm;
    end

    // detect duplicate addresses

    wire lsu_is_dup;
`ifdef LSU_DUP_ENABLE
    if (NUM_LANES > 1) begin    
        wire [NUM_LANES-2:0] addr_matches;
        for (genvar i = 0; i < (NUM_LANES-1); ++i) begin
            assign addr_matches[i] = (execute_if[0].data.rs1_data[i+1] == execute_if[0].data.rs1_data[0]) || ~execute_if[0].data.tmask[i+1];
        end
        assign lsu_is_dup = execute_if[0].data.tmask[0] && (& addr_matches);
    end else begin
        assign lsu_is_dup = 0;
    end
`else
    assign lsu_is_dup = 0;
`endif

    // detect address type

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [MEM_ADDRW-1:0] full_addr_b = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        // is non-cacheable I/O address
        wire is_addr_io = (full_addr_b >= MEM_ADDRW'(`XLEN'(`IO_BASE_ADDR) >> MEM_ASHIFT));
    `ifdef SM_ENABLE
        // is shared memory address
        wire is_addr_sm = (full_addr_b >= SMEM_START_B) && (full_addr_b < SMEM_END_B);
        assign lsu_addr_type[i] = {is_addr_io, is_addr_sm};
    `else
        assign lsu_addr_type[i] = is_addr_io;
    `endif
    end

    wire mem_req_empty;
    wire st_rsp_ready;
    wire lsu_valid, lsu_ready;

    // fence: stall the pipeline until all pending requests are sent
    wire is_fence = `INST_LSU_IS_FENCE(execute_if[0].data.op_type);
    wire fence_wait = is_fence && ~mem_req_empty;
    
    assign lsu_valid = execute_if[0].valid && ~fence_wait;
    assign execute_if[0].ready = lsu_ready && ~fence_wait;

    // schedule memory request    

    wire                            mem_req_valid;
    wire [NUM_LANES-1:0]            mem_req_mask;
    wire                            mem_req_rw;  
    wire [NUM_LANES-1:0][`MEM_ADDR_WIDTH-REQ_ASHIFT-1:0] mem_req_addr;
    reg  [NUM_LANES-1:0][DCACHE_WORD_SIZE-1:0] mem_req_byteen;
    reg  [NUM_LANES-1:0][`XLEN-1:0] mem_req_data;
    wire [TAG_WIDTH-1:0]            mem_req_tag;
    wire                            mem_req_ready;

    wire                            mem_rsp_valid;
    wire [NUM_LANES-1:0]            mem_rsp_mask;
    wire [NUM_LANES-1:0][`XLEN-1:0] mem_rsp_data;
    wire [TAG_WIDTH-1:0]            mem_rsp_tag;
    wire                            mem_rsp_sop;
    wire                            mem_rsp_eop;
    wire                            mem_rsp_ready;

    assign mem_req_valid = lsu_valid;
    assign lsu_ready = mem_req_ready 
                   && (~mem_req_rw || st_rsp_ready); // writes commit directly

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign mem_req_mask[i] = execute_if[0].data.tmask[i] && (~lsu_is_dup || (i == 0));
    end

    assign mem_req_rw = ~execute_if[0].data.wb;    

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
            case (`INST_LSU_WSIZE(execute_if[0].data.op_type))
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
                default : mem_req_byteen[i] = {DCACHE_WORD_SIZE{1'b1}};
            endcase
        end
    end

    // memory misalignment not supported!
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire lsu_req_fire = execute_if[0].valid && execute_if[0].ready;        
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if[0].data.tmask[i] || is_fence || (full_addr[i] % (1 << `INST_LSU_WSIZE(execute_if[0].data.op_type))) == 0), 
            ("misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)", 
                execute_if[0].data.wid, execute_if[0].data.PC, full_addr[i], `INST_LSU_WSIZE(execute_if[0].data.op_type), execute_if[0].data.uuid));
    end

    // store data formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            mem_req_data[i] = execute_if[0].data.rs2_data[i];
            case (req_align[i])
                1: mem_req_data[i][`XLEN-1:8]  = execute_if[0].data.rs2_data[i][`XLEN-9:0];
                2: mem_req_data[i][`XLEN-1:16] = execute_if[0].data.rs2_data[i][`XLEN-17:0];
                3: mem_req_data[i][`XLEN-1:24] = execute_if[0].data.rs2_data[i][`XLEN-25:0];
            `ifdef XLEN_64
                4: mem_req_data[i][`XLEN-1:32] = execute_if[0].data.rs2_data[i][`XLEN-33:0];
                5: mem_req_data[i][`XLEN-1:40] = execute_if[0].data.rs2_data[i][`XLEN-41:0];
                6: mem_req_data[i][`XLEN-1:48] = execute_if[0].data.rs2_data[i][`XLEN-49:0];
                7: mem_req_data[i][`XLEN-1:56] = execute_if[0].data.rs2_data[i][`XLEN-57:0];
            `endif
                default:;
            endcase
        end
    end

    // track SOP/EOP for out-of-order memory responses  

    wire [LSUQ_SIZEW-1:0] pkt_waddr, pkt_raddr;
    wire mem_rsp_sop_pkt, mem_rsp_eop_pkt;

    if (PID_BITS != 0) begin
        reg [`LSUQ_SIZE-1:0][PID_BITS:0] pkt_ctr;
        reg [`LSUQ_SIZE-1:0] pkt_sop, pkt_eop;

        wire mem_req_rd_fire     = mem_req_fire && execute_if[0].data.wb;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && execute_if[0].data.sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && execute_if[0].data.eop;
        wire mem_rsp_eop_fire    = mem_rsp_fire && mem_rsp_eop;
        wire full;
        
        VX_allocator #(
            .SIZE (`LSUQ_SIZE)
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

    assign mem_req_tag = {
        execute_if[0].data.uuid, lsu_addr_type, execute_if[0].data.wid, execute_if[0].data.tmask, execute_if[0].data.PC, execute_if[0].data.rd, execute_if[0].data.op_type, req_align, execute_if[0].data.pid, pkt_waddr
    `ifdef LSU_DUP_ENABLE
        , lsu_is_dup
    `endif
    };

    wire [DCACHE_NUM_REQS-1:0]              cache_req_valid;
    wire [DCACHE_NUM_REQS-1:0]              cache_req_rw;
    wire [DCACHE_NUM_REQS-1:0][(`XLEN/8)-1:0] cache_req_byteen;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0] cache_req_addr;
    wire [DCACHE_NUM_REQS-1:0][`XLEN-1:0] cache_req_data;
    wire [DCACHE_NUM_REQS-1:0][CACHE_TAG_WIDTH-1:0] cache_req_tag;
    wire [DCACHE_NUM_REQS-1:0]              cache_req_ready;
    wire [DCACHE_NUM_REQS-1:0]              cache_rsp_valid;
    wire [DCACHE_NUM_REQS-1:0][`XLEN-1:0] cache_rsp_data;
    wire [DCACHE_NUM_REQS-1:0][CACHE_TAG_WIDTH-1:0] cache_rsp_tag;
    wire [DCACHE_NUM_REQS-1:0]              cache_rsp_ready;

    `RESET_RELAY (mem_scheduler_reset, reset);

    VX_mem_scheduler #(
        .INSTANCE_ID ($sformatf("core%0d-lsu-memsched", CORE_ID)),
        .NUM_REQS    (LSU_MEM_REQS), 
        .NUM_BANKS   (DCACHE_NUM_REQS),
        .ADDR_WIDTH  (DCACHE_ADDR_WIDTH),
        .DATA_WIDTH  (`XLEN),
        .QUEUE_SIZE  (`LSUQ_SIZE),
        .TAG_WIDTH   (TAG_WIDTH),
        .MEM_TAG_ID  (`UUID_WIDTH + (NUM_LANES * `CACHE_ADDR_TYPE_BITS)),
        .UUID_WIDTH  (`UUID_WIDTH),
        .RSP_PARTIAL (1),
        .MEM_OUT_REG (2)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (mem_scheduler_reset),

        // Input request
        .req_valid      (mem_req_valid),
        .req_rw         (mem_req_rw),
        .req_mask       (mem_req_mask),
        .req_byteen     (mem_req_byteen),
        .req_addr       (mem_req_addr),
        .req_data       (mem_req_data),
        .req_tag        (mem_req_tag),
        .req_empty      (mem_req_empty),
        .req_ready      (mem_req_ready),
        `UNUSED_PIN     (write_notify),
        
        // Output response
        .rsp_valid      (mem_rsp_valid),
        .rsp_mask       (mem_rsp_mask),
        .rsp_data       (mem_rsp_data),
        .rsp_tag        (mem_rsp_tag),
        .rsp_sop        (mem_rsp_sop),
        .rsp_eop        (mem_rsp_eop),
        .rsp_ready      (mem_rsp_ready),

        // Memory request
        .mem_req_valid  (cache_req_valid),
        .mem_req_rw     (cache_req_rw),
        .mem_req_byteen (cache_req_byteen),
        .mem_req_addr   (cache_req_addr),
        .mem_req_data   (cache_req_data),
        .mem_req_tag    (cache_req_tag),
        .mem_req_ready  (cache_req_ready),

        // Memory response
        .mem_rsp_valid  (cache_rsp_valid),
        .mem_rsp_data   (cache_rsp_data),
        .mem_rsp_tag    (cache_rsp_tag),
        .mem_rsp_ready  (cache_rsp_ready)
    );

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign cache_bus_if[i].req_valid = cache_req_valid[i];
        assign cache_bus_if[i].req_data.rw = cache_req_rw[i];
        assign cache_bus_if[i].req_data.byteen = cache_req_byteen[i];
        assign cache_bus_if[i].req_data.addr = cache_req_addr[i];
        assign cache_bus_if[i].req_data.data = cache_req_data[i];
        assign cache_req_ready[i] = cache_bus_if[i].req_ready;

        assign cache_rsp_valid[i] = cache_bus_if[i].rsp_valid;
        assign cache_rsp_data[i] = cache_bus_if[i].rsp_data.data;
        assign cache_bus_if[i].rsp_ready = cache_rsp_ready[i];
    end

    // cache tag formatting: <uuid, tag, type>
    
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        wire [`UUID_WIDTH-1:0]                          cache_req_uuid, cache_rsp_uuid;
        wire [NUM_LANES-1:0][`CACHE_ADDR_TYPE_BITS-1:0] cache_req_type, cache_rsp_type;        
        wire [`CLOG2(`LSUQ_SIZE)-1:0]                   cache_req_tag_x, cache_rsp_tag_x;
        if (DCACHE_NUM_BATCHES > 1) begin

            wire [DCACHE_NUM_BATCHES-1:0][`CACHE_ADDR_TYPE_BITS-1:0] cache_req_type_b, cache_rsp_type_b;            
            wire [`CACHE_ADDR_TYPE_BITS-1:0] cache_req_type_bi, cache_rsp_type_bi;
            wire [DCACHE_BATCH_SEL_BITS-1:0] cache_req_bid, cache_rsp_bid;

            assign {cache_req_uuid, cache_req_type, cache_req_bid, cache_req_tag_x} = cache_req_tag[i];
            assign cache_req_type_bi = cache_req_type_b[cache_req_bid];
            assign cache_bus_if[i].req_data.tag = {cache_req_uuid, cache_req_bid, cache_req_tag_x, cache_req_type_bi};

            assign {cache_rsp_uuid, cache_rsp_bid, cache_rsp_tag_x, cache_rsp_type_bi} = cache_bus_if[i].rsp_data.tag;
            assign cache_rsp_type_b = {DCACHE_NUM_BATCHES{cache_rsp_type_bi}};
            assign cache_rsp_tag[i] = {cache_rsp_uuid, cache_rsp_type, cache_rsp_bid, cache_rsp_tag_x};

            for (genvar j = 0; j < DCACHE_NUM_BATCHES; ++j) begin
                localparam k = j * DCACHE_NUM_REQS + i;                
                if (k < NUM_LANES) begin
                    assign cache_req_type_b[j] = cache_req_type[k];
                    assign cache_rsp_type[k] = cache_rsp_type_b[j];
                end else begin
                    assign cache_req_type_b[j] = '0;
                    `UNUSED_VAR (cache_rsp_type_b[j])
                end
            end

        end else begin
            
            assign {cache_req_uuid, cache_req_type, cache_req_tag_x} = cache_req_tag[i];
            assign cache_bus_if[i].req_data.tag = {cache_req_uuid, cache_req_tag_x, cache_req_type[i]};

            assign {cache_rsp_uuid, cache_rsp_tag_x, cache_rsp_type[i]} = cache_bus_if[i].rsp_data.tag;
            assign cache_rsp_tag[i] = {cache_rsp_uuid, cache_rsp_type, cache_rsp_tag_x};        

            for (genvar j = 0; j < DCACHE_NUM_REQS; ++j) begin
                if (i != j) begin
                    `UNUSED_VAR (cache_req_type[j])
                    assign cache_rsp_type[j] = '0;
                end
            end
        end
    end
    
    wire [`UUID_WIDTH-1:0] rsp_uuid;
    wire [NUM_LANES-1:0][`CACHE_ADDR_TYPE_BITS-1:0] rsp_addr_type;
    wire [`NW_WIDTH-1:0] rsp_wid;
    wire [NUM_LANES-1:0] rsp_tmask_uq;
    wire [`XLEN-1:0] rsp_pc;
    wire [`NR_BITS-1:0] rsp_rd;
    wire [`INST_LSU_BITS-1:0] rsp_op_type;
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] rsp_align;
    wire [PID_WIDTH-1:0] rsp_pid;
    wire rsp_is_dup;
    
`ifndef LSU_DUP_ENABLE
    assign rsp_is_dup = 0;
`endif

    assign {
        rsp_uuid, rsp_addr_type, rsp_wid, rsp_tmask_uq, rsp_pc, rsp_rd, rsp_op_type, rsp_align, rsp_pid, pkt_raddr
    `ifdef LSU_DUP_ENABLE
        , rsp_is_dup
    `endif
    } = mem_rsp_tag;
    `UNUSED_VAR (rsp_addr_type)
    `UNUSED_VAR (rsp_op_type)

    // load response formatting

    reg [NUM_LANES-1:0][`XLEN-1:0] rsp_data;
    wire [NUM_LANES-1:0] rsp_tmask;

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
        wire [63:0] rsp_data64 = (i == 0 || rsp_is_dup) ? mem_rsp_data[0] : mem_rsp_data[i];
        wire [31:0] rsp_data32 = (i == 0 || rsp_is_dup) ? (rsp_align[0][2] ? mem_rsp_data[0][63:32] : mem_rsp_data[0][31:0]) :
                                                          (rsp_align[i][2] ? mem_rsp_data[i][63:32] : mem_rsp_data[i][31:0]);
    `else
        wire [31:0] rsp_data32 = (i == 0 || rsp_is_dup) ? mem_rsp_data[0] : mem_rsp_data[i];
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

    assign rsp_tmask = rsp_is_dup ? rsp_tmask_uq : mem_rsp_mask;

    // load commit

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + `NR_BITS + (NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) ld_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid),
        .ready_in  (mem_rsp_ready),
        .data_in   ({rsp_uuid, rsp_wid, rsp_tmask, rsp_pc, rsp_rd, rsp_data, rsp_pid, mem_rsp_sop_pkt, mem_rsp_eop_pkt}),
        .data_out  ({commit_ld_if.data.uuid, commit_ld_if.data.wid, commit_ld_if.data.tmask, commit_ld_if.data.PC, commit_ld_if.data.rd, commit_ld_if.data.data, commit_ld_if.data.pid, commit_ld_if.data.sop, commit_ld_if.data.eop}),
        .valid_out (commit_ld_if.valid),
        .ready_out (commit_ld_if.ready)
    );

    assign commit_ld_if.data.wb = 1'b1;

    // store commit

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) st_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_fire && mem_req_rw),
        .ready_in  (st_rsp_ready),
        .data_in   ({execute_if[0].data.uuid, execute_if[0].data.wid, execute_if[0].data.tmask, execute_if[0].data.PC, execute_if[0].data.pid, execute_if[0].data.sop, execute_if[0].data.eop}),
        .data_out  ({commit_st_if.data.uuid, commit_st_if.data.wid, commit_st_if.data.tmask, commit_st_if.data.PC, commit_st_if.data.pid, commit_st_if.data.sop, commit_st_if.data.eop}),
        .valid_out (commit_st_if.valid),
        .ready_out (commit_st_if.ready)
    );
    assign commit_st_if.data.rd   = '0;
    assign commit_st_if.data.wb   = 1'b0;
    assign commit_st_if.data.data = commit_ld_if.data.data; // force arbiter passthru

    // lsu commit
    
    `RESET_RELAY (commit_reset, reset);

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_arb_if[1]();

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_ARB_DATAW),
        .OUT_REG    (3)
    ) rsp_arb (
        .clk       (clk),
        .reset     (commit_reset),
        .valid_in  ({commit_st_if.valid, commit_ld_if.valid}),
        .ready_in  ({commit_st_if.ready, commit_ld_if.ready}),
        .data_in   ({commit_st_if.data, commit_ld_if.data}),
        .data_out  (commit_arb_if[0].data),
        .valid_out (commit_arb_if[0].valid), 
        .ready_out (commit_arb_if[0].ready),        
        `UNUSED_PIN (sel_out)
    );

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (3)
    ) gather_unit (
        .clk           (clk),
        .reset         (commit_reset),
        .commit_in_if  (commit_arb_if),
        .commit_out_if (commit_if)
    );

`ifdef DBG_SCOPE_LSU
    if (CORE_ID == 0) begin
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
            .probes({execute_if[0].data.uuid, full_addr, mem_req_rw, mem_req_byteen, mem_req_data, rsp_uuid, rsp_data}),
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
            .probe0 ({mem_req_data_0, execute_if[0].data.uuid, execute_if[0].data.wid, execute_if[0].data.PC, mem_req_mask, full_addr_0, mem_req_byteen, mem_req_rw, mem_req_ready, mem_req_valid}),
            .probe1 ({rsp_data_0, rsp_uuid, mem_rsp_eop, rsp_pc, rsp_rd, rsp_tmask, rsp_wid, mem_rsp_ready, mem_rsp_valid}),
            .probe2 ({cache_bus_if.req_data.data, cache_bus_if.req_data.tag, cache_bus_if.req_data.byteen, cache_bus_if.req_data.addr, cache_bus_if.req_data.rw, cache_bus_if.req_ready, cache_bus_if.req_valid}),
            .probe3 ({cache_bus_if.rsp_data.data, cache_bus_if.rsp_data.tag, cache_bus_if.rsp_ready, cache_bus_if.rsp_valid})
        );
    `endif
    end
`else
    `SCOPE_IO_UNUSED()
`endif
  
`ifdef DBG_TRACE_CORE_DCACHE
    always @(posedge clk) begin    
        if (execute_if[0].valid && fence_wait) begin
            `TRACE(1, ("%d: *** D$%0d fence wait\n", $time, CORE_ID));
        end
        if (mem_req_fire) begin
            if (mem_req_rw) begin
                `TRACE(1, ("%d: D$%0d Wr Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, CORE_ID, execute_if[0].data.wid, execute_if[0].data.PC, mem_req_mask));
                `TRACE_ARRAY1D(1, full_addr, NUM_LANES);
                `TRACE(1, (", tag=0x%0h, byteen=0x%0h, type=", mem_req_tag, mem_req_byteen));
                `TRACE_ARRAY1D(1, lsu_addr_type, NUM_LANES);
                `TRACE(1, (", data="));
                `TRACE_ARRAY1D(1, mem_req_data, NUM_LANES);
                `TRACE(1, (", is_dup=%b (#%0d)\n", lsu_is_dup, execute_if[0].data.uuid));
            end else begin
                `TRACE(1, ("%d: D$%0d Rd Req: wid=%0d, PC=0x%0h, tmask=%b, addr=", $time, CORE_ID, execute_if[0].data.wid, execute_if[0].data.PC, mem_req_mask));
                `TRACE_ARRAY1D(1, full_addr, NUM_LANES);
                `TRACE(1, (", tag=0x%0h, byteen=0x%0h, type=", mem_req_tag, mem_req_byteen));
                `TRACE_ARRAY1D(1, lsu_addr_type, NUM_LANES);
                `TRACE(1, (", rd=%0d, is_dup=%b (#%0d)\n", execute_if[0].data.rd, lsu_is_dup, execute_if[0].data.uuid));
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(1, ("%d: D$%0d Rsp: wid=%0d, PC=0x%0h, tmask=%b, tag=0x%0h, rd=%0d, sop=%b, eop=%b, data=",
                $time, CORE_ID, rsp_wid, rsp_pc, mem_rsp_mask, mem_rsp_tag, rsp_rd, mem_rsp_sop, mem_rsp_eop));
            `TRACE_ARRAY1D(1, mem_rsp_data, NUM_LANES);
            `TRACE(1, (", is_dup=%b (#%0d)\n", rsp_is_dup, rsp_uuid));
        end
    end
`endif
    
endmodule
