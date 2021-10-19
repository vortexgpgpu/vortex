`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_IO_VX_lsu_unit

    input wire              clk,
    input wire              reset,

   // Dcache interface
    VX_dcache_req_if.master dcache_req_if,
    VX_dcache_rsp_if.slave  dcache_rsp_if,

    // inputs
    VX_lsu_req_if.slave     lsu_req_if,

    // outputs
    VX_commit_if.master     ld_commit_if,
    VX_commit_if.master     st_commit_if
);
    localparam MEM_ASHIFT = `CLOG2(`MEM_BLOCK_SIZE);    
    localparam MEM_ADDRW  = 32 - MEM_ASHIFT;

    localparam REQ_ASHIFT = `CLOG2(`DCACHE_WORD_SIZE);

    localparam ADDR_TYPEW = `NC_TAG_BIT + `SM_ENABLE;

    `STATIC_ASSERT(0 == (`IO_BASE_ADDR % MEM_ASHIFT), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`SMEM_BASE_ADDR % MEM_ASHIFT), ("invalid parameter"))
    `STATIC_ASSERT(`SMEM_SIZE == `MEM_BLOCK_SIZE * (`SMEM_SIZE / `MEM_BLOCK_SIZE), ("invalid parameter"))
    
    wire                          req_valid;
    wire [`NUM_THREADS-1:0]       req_tmask;
    wire [`NUM_THREADS-1:0][31:0] req_addr;       
    wire [`INST_LSU_BITS-1:0]     req_type;
    wire [`NUM_THREADS-1:0][31:0] req_data;   
    wire [`NR_BITS-1:0]           req_rd;
    wire                          req_wb;
    wire [`NW_BITS-1:0]           req_wid;
    wire [31:0]                   req_pc;
    wire                          req_is_dup;
    wire                          req_is_prefetch;
    
    wire mbuf_empty;

    wire [`NUM_THREADS-1:0][ADDR_TYPEW-1:0] lsu_addr_type, req_addr_type;

    wire [`NUM_THREADS-1:0][31:0] full_addr;    
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign full_addr[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    // detect duplicate addresses
    wire [`NUM_THREADS-2:0] addr_matches;
    for (genvar i = 0; i < (`NUM_THREADS-1); i++) begin
        assign addr_matches[i] = (lsu_req_if.base_addr[i+1] == lsu_req_if.base_addr[0]) || ~lsu_req_if.tmask[i+1];
    end    
    wire lsu_is_dup = lsu_req_if.tmask[0] && (& addr_matches);

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        // is non-cacheable address
        wire is_addr_nc = (full_addr[i][MEM_ASHIFT +: MEM_ADDRW] >= MEM_ADDRW'(`IO_BASE_ADDR >> MEM_ASHIFT));

        if (`SM_ENABLE) begin
            // is shared memory address
            wire is_addr_sm = (full_addr[i][MEM_ASHIFT +: MEM_ADDRW] >= MEM_ADDRW'((`SMEM_BASE_ADDR - `SMEM_SIZE) >> MEM_ASHIFT))
                            & (full_addr[i][MEM_ASHIFT +: MEM_ADDRW] < MEM_ADDRW'(`SMEM_BASE_ADDR >> MEM_ASHIFT));
            assign lsu_addr_type[i] = {is_addr_nc, is_addr_sm};
        end else begin
            assign lsu_addr_type[i] = is_addr_nc;
        end
    end

    // fence stalls the pipeline until all pending requests are sent
    wire fence_wait = lsu_req_if.is_fence && (req_valid || !mbuf_empty);
    
    wire ready_in;
    wire stall_in = ~ready_in && req_valid; 
    
    wire lsu_valid = lsu_req_if.valid && ~fence_wait;

    wire lsu_wb = lsu_req_if.wb | lsu_req_if.is_prefetch;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + `NW_BITS + `NUM_THREADS + 32 + (`NUM_THREADS * 32) + (`NUM_THREADS * ADDR_TYPEW) + `INST_LSU_BITS + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) req_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_in),
        .data_in  ({lsu_valid, lsu_is_dup, lsu_req_if.is_prefetch, lsu_req_if.wid, lsu_req_if.tmask, lsu_req_if.PC, full_addr, lsu_addr_type, lsu_req_if.op_type, lsu_req_if.rd, lsu_wb, lsu_req_if.store_data}),
        .data_out ({req_valid, req_is_dup, req_is_prefetch,        req_wid,        req_tmask,        req_pc,        req_addr,  req_addr_type, req_type,           req_rd,        req_wb, req_data})
    );

    // Can accept new request?
    assign lsu_req_if.ready = ~stall_in && ~fence_wait;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0] rsp_pc;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;
    wire [`INST_LSU_BITS-1:0] rsp_type;
    wire rsp_is_dup;
    wire rsp_is_prefetch;

    `UNUSED_VAR (rsp_type)
    `UNUSED_VAR (rsp_is_prefetch)
    
    reg [`LSUQ_SIZE-1:0][`NUM_THREADS-1:0] rsp_rem_mask;         
    wire [`NUM_THREADS-1:0] rsp_rem_mask_n;
    wire [`NUM_THREADS-1:0] rsp_tmask;

    reg [`NUM_THREADS-1:0] req_sent_mask;
    reg is_req_start;

    wire [`LSUQ_ADDR_BITS-1:0] mbuf_waddr, mbuf_raddr;    
    wire mbuf_full;

    wire [`NUM_THREADS-1:0][REQ_ASHIFT-1:0] req_offset, rsp_offset;
    for (genvar i = 0; i < `NUM_THREADS; i++) begin  
        assign req_offset[i] = req_addr[i][1:0];
    end

    wire [`NUM_THREADS-1:0] dcache_req_fire = dcache_req_if.valid & dcache_req_if.ready;

    wire dcache_rsp_fire = dcache_rsp_if.valid && dcache_rsp_if.ready;

    wire [`NUM_THREADS-1:0] req_tmask_dup = req_tmask & {{(`NUM_THREADS-1){~req_is_dup}}, 1'b1};

    wire mbuf_push = ~mbuf_full
                  && (| ({`NUM_THREADS{req_valid}} & req_tmask_dup & dcache_req_if.ready))
                  && is_req_start   // first submission only                  
                  && req_wb;        // loads only

    wire mbuf_pop = dcache_rsp_fire && (0 == rsp_rem_mask_n);
    
    assign mbuf_raddr = dcache_rsp_if.tag[ADDR_TYPEW +: `LSUQ_ADDR_BITS];
    `UNUSED_VAR (dcache_rsp_if.tag)    

    // do not writeback from software prefetch
    wire req_wb2 = req_wb && ~req_is_prefetch;

    VX_index_buffer #(
        .DATAW (`NW_BITS + 32 + `NUM_THREADS + `NR_BITS + 1 + `INST_LSU_BITS + (`NUM_THREADS * REQ_ASHIFT) + 1 + 1),
        .SIZE  (`LSUQ_SIZE)
    ) req_metadata (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (mbuf_waddr),  
        .acquire_slot (mbuf_push),       
        .read_addr    (mbuf_raddr),
        .write_data   ({req_wid, req_pc, req_tmask, req_rd, req_wb2, req_type, req_offset, req_is_dup, req_is_prefetch}),                    
        .read_data    ({rsp_wid, rsp_pc, rsp_tmask, rsp_rd, rsp_wb,  rsp_type, rsp_offset, rsp_is_dup, rsp_is_prefetch}),
        .release_addr (mbuf_raddr),
        .release_slot (mbuf_pop),     
        .full         (mbuf_full),
        .empty        (mbuf_empty)
    );

    wire dcache_req_ready = &(dcache_req_if.ready | req_sent_mask | ~req_tmask_dup);

    wire [`NUM_THREADS-1:0] req_sent_mask_n = req_sent_mask | dcache_req_fire;

    always @(posedge clk) begin
        if (reset) begin
            req_sent_mask <= 0;
            is_req_start  <= 1;
        end else begin
            if (dcache_req_ready) begin
                req_sent_mask <= 0;
                is_req_start  <= 1;
            end else begin
                req_sent_mask <= req_sent_mask_n;
                is_req_start  <= (0 == req_sent_mask_n);
            end
        end
    end

    // need to hold the acquired tag index until the full request is submitted
    reg [`LSUQ_ADDR_BITS-1:0] req_tag_hold;
    wire [`LSUQ_ADDR_BITS-1:0] req_tag = is_req_start ? mbuf_waddr : req_tag_hold;
    always @(posedge clk) begin
        if (mbuf_push) begin            
            req_tag_hold <= mbuf_waddr;
        end
    end    

    assign rsp_rem_mask_n = rsp_rem_mask[mbuf_raddr] & ~dcache_rsp_if.tmask;

    always @(posedge clk) begin
        if (mbuf_push)  begin
            rsp_rem_mask[mbuf_waddr] <= req_tmask_dup;
        end    
        if (dcache_rsp_fire) begin
            rsp_rem_mask[mbuf_raddr] <= rsp_rem_mask_n;
        end
    end

    // ensure all dependencies for the requests are resolved
    wire req_dep_ready = (req_wb && ~(mbuf_full && is_req_start)) 
                      || (~req_wb && st_commit_if.ready);

    // DCache Request

    for (genvar i = 0; i < `NUM_THREADS; i++) begin

        reg [3:0]  mem_req_byteen;    
        reg [31:0] mem_req_data;

        always @(*) begin
            mem_req_byteen = {4{req_wb}};
            case (`INST_LSU_WSIZE(req_type))
                0: mem_req_byteen[req_offset[i]] = 1;
                1: begin
                    mem_req_byteen[req_offset[i]] = 1;
                    mem_req_byteen[{req_addr[i][1], 1'b1}] = 1;
                end
                default : mem_req_byteen = {4{1'b1}};
            endcase
        end

        always @(*) begin
            mem_req_data = req_data[i];
            case (req_offset[i])
                1: mem_req_data[31:8]  = req_data[i][23:0];
                2: mem_req_data[31:16] = req_data[i][15:0];
                3: mem_req_data[31:24] = req_data[i][7:0];
                default:;
            endcase
        end

        assign dcache_req_if.valid[i]  = req_valid && req_dep_ready && req_tmask_dup[i] && !req_sent_mask[i];
        assign dcache_req_if.rw[i]     = ~req_wb;
        assign dcache_req_if.addr[i]   = req_addr[i][31:2];
        assign dcache_req_if.byteen[i] = mem_req_byteen;
        assign dcache_req_if.data[i]   = mem_req_data;

    `ifdef DBG_CACHE_REQ_INFO
        assign dcache_req_if.tag[i] = {req_wid, req_pc, req_tag, req_addr_type[i]};
    `else
        assign dcache_req_if.tag[i] = {req_tag, req_addr_type[i]};
    `endif
    end
    
    assign ready_in = req_dep_ready && dcache_req_ready;

    // send store commit

    wire is_store_rsp = req_valid && ~req_wb && dcache_req_ready;

    assign st_commit_if.valid = is_store_rsp;
    assign st_commit_if.wid   = req_wid;
    assign st_commit_if.tmask = req_tmask;
    assign st_commit_if.PC    = req_pc;
    assign st_commit_if.rd    = 0;
    assign st_commit_if.wb    = 0;
    assign st_commit_if.eop   = 1'b1;
    assign st_commit_if.data  = 0;

    // load response formatting

    reg [`NUM_THREADS-1:0][31:0] rsp_data;
    wire [`NUM_THREADS-1:0] rsp_tmask_qual;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin     
        wire [31:0] rsp_data32 = (i == 0 || rsp_is_dup) ? dcache_rsp_if.data[0] : dcache_rsp_if.data[i];
        wire [15:0] rsp_data16 = rsp_offset[i][1] ? rsp_data32[31:16] : rsp_data32[15:0];
        wire [7:0]  rsp_data8  = rsp_offset[i][0] ? rsp_data16[15:8] : rsp_data16[7:0];

        always @(*) begin
            case (`INST_LSU_FMT(rsp_type))
            `INST_FMT_B:  rsp_data[i] = 32'(signed'(rsp_data8));
            `INST_FMT_H:  rsp_data[i] = 32'(signed'(rsp_data16));
            `INST_FMT_BU: rsp_data[i] = 32'(unsigned'(rsp_data8));
            `INST_FMT_HU: rsp_data[i] = 32'(unsigned'(rsp_data16));
            default: rsp_data[i] = rsp_data32;
            endcase
        end        
    end   

    assign rsp_tmask_qual = rsp_is_dup ? rsp_tmask : dcache_rsp_if.tmask;

    // send load commit

    wire load_rsp_stall = ~ld_commit_if.ready && ld_commit_if.valid;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!load_rsp_stall),
        .data_in  ({dcache_rsp_if.valid, rsp_wid,          rsp_tmask_qual,     rsp_pc,          rsp_rd,          rsp_wb,          rsp_data,          mbuf_pop}),
        .data_out ({ld_commit_if.valid,  ld_commit_if.wid, ld_commit_if.tmask, ld_commit_if.PC, ld_commit_if.rd, ld_commit_if.wb, ld_commit_if.data, ld_commit_if.eop})
    );

    // Can accept new cache response?
    assign dcache_rsp_if.ready = ~load_rsp_stall;

    // scope registration
    `SCOPE_ASSIGN (dcache_req_fire,  dcache_req_fire);
    `SCOPE_ASSIGN (dcache_req_wid,   req_wid);
    `SCOPE_ASSIGN (dcache_req_pc,    req_pc);
    `SCOPE_ASSIGN (dcache_req_addr,  req_addr);    
    `SCOPE_ASSIGN (dcache_req_rw,    ~req_wb);
    `SCOPE_ASSIGN (dcache_req_byteen,dcache_req_if.byteen);
    `SCOPE_ASSIGN (dcache_req_data,  dcache_req_if.data);
    `SCOPE_ASSIGN (dcache_req_tag,   req_tag);
    `SCOPE_ASSIGN (dcache_rsp_fire,  dcache_rsp_if.tmask & {`NUM_THREADS{dcache_rsp_fire}});
    `SCOPE_ASSIGN (dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN (dcache_rsp_tag,   mbuf_raddr);

`ifndef SYNTHESIS
    reg [`LSUQ_SIZE-1:0][(`NW_BITS + 32 + `NR_BITS + 64 + 1)-1:0] pending_reqs;
    wire [63:0] delay_timeout = 10000 * (1 ** (`L2_ENABLE + `L3_ENABLE));

    always @(posedge clk) begin
        if (reset) begin
            pending_reqs <= '0;
        end begin
            if (mbuf_push) begin            
                pending_reqs[mbuf_waddr] <= {req_wid, req_pc, req_rd, $time, 1'b1};
            end
            if (mbuf_pop) begin
                pending_reqs[mbuf_raddr] <= '0;
            end
        end

        for (integer i = 0; i < `LSUQ_SIZE; ++i) begin
            if (pending_reqs[i][0]) begin 
                `ASSERT(($time - pending_reqs[i][1 +: 64]) < delay_timeout, 
                    ("%t: *** D$%0d response timeout: remaining=%b, wid=%0d, PC=%0h, rd=%0d", 
                        $time, CORE_ID, rsp_rem_mask[i], pending_reqs[i][1+64+32+`NR_BITS +: `NW_BITS], pending_reqs[i][1+64+`NR_BITS +: 32], pending_reqs[i][1+64 +: `NR_BITS]));
            end
        end
    end
`endif
    
`ifdef DBG_TRACE_CORE_DCACHE
    wire dcache_req_fire_any = (| dcache_req_fire);
    always @(posedge clk) begin    
        if (lsu_req_if.valid && fence_wait) begin
            dpi_trace("%d: *** D$%0d fence wait\n", $time, CORE_ID);
        end
        if (dcache_req_fire_any) begin
            if (dcache_req_if.rw[0]) begin
                dpi_trace("%d: D$%0d Wr Req: wid=%0d, PC=%0h, tmask=%b, addr=", $time, CORE_ID, req_wid, req_pc, dcache_req_fire);
                `TRACE_ARRAY1D(req_addr, `NUM_THREADS);
                dpi_trace(", tag=%0h, byteen=%0h, type=", req_tag, dcache_req_if.byteen);
                `TRACE_ARRAY1D(req_addr_type, `NUM_THREADS);
                dpi_trace(", data=");
                `TRACE_ARRAY1D(dcache_req_if.data, `NUM_THREADS);
                dpi_trace("\n");
            end else begin
                dpi_trace("%d: D$%0d Rd Req: prefetch=%b, wid=%0d, PC=%0h, tmask=%b, addr=", $time, CORE_ID, req_is_prefetch, req_wid, req_pc, dcache_req_fire);
                `TRACE_ARRAY1D(req_addr, `NUM_THREADS);
                dpi_trace(", tag=%0h, byteen=%0h, type=", req_tag, dcache_req_if.byteen);
                `TRACE_ARRAY1D(req_addr_type, `NUM_THREADS);
                dpi_trace(", rd=%0d, is_dup=%b\n", req_rd, req_is_dup);
            end
        end
        if (dcache_rsp_fire) begin
            dpi_trace("%d: D$%0d Rsp: prefetch=%b, wid=%0d, PC=%0h, tmask=%b, tag=%0h, rd=%0d, data=", 
                $time, CORE_ID, rsp_is_prefetch, rsp_wid, rsp_pc, dcache_rsp_if.tmask, mbuf_raddr, rsp_rd);
            `TRACE_ARRAY1D(dcache_rsp_if.data, `NUM_THREADS);
            dpi_trace(", is_dup=%b\n", rsp_is_dup);
        end
    end
`endif
    
endmodule