`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_IO_VX_lsu_unit

    input wire clk,
    input wire reset,

   // Dcache interface
    VX_cache_core_req_if dcache_req_if,
    VX_cache_core_rsp_if dcache_rsp_if,

    // inputs
    VX_lsu_req_if   lsu_req_if,

    // outputs
    VX_commit_if    ld_commit_if,
    VX_commit_if    st_commit_if
);
    wire                          req_valid;
    wire [`NUM_THREADS-1:0]       req_tmask;
    wire                          req_rw;
    wire [`NUM_THREADS-1:0][29:0] req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  req_byteen;
    wire [`NUM_THREADS-1:0][31:0] req_data;    
    wire [1:0]                    req_sext; 
    wire [`NR_BITS-1:0]           req_rd;
    wire                          req_wb;
    wire [`NW_BITS-1:0]           req_wid;
    wire [31:0]                   req_pc;

    wire [`NUM_THREADS-1:0][31:0] full_address;    
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign full_address[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    reg [1:0] mem_req_sext;
    always @(*) begin
        case (lsu_req_if.byteen)
            `BYTEEN_SB: mem_req_sext = 2'h1;   
            `BYTEEN_SH: mem_req_sext = 2'h2;
            default:    mem_req_sext = 2'h0;
        endcase
    end

    wire [`NUM_THREADS-1:0][29:0] mem_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  mem_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  mem_req_byteen;    
    wire [`NUM_THREADS-1:0][31:0] mem_req_data;

    reg [3:0] wmask;
    always @(*) begin
        case (`BYTEEN_TYPE(lsu_req_if.byteen))
            0:       wmask = 4'b0001;   
            1:       wmask = 4'b0011;
            default: wmask = 4'b1111;
        endcase
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin  
        assign mem_req_addr[i]   = full_address[i][31:2];        
        assign mem_req_offset[i] = full_address[i][1:0];
        assign mem_req_byteen[i] = wmask << full_address[i][1:0];
        assign mem_req_data[i]   = lsu_req_if.store_data[i] << {full_address[i][1:0], 3'b0};
    end

`IGNORE_WARNINGS_BEGIN
    wire [`NUM_THREADS-1:0][31:0] req_address;
    reg [`LSUQ_SIZE-1:0][`DCORE_TAG_WIDTH-1:0] pending_tags;
`IGNORE_WARNINGS_END

    wire ready_in;
    wire stall_in = ~ready_in & req_valid; 

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + 1 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 2 + (`NUM_THREADS * (30 + 2 + 4 + 32))),
        .RESETW (1)
    ) req_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_in),
        .data_in  ({lsu_req_if.valid, lsu_req_if.wid, lsu_req_if.tmask, lsu_req_if.PC, lsu_req_if.rw, lsu_req_if.rd, lsu_req_if.wb, full_address, mem_req_sext, mem_req_addr, mem_req_offset, mem_req_byteen, mem_req_data}),
        .data_out ({req_valid,        req_wid,        req_tmask,        req_pc,        req_rw,        req_rd,        req_wb,        req_address,  req_sext,     req_addr,     req_offset,     req_byteen,     req_data})
    );

    // Can accept new request?
    assign lsu_req_if.ready = ~stall_in;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0] rsp_pc;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;
    wire [`NUM_THREADS-1:0][1:0] rsp_offset;
    wire [1:0] rsp_sext;

    reg [`NUM_THREADS-1:0][31:0] rsp_data;
    reg [`LSUQ_SIZE-1:0][`NUM_THREADS-1:0] rsp_rem_mask;         
    reg [`NUM_THREADS-1:0] req_sent_mask, rsp_rem_mask_n;
    wire req_sent_all;

    wire [`DCORE_TAG_ID_BITS-1:0] mbuf_waddr, mbuf_raddr;
    wire mbuf_full;

    wire mbuf_push = (| dcache_req_if.valid) && (| dcache_req_if.ready) 
                  && (0 == req_sent_mask)  // first submission only
                  && (0 == req_rw);        // loads only

    wire mbuf_pop_part = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;

    wire mbuf_pop = mbuf_pop_part && (0 == rsp_rem_mask_n);
    
    assign mbuf_raddr = dcache_rsp_if.tag[0][`DCORE_TAG_ID_BITS-1:0];    

    VX_index_buffer #(
        .DATAW   (`NW_BITS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 2) + 2),
        .SIZE    (`LSUQ_SIZE),
        .FASTRAM (1)
    ) req_metadata (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (mbuf_waddr),  
        .acquire_slot (mbuf_push),       
        .read_addr    (mbuf_raddr),
        .write_data   ({req_wid, req_pc, req_rd, req_wb, req_offset, req_sext}),                    
        .read_data    ({rsp_wid, rsp_pc, rsp_rd, rsp_wb, rsp_offset, rsp_sext}),
        .release_addr (mbuf_raddr),
        .release_slot (mbuf_pop),     
        .full         (mbuf_full)
    );

    assign req_sent_all = ((dcache_req_if.ready | req_sent_mask) & req_tmask) == req_tmask;
    always @(posedge clk) begin
        if (reset) begin
            req_sent_mask <= 0;
        end else begin
            if (req_sent_all)
                req_sent_mask <= 0;
            else
                req_sent_mask <= req_sent_mask | (dcache_req_if.valid & dcache_req_if.ready);            
        end
    end      

    // need to hold the acquired tag index until the full request is submitted
    reg [`DCORE_TAG_ID_BITS-1:0] req_tag_hold;
    wire [`DCORE_TAG_ID_BITS-1:0] req_tag = (0 == req_sent_mask) ? mbuf_waddr : req_tag_hold;
    always @(posedge clk) begin
        if (mbuf_push) 
            req_tag_hold <= mbuf_waddr;
    end

    assign rsp_rem_mask_n = rsp_rem_mask[mbuf_raddr] & ~dcache_rsp_if.valid;
    always @(posedge clk) begin
        if (mbuf_push)  begin
            rsp_rem_mask[mbuf_waddr] <= req_tmask;
            pending_tags[mbuf_waddr] <= dcache_req_if.tag[0];
        end    
        if (mbuf_pop_part) begin
            rsp_rem_mask[mbuf_raddr] <= rsp_rem_mask_n;
        end
    end

    wire req_ready_dep = (!req_rw && !mbuf_full) || (req_rw && st_commit_if.ready);

    // Core Request
    assign dcache_req_if.valid  = {`NUM_THREADS{req_valid && req_ready_dep}} & req_tmask & ~req_sent_mask;
    assign dcache_req_if.rw     = {`NUM_THREADS{req_rw}};
    assign dcache_req_if.byteen = req_byteen;
    assign dcache_req_if.addr   = req_addr;
    assign dcache_req_if.data   = req_data;  

`ifdef DBG_CACHE_REQ_INFO
    assign dcache_req_if.tag = {`NUM_THREADS{{req_pc, req_wid, req_tag}}};
`else
    assign dcache_req_if.tag = {`NUM_THREADS{req_tag}};
`endif
    
    assign ready_in = req_ready_dep && req_sent_all;

    // Core Response   
    for (genvar i = 0; i < `NUM_THREADS; i++) begin        
        wire [31:0] rsp_data_shifted = dcache_rsp_if.data[i] >> {rsp_offset[i], 3'b0};
        always @(*) begin
            case (rsp_sext)
                  1: rsp_data[i] = {{24{rsp_data_shifted[7]}}, rsp_data_shifted[7:0]};  
                  2: rsp_data[i] = {{16{rsp_data_shifted[15]}}, rsp_data_shifted[15:0]};
            default: rsp_data[i] = rsp_data_shifted;     
            endcase
        end        
    end   

    // send store commit

    wire is_store_rsp = req_valid && req_rw && req_sent_all;

    assign st_commit_if.valid = is_store_rsp;
    assign st_commit_if.wid   = req_wid;
    assign st_commit_if.tmask = req_tmask;
    assign st_commit_if.PC    = req_pc;
    assign st_commit_if.rd    = 0;
    assign st_commit_if.wb    = 0;
    assign st_commit_if.eop   = 1'b1;
    assign st_commit_if.data  = 0;

    // send load commit

    wire is_load_rsp = (| dcache_rsp_if.valid);

    wire load_rsp_stall = ~ld_commit_if.ready && ld_commit_if.valid;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!load_rsp_stall),
        .data_in  ({is_load_rsp,        rsp_wid,          dcache_rsp_if.valid, rsp_pc,          rsp_rd,          rsp_wb,          rsp_data,          mbuf_pop}),
        .data_out ({ld_commit_if.valid, ld_commit_if.wid, ld_commit_if.tmask,  ld_commit_if.PC, ld_commit_if.rd, ld_commit_if.wb, ld_commit_if.data, ld_commit_if.eop})
    );

    // Can accept new cache response?
    assign dcache_rsp_if.ready = ~load_rsp_stall;

    // scope registration
    `SCOPE_ASSIGN (dcache_req_fire,  dcache_req_if.valid & dcache_req_if.ready);
    `SCOPE_ASSIGN (dcache_req_wid,   req_wid);
    `SCOPE_ASSIGN (dcache_req_pc,    req_pc);
    `SCOPE_ASSIGN (dcache_req_addr,  req_address);    
    `SCOPE_ASSIGN (dcache_req_rw,    req_rw);
    `SCOPE_ASSIGN (dcache_req_byteen,dcache_req_if.byteen);
    `SCOPE_ASSIGN (dcache_req_data,  dcache_req_if.data);
    `SCOPE_ASSIGN (dcache_req_tag,   req_tag);
    `SCOPE_ASSIGN (dcache_rsp_fire,  dcache_rsp_if.valid & {`NUM_THREADS{dcache_rsp_if.ready}});
    `SCOPE_ASSIGN (dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN (dcache_rsp_tag,   mbuf_raddr);
    
`ifdef DBG_PRINT_CORE_DCACHE
   always @(posedge clk) begin
        if ((| dcache_req_if.valid) && (|dcache_req_if.ready)) begin
            if (dcache_req_if.rw[0])
                $display("%t: D$%0d Wr Req: wid=%0d, PC=%0h, tmask=%b, addr=%0h, tag=%0h, byteen=%0h, data=%0h", 
                     $time, CORE_ID, req_wid, req_pc, (dcache_req_if.valid & dcache_req_if.ready), req_address, dcache_req_if.tag, dcache_req_if.byteen, dcache_req_if.data);
            else
                $display("%t: D$%0d Rd Req: wid=%0d, PC=%0h, tmask=%b, addr=%0h, tag=%0h, byteen=%0h, rd=%0d", 
                     $time, CORE_ID, req_wid, req_pc, (dcache_req_if.valid & dcache_req_if.ready), req_address, dcache_req_if.tag, dcache_req_if.byteen, req_rd);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D$%0d Rsp: valid=%b, wid=%0d, PC=%0h, tag=%0h, rd=%0d, data=%0h", 
                     $time, CORE_ID, dcache_rsp_if.valid, rsp_wid, rsp_pc, dcache_rsp_if.tag, rsp_rd, dcache_rsp_if.data);
        end
        if (mbuf_full) begin
            $write("%t: D$%0d queue-full:", $time, CORE_ID);
            for (integer j = 0; j < `LSUQ_SIZE; j++) begin
                $write(" tag%0d=%0h", j, pending_tags[j]);
            end            
            $write("\n");
        end
    end
`endif
    
endmodule
