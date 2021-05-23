`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_IO_VX_lsu_unit

    input wire clk,
    input wire reset,

   // Dcache interface
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // inputs
    VX_lsu_req_if   lsu_req_if,

    // outputs
    VX_commit_if    ld_commit_if,
    VX_commit_if    st_commit_if
);

    `UNUSED_PARAM (CORE_ID)
    
    wire                          req_valid;
    wire [`NUM_THREADS-1:0]       req_tmask;
    wire [`NUM_THREADS-1:0][31:0] req_addr;       
    wire [`LSU_BITS-1:0]          req_type;
    wire [`NUM_THREADS-1:0][31:0] req_data;   
    wire [`NR_BITS-1:0]           req_rd;
    wire                          req_wb;
    wire [`NW_BITS-1:0]           req_wid;
    wire [31:0]                   req_pc;
    wire                          req_is_dup;

    wire [`NUM_THREADS-1:0][31:0] full_address;    
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign full_address[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    wire [`NUM_THREADS-1:0] addr_matches;
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign addr_matches[i] = (full_address[0][31:2] == full_address[i][31:2]) || ~lsu_req_if.tmask[i];
    end    
    wire is_dup_load = lsu_req_if.wb && lsu_req_if.tmask[0] && (& addr_matches);
    
    wire ready_in;
    wire stall_in = ~ready_in && req_valid; 

    VX_pipe_register #(
        .DATAW  (1 + 1 + `NW_BITS + `NUM_THREADS + 32 + (`NUM_THREADS * 32) + `LSU_BITS + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) req_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_in),
        .data_in  ({lsu_req_if.valid, is_dup_load, lsu_req_if.wid, lsu_req_if.tmask, lsu_req_if.PC, full_address, lsu_req_if.op_type, lsu_req_if.rd, lsu_req_if.wb, lsu_req_if.store_data}),
        .data_out ({req_valid,        req_is_dup,  req_wid,        req_tmask,        req_pc,        req_addr,     req_type,           req_rd,        req_wb,        req_data})
    );

    // Can accept new request?
    assign lsu_req_if.ready = ~stall_in;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0] rsp_pc;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;
    wire [`LSU_BITS-1:0] rsp_type;
    wire rsp_is_dup;

    `UNUSED_VAR (rsp_type)
    
    reg [`LSUQ_SIZE-1:0][`NUM_THREADS-1:0] rsp_rem_mask;         
    wire [`NUM_THREADS-1:0] rsp_rem_mask_n;
    wire [`NUM_THREADS-1:0] rsp_tmask;

    reg [`NUM_THREADS-1:0] req_sent_mask;
    wire req_ready_all;

    wire [`DCORE_TAG_ID_BITS-1:0] mbuf_waddr, mbuf_raddr;
    wire mbuf_full;

    wire [`NUM_THREADS-1:0][1:0] req_offset, rsp_offset;
    for (genvar i = 0; i < `NUM_THREADS; i++) begin  
        assign req_offset[i] = req_addr[i][1:0];
    end

    wire [`NUM_THREADS-1:0] dcache_req_fire = dcache_req_if.valid & dcache_req_if.ready;

    wire dcache_rsp_fire = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;

    wire mbuf_push = (| dcache_req_fire)
                  && (0 == req_sent_mask)  // first submission only
                  && req_wb;               // loads only

    wire mbuf_pop = dcache_rsp_fire && (0 == rsp_rem_mask_n);
    
    assign mbuf_raddr = dcache_rsp_if.tag[`DCORE_TAG_ID_BITS-1:0];    

    VX_index_buffer #(
        .DATAW   (`NW_BITS + 32 + `NUM_THREADS + `NR_BITS + 1 + `LSU_BITS + (`NUM_THREADS * 2) + 1),
        .SIZE    (`LSUQ_SIZE)
    ) req_metadata (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (mbuf_waddr),  
        .acquire_slot (mbuf_push),       
        .read_addr    (mbuf_raddr),
        .write_data   ({req_wid, req_pc, req_tmask, req_rd, req_wb, req_type, req_offset, req_is_dup}),                    
        .read_data    ({rsp_wid, rsp_pc, rsp_tmask, rsp_rd, rsp_wb, rsp_type, rsp_offset, rsp_is_dup}),
        .release_addr (mbuf_raddr),
        .release_slot (mbuf_pop),     
        .full         (mbuf_full),
        `UNUSED_PIN (empty)
    );

    assign req_ready_all = &(dcache_req_if.ready | req_sent_mask | ~req_tmask);

    wire [`NUM_THREADS-1:0] req_sent_dup = {{(`NUM_THREADS-1){dcache_req_fire[0] && req_is_dup}}, 1'b0};

    always @(posedge clk) begin
        if (reset) begin
            req_sent_mask <= 0;
        end else begin
            if (req_ready_all)
                req_sent_mask <= 0;
            else
                req_sent_mask <= req_sent_mask | dcache_req_fire | req_sent_dup;
        end
    end

    wire is_req_start = (0 == req_sent_mask);

    // need to hold the acquired tag index until the full request is submitted
    reg [`DCORE_TAG_ID_BITS-1:0] req_tag_hold;
    wire [`DCORE_TAG_ID_BITS-1:0] req_tag = is_req_start ? mbuf_waddr : req_tag_hold;
    always @(posedge clk) begin
        if (mbuf_push) begin            
            req_tag_hold <= mbuf_waddr;
        end
    end

    wire [`NUM_THREADS-1:0] req_tmask_dup = req_tmask & {{(`NUM_THREADS-1){~req_is_dup}}, 1'b1};

    assign rsp_rem_mask_n = rsp_rem_mask[mbuf_raddr] & ~dcache_rsp_if.valid;

    always @(posedge clk) begin
        if (mbuf_push)  begin
            rsp_rem_mask[mbuf_waddr] <= req_tmask_dup;
        end    
        if (dcache_rsp_fire) begin
            rsp_rem_mask[mbuf_raddr] <= rsp_rem_mask_n;
        end
    end

    // ensure all dependencies for the requests are resolved
    wire req_dep_ready = (req_wb && (~mbuf_full || ~is_req_start)) 
                      || (~req_wb && st_commit_if.ready);

    // DCache Request

    reg [`NUM_THREADS-1:0][29:0] mem_req_addr;    
    reg [`NUM_THREADS-1:0][3:0]  mem_req_byteen;    
    reg [`NUM_THREADS-1:0][31:0] mem_req_data;

    always @(*) begin
        for (integer i = 0; i < `NUM_THREADS; i++) begin
            mem_req_byteen[i] = {4{req_wb}};
            case (`LSU_WSIZE(req_type))
                0: mem_req_byteen[i][req_offset[i]] = 1;
                1: begin
                    mem_req_byteen[i][req_offset[i]] = 1;
                    mem_req_byteen[i][{req_addr[i][1], 1'b1}] = 1;
                end
                default : mem_req_byteen[i] = {4{1'b1}};
            endcase

            mem_req_data[i] = 'x;
            case (req_offset[i])
                1:       mem_req_data[i][31:8]  = req_data[i][23:0];
                2:       mem_req_data[i][31:16] = req_data[i][15:0];
                3:       mem_req_data[i][31:24] = req_data[i][7:0];
                default: mem_req_data[i]        = req_data[i];
            endcase

            mem_req_addr[i] = req_addr[i][31:2];
        end
    end

    assign dcache_req_if.valid  = {`NUM_THREADS{req_valid && req_dep_ready}} & req_tmask_dup & ~req_sent_mask;
    assign dcache_req_if.rw     = {`NUM_THREADS{~req_wb}};
    assign dcache_req_if.addr   = mem_req_addr;
    assign dcache_req_if.byteen = mem_req_byteen;
    assign dcache_req_if.data   = mem_req_data;

`ifdef DBG_CACHE_REQ_INFO
    assign dcache_req_if.tag = {`NUM_THREADS{req_pc, req_wid, req_tag}};
`else
    assign dcache_req_if.tag = {`NUM_THREADS{req_tag}};
`endif
    
    assign ready_in = req_dep_ready && req_ready_all;

    // send store commit

    wire is_store_rsp = req_valid && ~req_wb && req_ready_all;

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
        wire [31:0] src_data = (i == 0 || rsp_is_dup) ? dcache_rsp_if.data[0] : dcache_rsp_if.data[i];

        reg [31:0] rsp_data_shifted;
        always @(*) begin
            rsp_data_shifted[31:16] = src_data[31:16];
            rsp_data_shifted[15:0]  = rsp_offset[i][1] ? src_data[31:16] : src_data[15:0];
            rsp_data_shifted[7:0]   = rsp_offset[i][0] ? rsp_data_shifted[15:8] : rsp_data_shifted[7:0];
        end

        always @(*) begin
            case (`LSU_FMT(rsp_type))
            `FMT_B:  rsp_data[i] = 32'(signed'(rsp_data_shifted[7:0]));
            `FMT_H:  rsp_data[i] = 32'(signed'(rsp_data_shifted[15:0]));
            `FMT_BU: rsp_data[i] = 32'(unsigned'(rsp_data_shifted[7:0]));
            `FMT_HU: rsp_data[i] = 32'(unsigned'(rsp_data_shifted[15:0]));
            default: rsp_data[i] = rsp_data_shifted;     
            endcase
        end        
    end   

    assign rsp_tmask_qual = rsp_is_dup ? rsp_tmask : dcache_rsp_if.valid;

    // send load commit

    wire load_rsp_stall = ~ld_commit_if.ready && ld_commit_if.valid;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!load_rsp_stall),
        .data_in  ({(| dcache_rsp_if.valid), rsp_wid,          rsp_tmask_qual,     rsp_pc,          rsp_rd,          rsp_wb,          rsp_data,          mbuf_pop}),
        .data_out ({ld_commit_if.valid,      ld_commit_if.wid, ld_commit_if.tmask, ld_commit_if.PC, ld_commit_if.rd, ld_commit_if.wb, ld_commit_if.data, ld_commit_if.eop})
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
    `SCOPE_ASSIGN (dcache_rsp_fire,  dcache_rsp_if.valid & {`NUM_THREADS{dcache_rsp_if.ready}});
    `SCOPE_ASSIGN (dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN (dcache_rsp_tag,   mbuf_raddr);
    
`ifdef DBG_PRINT_CORE_DCACHE
`IGNORE_WARNINGS_BEGIN
    reg [`LSUQ_SIZE-1:0][`DCORE_TAG_WIDTH:0] pending_reqs;
`IGNORE_WARNINGS_END

    always @(posedge clk) begin
        if (reset) begin
            pending_reqs <= '0;
        end else if (mbuf_push) begin            
            pending_reqs[mbuf_waddr] <= {dcache_req_if.tag[0], 1'b1};
        end else if (mbuf_pop) begin            
            pending_reqs[mbuf_raddr] <= '0;
        end
    end

   always @(posedge clk) begin        
        if ((| dcache_req_fire)) begin
            if (dcache_req_if.rw[0]) begin
                $write("%t: D$%0d Wr Req: wid=%0d, PC=%0h, tmask=%b, addr=", $time, CORE_ID, req_wid, req_pc, dcache_req_fire);
                `PRINT_ARRAY1D(req_addr, `NUM_THREADS);
                 $write(", tag=%0h, byteen=%0h, data=", dcache_req_if.tag[0], dcache_req_if.byteen);
                 `PRINT_ARRAY1D(dcache_req_if.data, `NUM_THREADS);
                 $write("\n");
            end else begin
                 $write("%t: D$%0d Rd Req: wid=%0d, PC=%0h, tmask=%b, addr=", $time, CORE_ID, req_wid, req_pc, dcache_req_fire);
                `PRINT_ARRAY1D(req_addr, `NUM_THREADS);
                 $write(", tag=%0h, byteen=%0h, rd=%0d, is_dup=%b\n", dcache_req_if.tag[0], dcache_req_if.byteen, req_rd, req_is_dup);
            end
        end
        if (dcache_rsp_fire) begin
            $write("%t: D$%0d Rsp: valid=%b, wid=%0d, PC=%0h, tag=%0h, rd=%0d, data=", 
                $time, CORE_ID, dcache_rsp_if.valid, rsp_wid, rsp_pc, dcache_rsp_if.tag, rsp_rd);
            `PRINT_ARRAY1D(dcache_rsp_if.data, `NUM_THREADS);
            $write(", is_dup=%b\n", rsp_is_dup);
        end
        if (mbuf_full) begin
            $write("%t: *** D$%0d queue-full:", $time, CORE_ID);
            for (integer j = 0; j < `LSUQ_SIZE; j++) begin
                if (pending_reqs[j][0]) begin
                    $write(" %0d->%0h", j, pending_reqs[j][1 +: `DCORE_TAG_WIDTH]);
                end
            end            
            $write("\n");
        end
    end
`endif
    
endmodule