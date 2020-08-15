`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_SIGNALS_LSU_IO

    input wire clk,
    input wire reset,

   // Dcache interface
    VX_cache_core_req_if dcache_req_if,
    VX_cache_core_rsp_if dcache_rsp_if,

    // inputs
    VX_lsu_req_if       lsu_req_if,

    // outputs
    VX_exu_to_cmt_if    lsu_commit_if
);

    wire                          valid_in;
    wire                          ready_in;

    wire [`NUM_THREADS-1:0]       req_thread_mask;
    wire                          req_rw;
    wire [`NUM_THREADS-1:0][29:0] req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  req_byteen;
    wire [`NUM_THREADS-1:0][31:0] req_data;    
    wire [1:0]                    req_sext; 
    wire [`NR_BITS-1:0]           req_rd;
    wire [`NW_BITS-1:0]           req_wid;
    wire [`ISTAG_BITS-1:0]        req_issue_tag;
    wire                          req_wb;
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
        assign mem_req_data[i]   = lsu_req_if.store_data[i] << {mem_req_offset[i], 3'b0};
    end

`IGNORE_WARNINGS_BEGIN
    wire [`NUM_THREADS-1:0][31:0] req_address;
`IGNORE_WARNINGS_END

    // use a skid buffer because the dcache's ready signal is combinational
    // use buffer size of two for stall-free execution
    VX_elastic_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + `ISTAG_BITS + (`NUM_THREADS * 32) + 2 + 1 + (`NUM_THREADS * (30 + 2 + 4 + 32)) +  `NR_BITS + 1 + 32),
        .SIZE  (2)
    ) input_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_req_if.valid),
        .ready_in  (lsu_req_if.ready),
        .data_in   ({lsu_req_if.wid, lsu_req_if.thread_mask, lsu_req_if.issue_tag, full_address, mem_req_sext, lsu_req_if.rw, mem_req_addr, mem_req_offset, mem_req_byteen, mem_req_data, lsu_req_if.rd, lsu_req_if.wb, lsu_req_if.curr_PC}),
        .data_out  ({req_wid,        req_thread_mask,        req_issue_tag,        req_address,  req_sext,     req_rw,        req_addr,     req_offset,     req_byteen,     req_data,     req_rd,        req_wb,        req_pc}),        
        .ready_out (ready_in),
        .valid_out (valid_in)
    );

    reg [`ISSUEQ_SIZE-1:0][`NUM_THREADS-1:0]       mem_rsp_mask_buf; 
    reg [`ISSUEQ_SIZE-1:0][`NUM_THREADS-1:0][31:0] mem_rsp_data_prev_buf;

    reg [`NUM_THREADS-1:0][1:0] mem_rsp_offset_buf [`ISSUEQ_SIZE-1:0];
    reg [1:0]                   mem_rsp_sext_buf [`ISSUEQ_SIZE-1:0];      
    reg [`NW_BITS-1:0]          mem_rsp_wid_buf [`ISSUEQ_SIZE-1:0];
    reg [31:0]                  mem_rsp_curr_PC_buf [`ISSUEQ_SIZE-1:0];
    reg [`NR_BITS-1:0]          mem_rsp_rd_buf [`ISSUEQ_SIZE-1:0];

    reg [`NUM_THREADS-1:0][31:0] mem_rsp_data_curr;  

    wire [`ISTAG_BITS-1:0] rsp_issue_tag = dcache_rsp_if.tag[0][`ISTAG_BITS-1:0];    

    wire [`NUM_THREADS-1:0]     mem_rsp_mask       = mem_rsp_mask_buf [rsp_issue_tag]; 
    wire [`NUM_THREADS-1:0][1:0] mem_rsp_offset    = mem_rsp_offset_buf [rsp_issue_tag]; 
    wire [1:0]                  mem_rsp_sext       = mem_rsp_sext_buf [rsp_issue_tag]; 
    wire [`NUM_THREADS-1:0][31:0] mem_rsp_data_prev= mem_rsp_data_prev_buf [rsp_issue_tag]; 
    wire [`NW_BITS-1:0]         mem_rsp_wid   = mem_rsp_wid_buf [rsp_issue_tag]; 
    wire [31:0]                 mem_rsp_curr_PC    = mem_rsp_curr_PC_buf [rsp_issue_tag]; 
    wire [`NR_BITS-1:0]         mem_rsp_rd         = mem_rsp_rd_buf [rsp_issue_tag]; 

    wire dcache_req_fire = (| dcache_req_if.valid) && dcache_req_if.ready;
    wire dcache_rsp_fire = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;  

    wire [`NUM_THREADS-1:0] mem_rsp_mask_n = mem_rsp_mask & ~dcache_rsp_if.valid;

    always @(posedge clk) begin
        if (dcache_req_fire && (0 == req_rw))  begin
            mem_rsp_mask_buf [req_issue_tag]      <= req_thread_mask;
            mem_rsp_data_prev_buf [req_issue_tag] <= 0;   
        end    
        if (dcache_rsp_fire) begin
            mem_rsp_mask_buf [rsp_issue_tag]      <= mem_rsp_mask_n;   
            mem_rsp_data_prev_buf [rsp_issue_tag] <= mem_rsp_data_curr | mem_rsp_data_prev;
        end
    end

    always @(posedge clk) begin
        if (dcache_req_fire && (0 == req_rw))  begin
            mem_rsp_offset_buf [req_issue_tag]   <= req_offset;
            mem_rsp_sext_buf [req_issue_tag]     <= req_sext;            
            mem_rsp_wid_buf [req_issue_tag] <= req_wid;   
            mem_rsp_curr_PC_buf [req_issue_tag]  <= req_pc;   
            mem_rsp_rd_buf [req_issue_tag]       <= req_rd;   
        end    
    end

    wire stall_in;

    // Core Request
    assign dcache_req_if.valid  = {`NUM_THREADS{valid_in && ~stall_in}} & req_thread_mask;
    assign dcache_req_if.rw     = {`NUM_THREADS{req_rw}};
    assign dcache_req_if.byteen = req_byteen;
    assign dcache_req_if.addr   = req_addr;
    assign dcache_req_if.data   = req_data;  

    assign ready_in = dcache_req_if.ready && ~stall_in;

`ifdef DBG_CORE_REQ_INFO
    assign dcache_req_if.tag = {req_pc, req_wb, req_rd, req_wid, req_issue_tag};
`else
    assign dcache_req_if.tag = req_issue_tag;
`endif

    // Core Response   
    for (genvar i = 0; i < `NUM_THREADS; i++) begin        
        wire [31:0] rsp_data_shifted = dcache_rsp_if.data[i] >> {mem_rsp_offset[i], 3'b0};
        always @(*) begin
            case (mem_rsp_sext)
                  1: mem_rsp_data_curr[i] = {{24{rsp_data_shifted[7]}}, rsp_data_shifted[7:0]};  
                  2: mem_rsp_data_curr[i] = {{16{rsp_data_shifted[15]}}, rsp_data_shifted[15:0]};
            default: mem_rsp_data_curr[i] = rsp_data_shifted;     
            endcase
        end        
    end   

    reg is_load_rsp;
    reg [`NUM_THREADS-1:0][31:0] load_data;
    reg [`ISTAG_BITS-1:0] rsp_issue_tag_r;

    always @(posedge clk) begin
        if (reset) begin
            is_load_rsp <= 0;
        end else begin
            is_load_rsp     <= dcache_rsp_fire && (0 == mem_rsp_mask_n);
            load_data       <= mem_rsp_data_curr | mem_rsp_data_prev;
            rsp_issue_tag_r <= rsp_issue_tag;
        end    
    end

    wire is_store_req = dcache_req_fire && req_rw;
    assign stall_in = is_load_rsp && valid_in && req_rw; // LOAD has priority

    assign lsu_commit_if.valid     = is_load_rsp || is_store_req;
    assign lsu_commit_if.issue_tag = is_load_rsp ? rsp_issue_tag_r : req_issue_tag;
    assign lsu_commit_if.data      = load_data;

    // Can accept new cache response?
    assign dcache_rsp_if.ready = 1'b1; 

    // scope registration
    `SCOPE_ASSIGN (scope_dcache_req_valid, dcache_req_if.valid);   
    `SCOPE_ASSIGN (scope_dcache_req_addr,  req_address);    
    `SCOPE_ASSIGN (scope_dcache_req_rw,    dcache_req_if.rw );
    `SCOPE_ASSIGN (scope_dcache_req_byteen,dcache_req_if.byteen);
    `SCOPE_ASSIGN (scope_dcache_req_data,  dcache_req_if.data);
    `SCOPE_ASSIGN (scope_dcache_req_tag,   dcache_req_if.tag);
    `SCOPE_ASSIGN (scope_dcache_req_ready, dcache_req_if.ready); 
    `SCOPE_ASSIGN (scope_dcache_req_wid, req_wid);
    `SCOPE_ASSIGN (scope_dcache_req_curr_PC, req_pc);

    `SCOPE_ASSIGN (scope_dcache_rsp_valid, dcache_rsp_if.valid);
    `SCOPE_ASSIGN (scope_dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN (scope_dcache_rsp_tag,   dcache_rsp_if.tag);
    `SCOPE_ASSIGN (scope_dcache_rsp_ready, dcache_rsp_if.ready);

    `UNUSED_VAR (mem_rsp_wid)
    `UNUSED_VAR (mem_rsp_curr_PC)
    `UNUSED_VAR (mem_rsp_rd)
    `UNUSED_VAR (req_wb)
    
`ifdef DBG_PRINT_CORE_DCACHE
   always @(posedge clk) begin
        if ((| dcache_req_if.valid) && dcache_req_if.ready) begin
            $display("%t: D$%0d req: wid=%0d, PC=%0h, tmask=%b, addr=%0h, tag=%0h, rd=%0d, rw=%0b, byteen=%0h, data=%0h", 
                     $time, CORE_ID, req_wid, req_pc, dcache_req_if.valid, req_address, dcache_req_if.tag, req_rd, dcache_req_if.rw, dcache_req_if.byteen, dcache_req_if.data);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D$%0d rsp: valid=%b, wid=%0d, PC=%0h, tag=%0h, rd=%0d, data=%0h", 
                     $time, CORE_ID, dcache_rsp_if.valid, mem_rsp_wid, mem_rsp_curr_PC, dcache_rsp_if.tag, mem_rsp_rd, dcache_rsp_if.data);
        end
    end
`endif
    
endmodule
