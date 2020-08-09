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

    wire                          use_valid;
    wire [`NUM_THREADS-1:0]       use_thread_mask;
    wire                          use_req_rw;
    wire [`NUM_THREADS-1:0][29:0] use_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  use_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  use_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] use_req_data;    
    wire [1:0]                    use_req_sext; 
    wire [`NR_BITS-1:0]           use_rd;
    wire [`NW_BITS-1:0]           use_warp_num;
    wire [`ISTAG_BITS-1:0]        use_issue_tag;
    wire                          use_wb;
    wire [31:0]                   use_pc;

    genvar i;

    wire [`NUM_THREADS-1:0][31:0] full_address;    
    for (i = 0; i < `NUM_THREADS; i++) begin
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

    for (i = 0; i < `NUM_THREADS; i++) begin  
        assign mem_req_addr[i]   = full_address[i][31:2];        
        assign mem_req_offset[i] = full_address[i][1:0];
        assign mem_req_byteen[i] = wmask << full_address[i][1:0];
        assign mem_req_data[i]   = lsu_req_if.store_data[i] << {mem_req_offset[i], 3'b0};
    end   

     wire stall_in = ~dcache_req_if.ready && use_valid;

    // Can accept new request?
    assign lsu_req_if.ready = ~stall_in;   

`IGNORE_WARNINGS_BEGIN
    wire [`NUM_THREADS-1:0][31:0] use_address;
`IGNORE_WARNINGS_END

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + `ISTAG_BITS + (`NUM_THREADS * 32) + 2 + 1 + (`NUM_THREADS * (30 + 2 + 4 + 32)) +  `NR_BITS + 1 + 32)
    ) lsu_req_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_in),
        .flush (0),
        .in    ({lsu_req_if.valid, lsu_req_if.warp_num, lsu_req_if.thread_mask, lsu_req_if.issue_tag, full_address, mem_req_sext, lsu_req_if.rw, mem_req_addr, mem_req_offset, mem_req_byteen, mem_req_data, lsu_req_if.rd, lsu_req_if.wb, lsu_req_if.curr_PC}),
        .out   ({use_valid,        use_warp_num,        use_thread_mask,        use_issue_tag,        use_address,  use_req_sext, use_req_rw,    use_req_addr, use_req_offset, use_req_byteen, use_req_data, use_rd,        use_wb,        use_pc})
    );

    reg [`NUM_THREADS-1:0]      mem_rsp_mask_buf [`ISSUEQ_SIZE-1:0]; 
    reg [`NUM_THREADS-1:0][1:0] mem_rsp_offset_buf [`ISSUEQ_SIZE-1:0];
    reg [1:0]                   mem_rsp_sext_buf [`ISSUEQ_SIZE-1:0];  
    reg [`NUM_THREADS-1:0][31:0] mem_rsp_data_all_buf [`ISSUEQ_SIZE-1:0];
    reg [`NW_BITS-1:0]          mem_rsp_warp_num_buf [`ISSUEQ_SIZE-1:0];
    reg [31:0]                  mem_rsp_curr_PC_buf [`ISSUEQ_SIZE-1:0];
    reg [`NR_BITS-1:0]          mem_rsp_rd_buf [`ISSUEQ_SIZE-1:0];

    reg [`NUM_THREADS-1:0][31:0] mem_rsp_data_curr;  

    wire [`ISTAG_BITS-1:0] rsp_issue_tag = dcache_rsp_if.tag[0][`ISTAG_BITS-1:0];    

    wire [`NUM_THREADS-1:0]     mem_rsp_mask       = mem_rsp_mask_buf [rsp_issue_tag]; 
    wire [`NUM_THREADS-1:0][1:0] mem_rsp_offset    = mem_rsp_offset_buf [rsp_issue_tag]; 
    wire [1:0]                  mem_rsp_sext       = mem_rsp_sext_buf [rsp_issue_tag]; 
    wire [`NUM_THREADS-1:0][31:0] mem_rsp_data_all = mem_rsp_data_all_buf [rsp_issue_tag]; 
    wire [`NW_BITS-1:0]         mem_rsp_warp_num   = mem_rsp_warp_num_buf [rsp_issue_tag]; 
    wire [31:0]                 mem_rsp_curr_PC    = mem_rsp_curr_PC_buf [rsp_issue_tag]; 
    wire [`NR_BITS-1:0]         mem_rsp_rd         = mem_rsp_rd_buf [rsp_issue_tag]; 

    wire [`NUM_THREADS-1:0] mem_rsp_mask_n = mem_rsp_mask & ~dcache_rsp_if.valid;

    wire dcache_req_fire = (| dcache_req_if.valid) && dcache_req_if.ready;
    wire dcache_rsp_fire = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;  

    always @(posedge clk) begin
        if (dcache_req_fire && (0 == use_req_rw))  begin
            mem_rsp_mask_buf [use_issue_tag]     <= use_thread_mask;
            mem_rsp_offset_buf [use_issue_tag]   <= use_req_offset;
            mem_rsp_sext_buf [use_issue_tag]     <= use_req_sext;
            mem_rsp_data_all_buf [use_issue_tag] <= 0;   
            mem_rsp_warp_num_buf [use_issue_tag] <= use_warp_num;   
            mem_rsp_curr_PC_buf [use_issue_tag]  <= use_pc;   
            mem_rsp_rd_buf [use_issue_tag]       <= use_rd;   
        end    
        if (dcache_rsp_fire) begin
            mem_rsp_mask_buf [rsp_issue_tag]     <= mem_rsp_mask_n;   
            mem_rsp_data_all_buf [rsp_issue_tag] <= mem_rsp_data_all | mem_rsp_data_curr;
        end
    end

    // Core Request
    assign dcache_req_if.valid  = {`NUM_THREADS{use_valid}} & use_thread_mask;
    assign dcache_req_if.rw     = {`NUM_THREADS{use_req_rw}};
    assign dcache_req_if.byteen = use_req_byteen;
    assign dcache_req_if.addr   = use_req_addr;
    assign dcache_req_if.data   = use_req_data;  

`ifdef DBG_CORE_REQ_INFO
    assign dcache_req_if.tag = {use_pc, use_wb, use_rd, use_warp_num, use_issue_tag};
`else
    assign dcache_req_if.tag = use_issue_tag;
`endif

    // Core Response   
    for (i = 0; i < `NUM_THREADS; i++) begin        
        wire [31:0] rsp_data_shifted = dcache_rsp_if.data[i] >> {mem_rsp_offset[i], 3'b0};
        always @(*) begin
            case (mem_rsp_sext)
                  1: mem_rsp_data_curr[i] = {{24{rsp_data_shifted[7]}}, rsp_data_shifted[7:0]};  
                  2: mem_rsp_data_curr[i] = {{16{rsp_data_shifted[15]}}, rsp_data_shifted[15:0]};
            default: mem_rsp_data_curr[i] = rsp_data_shifted;     
            endcase
        end        
    end   

    wire is_store_rsp = dcache_req_fire && use_req_rw;
    wire is_load_rsp  = (| dcache_rsp_if.valid) && (0 == mem_rsp_mask_n);

    assign lsu_commit_if.valid     = is_load_rsp || is_store_rsp;
    assign lsu_commit_if.issue_tag = is_store_rsp ? use_issue_tag : rsp_issue_tag;
    assign lsu_commit_if.data      = mem_rsp_data_curr | mem_rsp_data_all;

    // Can accept new cache response?
    assign dcache_rsp_if.ready = lsu_commit_if.ready && ~is_store_rsp; // STORE has priority

    // scope registration
    `SCOPE_ASSIGN (scope_dcache_req_valid, dcache_req_if.valid);   
    `SCOPE_ASSIGN (scope_dcache_req_addr,  use_address);    
    `SCOPE_ASSIGN (scope_dcache_req_rw,    dcache_req_if.rw );
    `SCOPE_ASSIGN (scope_dcache_req_byteen,dcache_req_if.byteen);
    `SCOPE_ASSIGN (scope_dcache_req_data,  dcache_req_if.data);
    `SCOPE_ASSIGN (scope_dcache_req_tag,   dcache_req_if.tag);
    `SCOPE_ASSIGN (scope_dcache_req_ready, dcache_req_if.ready); 
    `SCOPE_ASSIGN (scope_dcache_req_warp_num, use_warp_num);
    `SCOPE_ASSIGN (scope_dcache_req_curr_PC, use_pc);

    `SCOPE_ASSIGN (scope_dcache_rsp_valid, dcache_rsp_if.valid);
    `SCOPE_ASSIGN (scope_dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN (scope_dcache_rsp_tag,   dcache_rsp_if.tag);
    `SCOPE_ASSIGN (scope_dcache_rsp_ready, dcache_rsp_if.ready);

    `UNUSED_VAR (mem_rsp_warp_num)
    `UNUSED_VAR (mem_rsp_curr_PC)
    `UNUSED_VAR (mem_rsp_rd)
    `UNUSED_VAR (use_wb)
    
`ifdef DBG_PRINT_CORE_DCACHE
   always @(posedge clk) begin
        if ((| dcache_req_if.valid) && dcache_req_if.ready) begin
            $display("%t: D$%0d req: warp=%0d, PC=%0h, tmask=%b, addr=%0h, tag=%0h, rd=%0d, rw=%0b, byteen=%0h, data=%0h", 
                     $time, CORE_ID, use_warp_num, use_pc, dcache_req_if.valid, use_address, dcache_req_if.tag, use_rd, dcache_req_if.rw, dcache_req_if.byteen, dcache_req_if.data);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D$%0d rsp: valid=%b, warp=%0d, PC=%0h, tag=%0h, rd=%0d, data=%0h", 
                     $time, CORE_ID, dcache_rsp_if.valid, mem_rsp_warp_num, mem_rsp_curr_PC, dcache_rsp_if.tag, mem_rsp_rd, dcache_rsp_if.data);
        end
    end
`endif
    
endmodule
