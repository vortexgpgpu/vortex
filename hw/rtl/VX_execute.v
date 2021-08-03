`include "VX_define.vh"

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_execute

    input wire clk, 
    input wire reset,    

    // Dcache interface
    VX_dcache_req_if    dcache_req_if,
    VX_dcache_rsp_if    dcache_rsp_if,

    // commit status
    VX_cmt_to_csr_if    cmt_to_csr_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if   perf_memsys_if,
    VX_perf_pipeline_if perf_pipeline_if,
 `endif
    
    // inputs    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,  
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if,
    
    // outputs
    VX_branch_ctl_if    branch_ctl_if,    
    VX_warp_ctl_if      warp_ctl_if,
    VX_commit_if        alu_commit_if,
    VX_commit_if        ld_commit_if,
    VX_commit_if        st_commit_if,
    VX_commit_if        csr_commit_if,
    VX_commit_if        fpu_commit_if,
    VX_commit_if        gpu_commit_if,
    
    input wire          busy
);
    VX_fpu_to_csr_if fpu_to_csr_if(); 

`ifdef EXT_TEX_ENABLE

    VX_dcache_req_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`LSU_DCACHE_TAG_BITS)
    ) lsu_dcache_req_if();

    VX_dcache_rsp_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`LSU_DCACHE_TAG_BITS)
    ) lsu_dcache_rsp_if();

    VX_dcache_req_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`TEX_DCACHE_TAG_BITS)
    ) tex_dcache_req_if();

    VX_dcache_rsp_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`TEX_DCACHE_TAG_BITS)
    ) tex_dcache_rsp_if();

    VX_tex_csr_if tex_csr_if();

    wire [`NUM_THREADS-1:0][`LSU_TEX_DCACHE_TAG_BITS-1:0] tex_tag_in, lsu_tag_in;
    wire [`LSU_TEX_DCACHE_TAG_BITS-1:0] tex_tag_out, lsu_tag_out;

    `UNUSED_VAR (tex_tag_out)
    `UNUSED_VAR (lsu_tag_out)

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign tex_tag_in[i][`LSU_TEX_TAG_ID_BITS-1:0] = `LSU_TEX_TAG_ID_BITS'(tex_dcache_req_if.tag[i][`TEX_TAG_ID_BITS-1:0]);
        assign lsu_tag_in[i][`LSU_TEX_TAG_ID_BITS-1:0] = `LSU_TEX_TAG_ID_BITS'(lsu_dcache_req_if.tag[i][`LSU_TAG_ID_BITS-1:0]);
    `ifdef DBG_CACHE_REQ_INFO
        assign tex_tag_in[i][`LSU_TEX_DCACHE_TAG_BITS-1:`LSU_TEX_TAG_ID_BITS] = tex_dcache_req_if.tag[i][`TEX_DCACHE_TAG_BITS-1:`TEX_TAG_ID_BITS];
        assign lsu_tag_in[i][`LSU_TEX_DCACHE_TAG_BITS-1:`LSU_TEX_TAG_ID_BITS] = lsu_dcache_req_if.tag[i][`LSU_DCACHE_TAG_BITS-1:`LSU_TAG_ID_BITS];
    `endif
    end

    assign tex_dcache_rsp_if.tag[`TEX_TAG_ID_BITS-1:0] = tex_tag_out[`TEX_TAG_ID_BITS-1:0];
    assign lsu_dcache_rsp_if.tag[`LSU_TAG_ID_BITS-1:0] = lsu_tag_out[`LSU_TAG_ID_BITS-1:0];
`ifdef DBG_CACHE_REQ_INFO
    assign tex_dcache_rsp_if.tag[`TEX_DCACHE_TAG_BITS-1:`TEX_TAG_ID_BITS] = tex_tag_out[`LSU_TEX_DCACHE_TAG_BITS-1:`LSU_TEX_TAG_ID_BITS];
    assign lsu_dcache_rsp_if.tag[`LSU_DCACHE_TAG_BITS-1:`LSU_TAG_ID_BITS] = lsu_tag_out[`LSU_TEX_DCACHE_TAG_BITS-1:`LSU_TEX_TAG_ID_BITS];
`endif

    VX_cache_arb #(
        .NUM_REQS      (2),
        .LANES         (`NUM_THREADS),
        .DATA_SIZE     (4),            
        .TAG_IN_WIDTH  (`LSU_TEX_DCACHE_TAG_BITS),
        .TAG_SEL_IDX   (`NC_ADDR_BITS + `SM_ENABLE)
    ) tex_lsu_arb (
        .clk            (clk),
        .reset          (reset),

        // Tex/LSU request
        .req_valid_in   ({tex_dcache_req_if.valid,  lsu_dcache_req_if.valid}),
        .req_rw_in      ({tex_dcache_req_if.rw,     lsu_dcache_req_if.rw}),
        .req_byteen_in  ({tex_dcache_req_if.byteen, lsu_dcache_req_if.byteen}),
        .req_addr_in    ({tex_dcache_req_if.addr,   lsu_dcache_req_if.addr}),
        .req_data_in    ({tex_dcache_req_if.data,   lsu_dcache_req_if.data}),  
        .req_tag_in     ({tex_tag_in,               lsu_tag_in}),  
        .req_ready_in   ({tex_dcache_req_if.ready,  lsu_dcache_req_if.ready}),

        // Dcache request
        .req_valid_out  (dcache_req_if.valid),
        .req_rw_out     (dcache_req_if.rw),        
        .req_byteen_out (dcache_req_if.byteen),        
        .req_addr_out   (dcache_req_if.addr),
        .req_data_out   (dcache_req_if.data),
        .req_tag_out    (dcache_req_if.tag),
        .req_ready_out  (dcache_req_if.ready),
        
        // Dcache response
        .rsp_valid_in   (dcache_rsp_if.valid),
        .rsp_tmask_in   (dcache_rsp_if.tmask),
        .rsp_tag_in     (dcache_rsp_if.tag),
        .rsp_data_in    (dcache_rsp_if.data),
        .rsp_ready_in   (dcache_rsp_if.ready),

        // Tex/LSU response
        .rsp_valid_out  ({tex_dcache_rsp_if.valid, lsu_dcache_rsp_if.valid}),
        .rsp_tmask_out  ({tex_dcache_rsp_if.tmask, lsu_dcache_rsp_if.tmask}),
        .rsp_data_out   ({tex_dcache_rsp_if.data,  lsu_dcache_rsp_if.data}),
        .rsp_tag_out    ({tex_tag_out,             lsu_tag_out}),
        .rsp_ready_out  ({tex_dcache_rsp_if.ready, lsu_dcache_rsp_if.ready})
    );

`endif

    wire [`NUM_WARPS-1:0] csr_pending;
    wire [`NUM_WARPS-1:0] fpu_pending;

    `RESET_RELAY (alu_reset);
    `RESET_RELAY (lsu_reset);
    `RESET_RELAY (csr_reset);
    `RESET_RELAY (gpu_reset);
    
    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (alu_reset),
        .alu_req_if     (alu_req_if),
        .branch_ctl_if  (branch_ctl_if),
        .alu_commit_if  (alu_commit_if)
    );

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_BIND_VX_execute_lsu_unit
        .clk            (clk),
        .reset          (lsu_reset),
    `ifdef EXT_TEX_ENABLE
        .dcache_req_if  (lsu_dcache_req_if),
        .dcache_rsp_if  (lsu_dcache_rsp_if),
    `else 
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
    `endif
        .lsu_req_if     (lsu_req_if),
        .ld_commit_if   (ld_commit_if),
        .st_commit_if   (st_commit_if)
    );

    VX_csr_unit #(
        .CORE_ID(CORE_ID)
    ) csr_unit (
        .clk            (clk),
        .reset          (csr_reset),   
    `ifdef PERF_ENABLE
        .perf_memsys_if  (perf_memsys_if),
        .perf_pipeline_if (perf_pipeline_if),
    `endif    
        .cmt_to_csr_if  (cmt_to_csr_if),    
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
    `endif
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if),
        .fpu_pending    (fpu_pending),
        .pending        (csr_pending),
        .busy           (busy)
    );

`ifdef EXT_F_ENABLE
    `RESET_RELAY (fpu_reset);

    VX_fpu_unit #(
        .CORE_ID(CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (fpu_reset),        
        .fpu_req_if     (fpu_req_if), 
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .fpu_commit_if  (fpu_commit_if),
        .csr_pending    (csr_pending),
        .pending        (fpu_pending) 
    );
`else
    `UNUSED_VAR (csr_pending)
    `UNUSED_VAR (fpu_to_csr_if.read_frm)
    assign fpu_req_if.ready     = 0;
    assign fpu_commit_if.valid  = 0;
    assign fpu_commit_if.wid    = 0;
    assign fpu_commit_if.PC     = 0;
    assign fpu_commit_if.tmask  = 0;
    assign fpu_commit_if.wb     = 0;
    assign fpu_commit_if.rd     = 0;
    assign fpu_commit_if.data   = 0;  
    assign fpu_to_csr_if.write_enable = 0;  
    assign fpu_to_csr_if.write_wid = 0;
    assign fpu_to_csr_if.write_fflags = 0;
    assign fpu_to_csr_if.read_wid = 0;
    assign fpu_pending = 0;
`endif

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        `SCOPE_BIND_VX_execute_gpu_unit
        .clk            (clk),
        .reset          (gpu_reset),    
        .gpu_req_if     (gpu_req_if),
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
        .dcache_req_if  (tex_dcache_req_if),
        .dcache_rsp_if  (tex_dcache_rsp_if),
    `endif
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    // special workaround to get RISC-V tests Pass/Fail status
    wire ebreak /* verilator public */;
    assign ebreak = alu_req_if.valid && alu_req_if.ready
                 && `ALU_IS_BR(alu_req_if.op_mod)
                 && (`BR_OP(alu_req_if.op_type) == `BR_EBREAK 
                  || `BR_OP(alu_req_if.op_type) == `BR_ECALL);

endmodule