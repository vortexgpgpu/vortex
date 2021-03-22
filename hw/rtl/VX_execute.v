`include "VX_define.vh"

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_execute

    input wire clk, 
    input wire reset, 

    // CSR io interface
    VX_csr_io_req_if    csr_io_req_if,
    VX_csr_io_rsp_if    csr_io_rsp_if,    

    // Dcache interface
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // commit status
    VX_cmt_to_csr_if    cmt_to_csr_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if    perf_memsys_if,
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
    
    input wire          busy,
    output wire         ebreak
);
    VX_fpu_to_csr_if     fpu_to_csr_if(); 

`ifdef EXT_TEX_ENABLE

    VX_dcache_core_req_if #(
        .LANES(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`LSU_DACHE_TAG_BITS)
    ) lsu_dcache_req_if();

    VX_dcache_core_rsp_if #(
        .LANES(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`LSU_DACHE_TAG_BITS)
    ) lsu_dcache_rsp_if();    

    VX_dcache_core_req_if #(
        .LANES(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`TEX_DACHE_TAG_BITS)
    ) tex_dcache_req_if();

    VX_dcache_core_rsp_if #(
        .LANES(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`TEX_DACHE_TAG_BITS)
    ) tex_dcache_rsp_if();

    VX_tex_csr_if  tex_csr_if();

    wire [`NUM_THREADS-1:0][`LSU_TEX_DACHE_TAG_BITS-1:0] lsu_tag_in;
    wire [`LSU_TEX_DACHE_TAG_BITS-1:0] lsu_tag_out;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign lsu_tag_in[i][`LSUQ_ADDR_BITS-1:0] = lsu_dcache_req_if.tag[i][`LSUQ_ADDR_BITS-1:0];
        assign lsu_tag_in[i][`LSUQ_ADDR_BITS+:2] = '0;
        assign lsu_tag_in[i][(`LSUQ_ADDR_BITS+2)+:`DBG_CACHE_REQ_MDATAW] = lsu_dcache_req_if.tag[i][`LSUQ_ADDR_BITS+:`DBG_CACHE_REQ_MDATAW];
    end
    assign lsu_dcache_rsp_if.tag[`LSUQ_ADDR_BITS-1:0] = lsu_tag_out[`LSUQ_ADDR_BITS-1:0];
    assign lsu_dcache_rsp_if.tag[`LSUQ_ADDR_BITS+:`DBG_CACHE_REQ_MDATAW] = lsu_tag_out[(`LSUQ_ADDR_BITS+2)+:`DBG_CACHE_REQ_MDATAW];
    `UNUSED_VAR (lsu_tag_out)

    VX_tex_lsu_arb #(
        .NUM_REQS      (2),
        .LANES         (`NUM_THREADS),
        .WORD_SIZE     (4),            
        .TAG_IN_WIDTH  (`LSU_TEX_DACHE_TAG_BITS),
        .TAG_OUT_WIDTH (`DCORE_TAG_WIDTH)
    ) tex_lsu_arb (
        .clk            (clk),
        .reset          (reset),

        // Tex/LSU request
        .req_valid_in   ({tex_dcache_req_if.valid,  lsu_dcache_req_if.valid}),
        .req_rw_in      ({tex_dcache_req_if.rw,     lsu_dcache_req_if.rw}),
        .req_byteen_in  ({tex_dcache_req_if.byteen, lsu_dcache_req_if.byteen}),
        .req_addr_in    ({tex_dcache_req_if.addr,   lsu_dcache_req_if.addr}),
        .req_data_in    ({tex_dcache_req_if.data,   lsu_dcache_req_if.data}),  
        .req_tag_in     ({tex_dcache_req_if.tag,    lsu_tag_in}),  
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
        .rsp_tag_in     (dcache_rsp_if.tag),
        .rsp_data_in    (dcache_rsp_if.data),
        .rsp_ready_in   (dcache_rsp_if.ready),

        // Tex/LSU response
        .rsp_valid_out  ({tex_dcache_rsp_if.valid, lsu_dcache_rsp_if.valid}),
        .rsp_data_out   ({tex_dcache_rsp_if.data,  lsu_dcache_rsp_if.data}),
        .rsp_tag_out    ({tex_dcache_rsp_if.tag,   lsu_tag_out}),
        .rsp_ready_out  ({tex_dcache_rsp_if.ready, lsu_dcache_rsp_if.ready})
    );

`endif

    wire[`NUM_WARPS-1:0] csr_pending;
    wire[`NUM_WARPS-1:0] fpu_pending;
    
    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .alu_req_if     (alu_req_if),
        .branch_ctl_if  (branch_ctl_if),
        .alu_commit_if  (alu_commit_if)
    );

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_BIND_VX_execute_lsu_unit
        .clk            (clk),
        .reset          (reset),
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
        .reset          (reset),   
    `ifdef PERF_ENABLE
        .perf_memsys_if  (perf_memsys_if),
        .perf_pipeline_if (perf_pipeline_if),
    `endif    
        .cmt_to_csr_if  (cmt_to_csr_if),    
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
    `endif
        .csr_io_req_if  (csr_io_req_if),           
        .csr_io_rsp_if  (csr_io_rsp_if),
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if),
        .fpu_pending    (fpu_pending),
        .pending        (csr_pending),
        .busy           (busy)
    );

`ifdef EXT_F_ENABLE
    VX_fpu_unit #(
        .CORE_ID(CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),        
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
        .reset          (reset),    
        .gpu_req_if     (gpu_req_if),
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
        .dcache_req_if  (tex_dcache_req_if),
        .dcache_rsp_if  (tex_dcache_rsp_if),
    `endif
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    assign ebreak = alu_req_if.valid 
                 && `IS_BR_MOD(alu_req_if.op_mod)
                 && (`BR_OP(alu_req_if.op_type) == `BR_EBREAK 
                  || `BR_OP(alu_req_if.op_type) == `BR_ECALL);

endmodule
