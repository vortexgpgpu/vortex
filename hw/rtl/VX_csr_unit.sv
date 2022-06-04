`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_csr_unit #(
    parameter CORE_ID = 0
) (
    input wire                  clk,
    input wire                  reset,

    input base_dcrs_t           base_dcrs,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave     perf_memsys_if,
    VX_perf_pipeline_if.slave   perf_pipeline_if,
`endif

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if.master        tex_csr_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave        tex_perf_if,
    VX_perf_cache_if.slave      perf_tcache_if,
`endif
`endif
`ifdef EXT_RASTER_ENABLE
    VX_gpu_csr_if.master        raster_csr_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave     raster_perf_if,
    VX_perf_cache_if.slave      perf_rcache_if,
`endif
`endif
`ifdef EXT_ROP_ENABLE
    VX_gpu_csr_if.master        rop_csr_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave        rop_perf_if,
    VX_perf_cache_if.slave      perf_ocache_if,
`endif
`endif

    VX_cmt_to_csr_if.slave      cmt_to_csr_if,
    VX_fetch_to_csr_if.slave    fetch_to_csr_if,
    VX_csr_req_if.slave         csr_req_if,
    VX_commit_if.master         csr_commit_if,
    
`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave      fpu_to_csr_if,
    input wire[`NUM_WARPS-1:0]  fpu_pending,
`endif

    output wire[`NUM_WARPS-1:0] req_pending
);    
    
    wire [`NUM_THREADS-1:0][31:0] csr_req_data;
    wire [`NUM_THREADS-1:0][31:0] csr_read_data, csr_read_data_s0;
    reg [`NUM_THREADS-1:0][31:0]  csr_write_data_s0, csr_write_data_s1;
    wire [`UP(`NT_BITS)-1:0]      write_tid;
    wire                          write_enable;
    wire [`CSR_ADDR_BITS-1:0]     csr_addr_s1;
    reg                           csr_we_s0;
    wire                          csr_we_s1;

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if(perf_pipeline_if),
    `endif

        .cmt_to_csr_if  (cmt_to_csr_if),
        .fetch_to_csr_if(fetch_to_csr_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `endif        

    `ifdef EXT_TEX_ENABLE        
        .tex_csr_if     (tex_csr_if),    
    `ifdef PERF_ENABLE
        .tex_perf_if    (tex_perf_if),
        .perf_tcache_if (perf_tcache_if),
    `endif
    `endif
    
    `ifdef EXT_RASTER_ENABLE        
        .raster_csr_if  (raster_csr_if),
    `ifdef PERF_ENABLE
        .raster_perf_if (raster_perf_if),
        .perf_rcache_if (perf_rcache_if),
    `endif
    `endif

    `ifdef EXT_ROP_ENABLE
        .rop_csr_if     (rop_csr_if),
    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
        .perf_ocache_if (perf_ocache_if),
    `endif
    `endif

        .read_enable    (csr_req_if.valid),
        .read_uuid      (csr_req_if.uuid),
        .read_wid       (csr_req_if.wid),    
        .read_tmask     (csr_req_if.tmask),    
        .read_addr      (csr_req_if.addr),
        .read_data      (csr_read_data),

        .write_tid      (write_tid),
        .write_enable   (write_enable),       
        .write_uuid     (csr_commit_if.uuid),
        .write_wid      (csr_commit_if.wid),
        .write_tmask    (csr_commit_if.tmask),
        .write_addr     (csr_addr_s1),        
        .write_data     (csr_write_data_s1)
    );    
    
    wire write_hazard = (csr_addr_s1 == csr_req_if.addr)
                     && (csr_commit_if.wid == csr_req_if.wid) 
                     && csr_commit_if.valid;

    assign csr_req_data = csr_req_if.use_imm ? {`NUM_THREADS{32'(csr_req_if.imm)}} : csr_req_if.rs1_data;
    
    assign csr_read_data_s0 = write_hazard ? csr_write_data_s1 : csr_read_data;    
    
    always @(*) begin      
        csr_we_s0 = 0;  
        for (integer i = 0; i < `NUM_THREADS; ++i) begin
            csr_we_s0 |= (csr_req_data[i] != 0);
        end
        case (csr_req_if.op_type)
            `INST_CSR_RW: begin
                for (integer i = 0; i < `NUM_THREADS; ++i) begin
                    csr_write_data_s0[i] = csr_req_data[i];
                end
                csr_we_s0 = 1;
            end
            `INST_CSR_RS: begin
                for (integer i = 0; i < `NUM_THREADS; ++i) begin
                    csr_write_data_s0[i] = csr_read_data_s0[i] | csr_req_data[i];
                end
            end
            //`INST_CSR_RC
            default: begin
                for (integer i = 0; i < `NUM_THREADS; ++i) begin
                    csr_write_data_s0[i] = csr_read_data_s0[i] & ~csr_req_data[i];
                end
            end
        endcase
    end         

`ifdef EXT_F_ENABLE
    wire stall_in = fpu_pending[csr_req_if.wid];
`else 
    wire stall_in = 0;
`endif

    wire csr_rsp_valid = csr_req_if.valid && ~stall_in;  
    wire csr_rsp_ready;

    VX_skid_buffer #(
        .DATAW (`UUID_BITS + `UP(`NW_BITS) + `NUM_THREADS + 32 + `NR_BITS + 1 + 1 + `CSR_ADDR_BITS + `UP(`NT_BITS) + 2 * (`NUM_THREADS * 32))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_rsp_valid),
        .ready_in  (csr_rsp_ready),
        .data_in   ({csr_req_if.uuid,    csr_req_if.wid,    csr_req_if.tmask,    csr_req_if.PC,    csr_req_if.rd,    csr_req_if.wb,    csr_we_s0, csr_req_if.addr, csr_read_data_s0,   csr_req_if.tid, csr_write_data_s0}),
        .data_out  ({csr_commit_if.uuid, csr_commit_if.wid, csr_commit_if.tmask, csr_commit_if.PC, csr_commit_if.rd, csr_commit_if.wb, csr_we_s1, csr_addr_s1,     csr_commit_if.data, write_tid,      csr_write_data_s1}),
        .valid_out (csr_commit_if.valid),
        .ready_out (csr_commit_if.ready)
    );

    assign write_enable = csr_commit_if.valid && csr_we_s1;

    assign csr_commit_if.eop = 1'b1;

    // can accept new request?
    assign csr_req_if.ready = csr_rsp_ready && ~stall_in;

    // pending request
    reg [`NUM_WARPS-1:0] req_pending_r;
    always @(posedge clk) begin
        if (reset) begin
            req_pending_r <= 0;
        end else begin
            if (csr_req_if.valid && csr_req_if.ready) begin
                 req_pending_r[csr_req_if.wid] <= 1;
            end
            if (csr_commit_if.valid && csr_commit_if.ready) begin
                 req_pending_r[csr_commit_if.wid] <= 0;
            end
        end
    end
    assign req_pending = req_pending_r;

endmodule
