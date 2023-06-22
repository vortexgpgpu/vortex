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
    VX_perf_gpu_if.slave        perf_gpu_if,
`endif

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if.master        tex_csr_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave        perf_tex_if,
    VX_perf_cache_if.slave      perf_tcache_if,
`endif
`endif
`ifdef EXT_RASTER_ENABLE
    VX_gpu_csr_if.master        raster_csr_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave     perf_raster_if,
    VX_perf_cache_if.slave      perf_rcache_if,
`endif
`endif
`ifdef EXT_ROP_ENABLE
    VX_gpu_csr_if.master        rop_csr_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave        perf_rop_if,
    VX_perf_cache_if.slave      perf_ocache_if,
`endif
`endif

    VX_commit_csr_if.slave      commit_csr_if,
    VX_sched_csr_if.slave       sched_csr_if,
    VX_csr_req_if.slave         csr_req_if,
    VX_commit_if.master         csr_commit_if,
    
`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave      fpu_to_csr_if,
    input wire                  fpu_pending,
`endif
    input wire                  gpu_pending,

    output wire                 req_pending
);
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    
    reg [`NUM_THREADS-1:0][31:0] csr_read_data;
    reg  [31:0]                 csr_write_data;
    wire [31:0]                 csr_read_data_ro, csr_read_data_rw;
    wire [31:0]                 csr_req_data;
    reg                         csr_rd_enable;
    wire                        csr_wr_enable;

    `UNUSED_VAR (gpu_pending)
    wire csr_access_pending = (0    
    `ifdef EXT_GFX_ENABLE
        || gpu_pending
    `endif
    `ifdef EXT_F_ENABLE
        || fpu_pending
    `endif
        );

    wire csr_req_valid = csr_req_if.valid && ~csr_access_pending;  
    wire csr_req_ready;

    // can accept new request?
    assign csr_req_if.ready = csr_req_ready && ~csr_access_pending;

    wire csr_write_enable = (csr_req_if.op_type == `INST_CSR_RW);

`ifdef EXT_TEX_ENABLE

    wire tex_addr_enable = (csr_req_if.addr >= `CSR_TEX_BEGIN && csr_req_if.addr < `CSR_TEX_END);

    assign tex_csr_if.read_enable = csr_req_valid && ~csr_write_enable && tex_addr_enable;
    assign tex_csr_if.read_uuid   = csr_req_if.uuid;
    assign tex_csr_if.read_wid    = csr_req_if.wid;
    assign tex_csr_if.read_tmask  = csr_req_if.tmask;
    assign tex_csr_if.read_addr   = csr_req_if.addr;
    `UNUSED_VAR (tex_csr_if.read_data)
    
    assign tex_csr_if.write_enable = csr_req_valid && csr_write_enable && tex_addr_enable;
    assign tex_csr_if.write_uuid   = csr_req_if.uuid;
    assign tex_csr_if.write_wid    = csr_req_if.wid;
    assign tex_csr_if.write_tmask  = csr_req_if.tmask;
    assign tex_csr_if.write_addr   = csr_req_if.addr;
    assign tex_csr_if.write_data   = csr_req_if.rs1_data;
`endif

`ifdef EXT_RASTER_ENABLE

    wire raster_addr_enable = (csr_req_if.addr >= `CSR_RASTER_BEGIN && csr_req_if.addr < `CSR_RASTER_END);
    
    assign raster_csr_if.read_enable = csr_req_valid && ~csr_write_enable && raster_addr_enable;
    assign raster_csr_if.read_uuid   = csr_req_if.uuid;
    assign raster_csr_if.read_wid    = csr_req_if.wid;
    assign raster_csr_if.read_tmask  = csr_req_if.tmask;
    assign raster_csr_if.read_addr   = csr_req_if.addr;
    
    assign raster_csr_if.write_enable = csr_req_valid && csr_write_enable && raster_addr_enable;
    assign raster_csr_if.write_uuid   = csr_req_if.uuid;
    assign raster_csr_if.write_wid    = csr_req_if.wid;
    assign raster_csr_if.write_tmask  = csr_req_if.tmask;
    assign raster_csr_if.write_addr   = csr_req_if.addr;
    assign raster_csr_if.write_data   = csr_req_if.rs1_data;
`endif

`ifdef EXT_ROP_ENABLE

    wire rop_addr_enable = (csr_req_if.addr >= `CSR_ROP_BEGIN && csr_req_if.addr < `CSR_ROP_END);

    assign rop_csr_if.read_enable = csr_req_valid && ~csr_write_enable && rop_addr_enable;
    assign rop_csr_if.read_uuid   = csr_req_if.uuid;
    assign rop_csr_if.read_wid    = csr_req_if.wid;
    assign rop_csr_if.read_tmask  = csr_req_if.tmask;
    assign rop_csr_if.read_addr   = csr_req_if.addr;
    `UNUSED_VAR (rop_csr_if.read_data)
    
    assign rop_csr_if.write_enable = csr_req_valid && csr_write_enable && rop_addr_enable; 
    assign rop_csr_if.write_uuid   = csr_req_if.uuid;
    assign rop_csr_if.write_wid    = csr_req_if.wid;
    assign rop_csr_if.write_tmask  = csr_req_if.tmask;
    assign rop_csr_if.write_addr   = csr_req_if.addr;
    assign rop_csr_if.write_data   = csr_req_if.rs1_data;
`endif

    VX_csr_data #(
        .CORE_ID (CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if(perf_pipeline_if),
        .perf_gpu_if    (perf_gpu_if),
    `ifdef EXT_TEX_ENABLE        
        .perf_tex_if    (perf_tex_if),
        .perf_tcache_if (perf_tcache_if),
    `endif    
    `ifdef EXT_RASTER_ENABLE        
        .perf_raster_if (perf_raster_if),
        .perf_rcache_if (perf_rcache_if),
    `endif
    `ifdef EXT_ROP_ENABLE
        .perf_rop_if    (perf_rop_if),
        .perf_ocache_if (perf_ocache_if),
    `endif
    `endif

        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `endif    

        .read_enable    (csr_req_valid && csr_rd_enable),
        .read_uuid      (csr_req_if.uuid),
        .read_wid       (csr_req_if.wid),    
        .read_tmask     (csr_req_if.tmask),    
        .read_addr      (csr_req_if.addr),
        .read_data_ro   (csr_read_data_ro[31:0]),
        .read_data_rw   (csr_read_data_rw[31:0]),

        .write_enable   (csr_req_valid && csr_wr_enable),       
        .write_uuid     (csr_req_if.uuid),
        .write_wid      (csr_req_if.wid),
        .write_addr     (csr_req_if.addr),        
        .write_data     (csr_write_data[31:0])
    );

    // CSR read

    wire [`NUM_THREADS-1:0][31:0] wtid, ltid, gtid;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign wtid[i] = 32'(i);
        assign ltid[i] = (32'(csr_req_if.wid) << `NT_BITS) + i;
        assign gtid[i] = 32'((CORE_ID << (`NW_BITS + `NT_BITS)) + (32'(csr_req_if.wid) << `NT_BITS) + i);
    end  

    always @(*) begin
        csr_rd_enable = 0;
    `ifdef EXT_RASTER_ENABLE
        if (raster_addr_enable) begin
            csr_read_data = raster_csr_if.read_data;
        end else
    `endif
        case (csr_req_if.addr)
        `CSR_WTID : csr_read_data = wtid;
        `CSR_LTID : csr_read_data = ltid;
        `CSR_GTID : csr_read_data = gtid;
        default : begin
            csr_read_data = {`NUM_THREADS{csr_read_data_ro | csr_read_data_rw}};
            csr_rd_enable = 1;
        end
        endcase
    end

    // CSR write

    assign csr_req_data = csr_req_if.use_imm ? 32'(csr_req_if.imm) : csr_req_if.rs1_data[csr_req_if.tid];

    assign csr_wr_enable = (csr_write_enable || (csr_req_data != 0))
                `ifdef EXT_ROP_ENABLE
                    && !rop_addr_enable
                `endif    
                    ;

    always @(*) begin
        case (csr_req_if.op_type)
            `INST_CSR_RW: begin
                csr_write_data = csr_req_data;
            end
            `INST_CSR_RS: begin
                csr_write_data = csr_read_data_rw | csr_req_data;
            end
            //`INST_CSR_RC
            default: begin
                csr_write_data = csr_read_data_rw & ~csr_req_data;
            end
        endcase
    end

    // send response
    wire [`NUM_THREADS-1:0][31:0] csr_commit_data;

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + `NUM_THREADS * 32)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({csr_req_if.uuid,    csr_req_if.wid,    csr_req_if.tmask,    csr_req_if.PC,    csr_req_if.rd,    csr_req_if.wb,    csr_read_data}),
        .data_out  ({csr_commit_if.uuid, csr_commit_if.wid, csr_commit_if.tmask, csr_commit_if.PC, csr_commit_if.rd, csr_commit_if.wb, csr_commit_data}),
        .valid_out (csr_commit_if.valid),
        .ready_out (csr_commit_if.ready)
    );
    
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign csr_commit_if.data[i] = `XLEN'(csr_commit_data[i]);
    end

    assign csr_commit_if.eop = 1'b1;

    // pending request
    reg req_pending_r;
    always @(posedge clk) begin
        if (reset) begin
            req_pending_r <= 0;
        end else begin
            if (csr_req_if.valid && csr_req_if.ready) begin
                 req_pending_r <= 1;
            end
            if (csr_commit_if.valid && csr_commit_if.ready) begin
                 req_pending_r <= 0;
            end
        end
    end
    assign req_pending = req_pending_r;

endmodule
