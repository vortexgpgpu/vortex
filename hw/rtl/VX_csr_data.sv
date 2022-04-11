`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave             perf_memsys_if,
    VX_perf_pipeline_if.slave           perf_pipeline_if,
`endif

    VX_dcr_base_if                      dcr_base_if,

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if.master                tex_csr_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave                tex_perf_if,
`endif
`endif
`ifdef EXT_RASTER_ENABLE
    VX_gpu_csr_if.master                raster_csr_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave             raster_perf_if,
`endif
`endif
`ifdef EXT_ROP_ENABLE
    VX_gpu_csr_if.master                rop_csr_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave                rop_perf_if,
`endif
`endif

    VX_cmt_to_csr_if.slave              cmt_to_csr_if,
    VX_fetch_to_csr_if.slave            fetch_to_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave              fpu_to_csr_if,
`endif

    input wire                          read_enable,
    input wire [`UUID_BITS-1:0]         read_uuid,
    input wire [`NW_BITS-1:0]           read_wid,
    input wire [`NUM_THREADS-1:0]       read_tmask,
    input wire [`CSR_ADDR_BITS-1:0]     read_addr,
    output wire [`NUM_THREADS-1:0][31:0] read_data,

    input wire                          write_enable, 
    input wire [`UUID_BITS-1:0]         write_uuid,
    input wire [`NW_BITS-1:0]           write_wid,
    input wire [`NUM_THREADS-1:0]       write_tmask,
    input wire [`CSR_ADDR_BITS-1:0]     write_addr,
    input wire [`NUM_THREADS-1:0][31:0] write_data
);
    reg [31:0] csr_satp;
    reg [31:0] csr_mstatus;
    reg [31:0] csr_medeleg;
    reg [31:0] csr_mideleg;
    reg [31:0] csr_mie;
    reg [31:0] csr_mtvec;
    reg [31:0] csr_mepc;    
    reg [31:0] csr_pmpcfg [0:0];
    reg [31:0] csr_pmpaddr [0:0];
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FFLAGS_BITS-1:0] fcsr;

    wire [`NT_BITS-1:0] tid;

    VX_lzc #(
        .N (`NUM_THREADS)
    ) tid_select (
        .in_i       (write_tmask),
        .cnt_o      (tid),
        `UNUSED_PIN (valid_o)
    );    

`ifdef EXT_TEX_ENABLE    
    wire tex_read_enable  = (read_addr >= `CSR_TEX_BEGIN && read_addr < `CSR_TEX_END);
    wire tex_write_enable = (write_addr >= `CSR_TEX_BEGIN && write_addr < `CSR_TEX_END);

    assign tex_csr_if.read_enable = read_enable && tex_read_enable;
    assign tex_csr_if.read_uuid   = read_uuid;
    assign tex_csr_if.read_wid    = read_wid;
    assign tex_csr_if.read_tmask  = read_tmask;
    assign tex_csr_if.read_addr   = read_addr;
    
    assign tex_csr_if.write_enable = write_enable && tex_write_enable; 
    assign tex_csr_if.write_uuid   = write_uuid;
    assign tex_csr_if.write_wid    = write_wid;
    assign tex_csr_if.write_tmask  = write_tmask;
    assign tex_csr_if.write_addr   = write_addr;
    assign tex_csr_if.write_data   = write_data;
`endif

`ifdef EXT_RASTER_ENABLE
    wire raster_read_enable  = (read_addr >= `CSR_RASTER_BEGIN && read_addr < `CSR_RASTER_END);
    wire raster_write_enable = (write_addr >= `CSR_RASTER_BEGIN && write_addr < `CSR_RASTER_END);

    assign raster_csr_if.read_enable = read_enable && raster_read_enable;
    assign raster_csr_if.read_uuid   = read_uuid;
    assign raster_csr_if.read_wid    = read_wid;
    assign raster_csr_if.read_tmask  = read_tmask;
    assign raster_csr_if.read_addr   = read_addr;
    
    assign raster_csr_if.write_enable = write_enable && raster_write_enable; 
    assign raster_csr_if.write_uuid   = write_uuid;
    assign raster_csr_if.write_wid    = write_wid;
    assign raster_csr_if.write_tmask  = write_tmask;
    assign raster_csr_if.write_addr   = write_addr;
    assign raster_csr_if.write_data   = write_data;
`endif

`ifdef EXT_ROP_ENABLE
    wire rop_read_enable  = (read_addr >= `CSR_ROP_BEGIN && read_addr < `CSR_ROP_END);
    wire rop_write_enable = (write_addr >= `CSR_ROP_BEGIN && write_addr < `CSR_ROP_END);;

    assign rop_csr_if.read_enable = read_enable && rop_read_enable;
    assign rop_csr_if.read_uuid   = read_uuid;
    assign rop_csr_if.read_wid    = read_wid;
    assign rop_csr_if.read_tmask  = read_tmask;
    assign rop_csr_if.read_addr   = read_addr;
    
    assign rop_csr_if.write_enable = write_enable && rop_write_enable; 
    assign rop_csr_if.write_uuid   = write_uuid;
    assign rop_csr_if.write_wid    = write_wid;
    assign rop_csr_if.write_tmask  = write_tmask;
    assign rop_csr_if.write_addr   = write_addr;
    assign rop_csr_if.write_data   = write_data;
`endif

    always @(posedge clk) begin
        reg write_addr_valid_r;
        if (reset) begin
            fcsr <= '0;
        end else begin
        `ifdef EXT_F_ENABLE
            if (fpu_to_csr_if.write_enable) begin
                fcsr[fpu_to_csr_if.write_wid][`FFLAGS_BITS-1:0] <= fcsr[fpu_to_csr_if.write_wid][`FFLAGS_BITS-1:0]
                                                                 | fpu_to_csr_if.write_fflags;
            end
        `endif
            if (write_enable) begin
                case (write_addr)
                    `CSR_FFLAGS:   fcsr[write_wid][`FFLAGS_BITS-1:0] <= write_data[tid][`FFLAGS_BITS-1:0];
                    `CSR_FRM:      fcsr[write_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS] <= write_data[tid][`INST_FRM_BITS-1:0];
                    `CSR_FCSR:     fcsr[write_wid]  <= write_data[tid][`FFLAGS_BITS+`INST_FRM_BITS-1:0];
                    `CSR_SATP:     csr_satp         <= write_data[tid];
                    `CSR_MSTATUS:  csr_mstatus      <= write_data[tid];
                    `CSR_MEDELEG:  csr_medeleg      <= write_data[tid];
                    `CSR_MIDELEG:  csr_mideleg      <= write_data[tid];
                    `CSR_MIE:      csr_mie          <= write_data[tid];
                    `CSR_MTVEC:    csr_mtvec        <= write_data[tid];
                    `CSR_MEPC:     csr_mepc         <= write_data[tid];
                    `CSR_PMPCFG0:  csr_pmpcfg[0]    <= write_data[tid];
                    `CSR_PMPADDR0: csr_pmpaddr[0]   <= write_data[tid];
                    default: begin
                        write_addr_valid_r = 0;
                        `ifdef EXT_TEX_ENABLE
                            if (tex_write_enable) begin
                                write_addr_valid_r = 1;
                            end
                        `endif
                        `ifdef EXT_RASTER_ENABLE
                            if (raster_write_enable) begin
                                write_addr_valid_r = 1;
                            end
                        `endif
                        `ifdef EXT_ROP_ENABLE
                            if (rop_write_enable) begin
                                write_addr_valid_r = 1;
                            end
                        `endif
                            `ASSERT(write_addr_valid_r, ("%t: *** invalid CSR write address: %0h (#%0d)", $time, write_addr, write_uuid));
                        end                
                endcase
            end
        end
    end

    `UNUSED_VAR (write_tmask)
    `UNUSED_VAR (write_data)

///////////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] wtid, ltid, gtid;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign wtid[i] = 32'(i);
        assign ltid[i] = (32'(read_wid) << `NT_BITS) + i;
        assign gtid[i] = 32'((CORE_ID << (`NW_BITS + `NT_BITS)) + (32'(read_wid) << `NT_BITS) + i);
    end  

    reg [`NUM_THREADS-1:0][31:0] read_data_r;
    reg read_addr_valid_r;

    always @(*) begin
        read_data_r = 'x;
        read_addr_valid_r = 1;
        case (read_addr)
            `CSR_FFLAGS     : read_data_r = {`NUM_THREADS{32'(fcsr[read_wid][`FFLAGS_BITS-1:0])}};
            `CSR_FRM        : read_data_r = {`NUM_THREADS{32'(fcsr[read_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS])}};
            `CSR_FCSR       : read_data_r = {`NUM_THREADS{32'(fcsr[read_wid])}};

            `CSR_WTID       : read_data_r = wtid;
            `CSR_LTID       : read_data_r = ltid;
            `CSR_GTID       : read_data_r = gtid;            
            `CSR_LWID       : read_data_r = {`NUM_THREADS{32'(read_wid)}};
            /*`CSR_MHARTID ,*/
            `CSR_GWID       : read_data_r = {`NUM_THREADS{(CORE_ID << `NW_BITS) + 32'(read_wid)}};
            `CSR_GCID       : read_data_r = {`NUM_THREADS{CORE_ID}};

            `CSR_TMASK      : read_data_r = {`NUM_THREADS{32'(read_tmask)}};

            `CSR_NT         : read_data_r = {`NUM_THREADS{32'd`NUM_THREADS}};
            `CSR_NW         : read_data_r = {`NUM_THREADS{32'd`NUM_WARPS}};
            `CSR_NC         : read_data_r = {`NUM_THREADS{32'(`NUM_CORES * `NUM_CLUSTERS)}};
            
            `CSR_MCYCLE     : read_data_r = {`NUM_THREADS{fetch_to_csr_if.cycles[31:0]}};
            `CSR_MCYCLE_H   : read_data_r = {`NUM_THREADS{32'(fetch_to_csr_if.cycles[`PERF_CTR_BITS-1:32])}};
            `CSR_MPM_RESERVED : read_data_r = 'x;
            `CSR_MPM_RESERVED_H : read_data_r = 'x;  
            `CSR_MINSTRET   : read_data_r = {`NUM_THREADS{cmt_to_csr_if.instret[31:0]}};
            `CSR_MINSTRET_H : read_data_r = {`NUM_THREADS{32'(cmt_to_csr_if.instret[`PERF_CTR_BITS-1:32])}};       
            
            `CSR_SATP       : read_data_r = {`NUM_THREADS{32'(csr_satp)}};
            
            `CSR_MSTATUS    : read_data_r = {`NUM_THREADS{32'(csr_mstatus)}};
            `CSR_MISA       : read_data_r = {`NUM_THREADS{((($clog2(`XLEN)-4) << (`XLEN-2)) | `MISA_STD)}};
            `CSR_MEDELEG    : read_data_r = {`NUM_THREADS{32'(csr_medeleg)}};
            `CSR_MIDELEG    : read_data_r = {`NUM_THREADS{32'(csr_mideleg)}};
            `CSR_MIE        : read_data_r = {`NUM_THREADS{32'(csr_mie)}};
            `CSR_MTVEC      : read_data_r = {`NUM_THREADS{32'(csr_mtvec)}};

            `CSR_MEPC       : read_data_r = {`NUM_THREADS{32'(csr_mepc)}};

            `CSR_PMPCFG0    : read_data_r = {`NUM_THREADS{32'(csr_pmpcfg[0])}};
            `CSR_PMPADDR0   : read_data_r = {`NUM_THREADS{32'(csr_pmpaddr[0])}};
            
            `CSR_MVENDORID  : read_data_r = {`NUM_THREADS{32'd`VENDOR_ID}};
            `CSR_MARCHID    : read_data_r = {`NUM_THREADS{32'd`ARCHITECTURE_ID}};
            `CSR_MIMPID     : read_data_r = {`NUM_THREADS{32'd`IMPLEMENTATION_ID}};

            default: begin
                read_addr_valid_r = 0;
                if ((read_addr >= `CSR_MPM_BASE   && read_addr < (`CSR_MPM_BASE + 32))
                 || (read_addr >= `CSR_MPM_BASE_H && read_addr < (`CSR_MPM_BASE_H + 32))) begin
                    read_addr_valid_r = 1;
                `ifdef PERF_ENABLE 
                    case (dcr_base_if.data.mpm_class)
                    `DCR_MPM_CLASS_CORE: begin
                        case (read_addr)                        
                        // PERF: pipeline
                        `CSR_MPM_IBUF_ST    : read_data_r = {`NUM_THREADS{perf_pipeline_if.ibf_stalls[31:0]}};
                        `CSR_MPM_IBUF_ST_H  : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.ibf_stalls[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_SCRB_ST    : read_data_r = {`NUM_THREADS{perf_pipeline_if.scb_stalls[31:0]}};
                        `CSR_MPM_SCRB_ST_H  : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.scb_stalls[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_ALU_ST     : read_data_r = {`NUM_THREADS{perf_pipeline_if.alu_stalls[31:0]}};
                        `CSR_MPM_ALU_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.alu_stalls[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_LSU_ST     : read_data_r = {`NUM_THREADS{perf_pipeline_if.lsu_stalls[31:0]}};
                        `CSR_MPM_LSU_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.lsu_stalls[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_CSR_ST     : read_data_r = {`NUM_THREADS{perf_pipeline_if.csr_stalls[31:0]}};
                        `CSR_MPM_CSR_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.csr_stalls[`PERF_CTR_BITS-1:32])}};
                    `ifdef EXT_F_ENABLE    
                        `CSR_MPM_FPU_ST     : read_data_r = {`NUM_THREADS{perf_pipeline_if.fpu_stalls[31:0]}};
                        `CSR_MPM_FPU_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.fpu_stalls[`PERF_CTR_BITS-1:32])}};
                    `else        
                        `CSR_MPM_FPU_ST     : read_data_r = '0;
                        `CSR_MPM_FPU_ST_H   : read_data_r = '0;
                    `endif
                        `CSR_MPM_GPU_ST     : read_data_r = {`NUM_THREADS{perf_pipeline_if.gpu_stalls[31:0]}};
                        `CSR_MPM_GPU_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.gpu_stalls[`PERF_CTR_BITS-1:32])}};
                        // PERF: decode
                        `CSR_MPM_LOADS      : read_data_r = {`NUM_THREADS{perf_pipeline_if.loads[31:0]}};
                        `CSR_MPM_LOADS_H    : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.loads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_STORES     : read_data_r = {`NUM_THREADS{perf_pipeline_if.stores[31:0]}};
                        `CSR_MPM_STORES_H   : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.stores[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_BRANCHES   : read_data_r = {`NUM_THREADS{perf_pipeline_if.branches[31:0]}};
                        `CSR_MPM_BRANCHES_H : read_data_r = {`NUM_THREADS{32'(perf_pipeline_if.branches[`PERF_CTR_BITS-1:32])}};
                        // PERF: icache
                        `CSR_MPM_ICACHE_READS       : read_data_r = {`NUM_THREADS{perf_memsys_if.icache_reads[31:0]}};
                        `CSR_MPM_ICACHE_READS_H     : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.icache_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_ICACHE_MISS_R      : read_data_r = {`NUM_THREADS{perf_memsys_if.icache_read_misses[31:0]}};
                        `CSR_MPM_ICACHE_MISS_R_H    : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.icache_read_misses[`PERF_CTR_BITS-1:32])}};
                        // PERF: dcache
                        `CSR_MPM_DCACHE_READS       : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_reads[31:0]}};
                        `CSR_MPM_DCACHE_READS_H     : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_DCACHE_WRITES      : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_writes[31:0]}};
                        `CSR_MPM_DCACHE_WRITES_H    : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_writes[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_DCACHE_MISS_R      : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_read_misses[31:0]}};
                        `CSR_MPM_DCACHE_MISS_R_H    : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_read_misses[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_DCACHE_MISS_W      : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_write_misses[31:0]}};
                        `CSR_MPM_DCACHE_MISS_W_H    : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_write_misses[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_DCACHE_BANK_ST     : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_bank_stalls[31:0]}};
                        `CSR_MPM_DCACHE_BANK_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_bank_stalls[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_DCACHE_MSHR_ST     : read_data_r = {`NUM_THREADS{perf_memsys_if.dcache_mshr_stalls[31:0]}};
                        `CSR_MPM_DCACHE_MSHR_ST_H   : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.dcache_mshr_stalls[`PERF_CTR_BITS-1:32])}};
                        // PERF: smem          
                        `CSR_MPM_SMEM_READS     : read_data_r = {`NUM_THREADS{perf_memsys_if.smem_reads[31:0]}};
                        `CSR_MPM_SMEM_READS_H   : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.smem_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_SMEM_WRITES    : read_data_r = {`NUM_THREADS{perf_memsys_if.smem_writes[31:0]}};
                        `CSR_MPM_SMEM_WRITES_H  : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.smem_writes[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_SMEM_BANK_ST   : read_data_r = {`NUM_THREADS{perf_memsys_if.smem_bank_stalls[31:0]}};
                        `CSR_MPM_SMEM_BANK_ST_H : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.smem_bank_stalls[`PERF_CTR_BITS-1:32])}};
                        // PERF: memory
                        `CSR_MPM_MEM_READS      : read_data_r = {`NUM_THREADS{perf_memsys_if.mem_reads[31:0]}};
                        `CSR_MPM_MEM_READS_H    : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.mem_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_MEM_WRITES     : read_data_r = {`NUM_THREADS{perf_memsys_if.mem_writes[31:0]}};
                        `CSR_MPM_MEM_WRITES_H   : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.mem_writes[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_MEM_LAT        : read_data_r = {`NUM_THREADS{perf_memsys_if.mem_latency[31:0]}};
                        `CSR_MPM_MEM_LAT_H      : read_data_r = {`NUM_THREADS{32'(perf_memsys_if.mem_latency[`PERF_CTR_BITS-1:32])}};                        
                        default:;            
                        endcase                                                     
                    end
                    `DCR_MPM_CLASS_TEX: begin
                    `ifdef EXT_TEX_ENABLE
                        case (read_addr)
                        `CSR_MPM_TEX_READS      : read_data_r = {`NUM_THREADS{tex_perf_if.mem_reads[31:0]}};
                        `CSR_MPM_TEX_READS_H    : read_data_r = {`NUM_THREADS{32'(tex_perf_if.mem_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_TEX_LAT        : read_data_r = {`NUM_THREADS{tex_perf_if.mem_latency[31:0]}};
                        `CSR_MPM_TEX_LAT_H      : read_data_r = {`NUM_THREADS{32'(tex_perf_if.mem_latency[`PERF_CTR_BITS-1:32])}};
                        default:;
                        endcase
                    `endif
                    end
                    `DCR_MPM_CLASS_RASTER: begin
                    `ifdef EXT_RASTER_ENABLE
                        case (read_addr)
                        `CSR_MPM_RAS_READS      : read_data_r = {`NUM_THREADS{raster_perf_if.mem_reads[31:0]}};
                        `CSR_MPM_RAS_READS_H    : read_data_r = {`NUM_THREADS{32'(raster_perf_if.mem_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_RAS_LAT        : read_data_r = {`NUM_THREADS{raster_perf_if.mem_latency[31:0]}};
                        `CSR_MPM_RAS_LAT_H      : read_data_r = {`NUM_THREADS{32'(raster_perf_if.mem_latency[`PERF_CTR_BITS-1:32])}};
                        default:;
                        endcase
                    `endif
                    end
                    `DCR_MPM_CLASS_ROP: begin
                    `ifdef EXT_ROP_ENABLE
                        case (read_addr)
                        `CSR_MPM_ROP_READS      : read_data_r = {`NUM_THREADS{rop_perf_if.mem_reads[31:0]}};
                        `CSR_MPM_ROP_READS_H    : read_data_r = {`NUM_THREADS{32'(rop_perf_if.mem_reads[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_ROP_WRITES     : read_data_r = {`NUM_THREADS{rop_perf_if.mem_writes[31:0]}};
                        `CSR_MPM_ROP_WRITES_H   : read_data_r = {`NUM_THREADS{32'(rop_perf_if.mem_writes[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_ROP_LAT        : read_data_r = {`NUM_THREADS{rop_perf_if.mem_latency[31:0]}};
                        `CSR_MPM_ROP_LAT_H      : read_data_r = {`NUM_THREADS{32'(rop_perf_if.mem_latency[`PERF_CTR_BITS-1:32])}};
                        `CSR_MPM_ROP_IDLE_CYC   : read_data_r = {`NUM_THREADS{rop_perf_if.idle_cycles[31:0]}};
                        `CSR_MPM_ROP_IDLE_CYC_H : read_data_r = {`NUM_THREADS{32'(rop_perf_if.idle_cycles[`PERF_CTR_BITS-1:32])}};
                        default:;
                        endcase
                    `endif
                    end
                    default:;
                    endcase
                `endif
                end     
            `ifdef EXT_TEX_ENABLE    
                else if (tex_read_enable) begin
                    read_data_r = tex_csr_if.read_data;
                    read_addr_valid_r = 1;                    
                end
            `endif
            `ifdef EXT_RASTER_ENABLE
                else if (raster_read_enable) begin
                    read_data_r = raster_csr_if.read_data;
                    read_addr_valid_r = 1;                    
                end
            `endif
            `ifdef EXT_ROP_ENABLE
                else if (rop_read_enable) begin
                    read_data_r = rop_csr_if.read_data;
                    read_addr_valid_r = 1;                    
                end
            `endif
            end
        endcase
    end

    `UNUSED_VAR (dcr_base_if.data)

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_r, ("%t: *** invalid CSR read address: 0x%0h (#%0d)", $time, read_addr, read_uuid))

    assign read_data = read_data_r;

`ifdef EXT_F_ENABLE    
    assign fpu_to_csr_if.read_frm = fcsr[fpu_to_csr_if.read_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS];
`endif

endmodule
