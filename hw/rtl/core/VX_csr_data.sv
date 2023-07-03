`include "VX_define.vh"
`include "VX_gpu_types.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave                mem_perf_if,
    VX_pipeline_perf_if.slave           pipeline_perf_if,
    VX_gpu_perf_if.slave                gpu_perf_if,
`ifdef EXT_TEX_ENABLE
    VX_tex_perf_if.slave                perf_tex_if,
    VX_cache_perf_if.slave              perf_tcache_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_perf_if.slave             perf_raster_if,
    VX_cache_perf_if.slave              perf_rcache_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_perf_if.slave                perf_rop_if,
    VX_cache_perf_if.slave              perf_ocache_if,
`endif
`endif

    VX_commit_csr_if.slave              commit_csr_if,
    VX_sched_csr_if.slave               sched_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave              fpu_to_csr_if,
`endif

    input wire                          read_enable,
    input wire [`UP(`UUID_BITS)-1:0]    read_uuid,
    input wire [`UP(`NW_BITS)-1:0]      read_wid,
    input wire [`NUM_THREADS-1:0]       read_tmask,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [31:0]                  read_data_ro,
    output wire [31:0]                  read_data_rw,

    input wire                          write_enable, 
    input wire [`UP(`UUID_BITS)-1:0]    write_uuid,
    input wire [`UP(`NW_BITS)-1:0]      write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [31:0]                   write_data
);

    `UNUSED_VAR (reset)
    `UNUSED_VAR (write_wid)
    `UNUSED_VAR (write_data)

    // CSRs Write /////////////////////////////////////////////////////////////

`ifdef EXT_F_ENABLE    
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr;
`endif

    always @(posedge clk) begin
    `ifdef EXT_F_ENABLE
        if (reset) begin
            fcsr <= '0;
        end else begin
            if (fpu_to_csr_if.write_enable) begin
                fcsr[fpu_to_csr_if.write_wid][`FP_FLAGS_BITS-1:0] <= fcsr[fpu_to_csr_if.write_wid][`FP_FLAGS_BITS-1:0]
                                                                   | fpu_to_csr_if.write_fflags;
            end
        end
    `endif
        if (write_enable) begin
            case (write_addr)
            `ifdef EXT_F_ENABLE
                `VX_CSR_FFLAGS:   fcsr[write_wid][`FP_FLAGS_BITS-1:0] <= write_data[`FP_FLAGS_BITS-1:0];
                `VX_CSR_FRM:      fcsr[write_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] <= write_data[`INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:     fcsr[write_wid] <= write_data[`FP_FLAGS_BITS+`INST_FRM_BITS-1:0];
            `endif
                `VX_CSR_SATP,
                `VX_CSR_MSTATUS,
                `VX_CSR_MNSTATUS,
                `VX_CSR_MEDELEG,
                `VX_CSR_MIDELEG,
                `VX_CSR_MIE,
                `VX_CSR_MTVEC,
                `VX_CSR_MEPC,
                `VX_CSR_PMPCFG0,
                `VX_CSR_PMPADDR0: /* do nothing!*/;
                default: begin
                    `ASSERT(0, ("%t: *** invalid CSR write address: %0h (#%0d)", $time, write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    reg [31:0] read_data_ro_r;
    reg [31:0] read_data_rw_r;
    reg read_addr_valid_r;

    always @(*) begin
        read_data_ro_r    = '0;
        read_data_rw_r    = '0;
        read_addr_valid_r = 1;
        case (read_addr)            
            `VX_CSR_MVENDORID  : read_data_ro_r = 32'(`VENDOR_ID);
            `VX_CSR_MARCHID    : read_data_ro_r = 32'(`ARCHITECTURE_ID);
            `VX_CSR_MIMPID     : read_data_ro_r = 32'(`IMPLEMENTATION_ID);
            `VX_CSR_MISA       : read_data_ro_r = ((($clog2(`XLEN)-4) << (`XLEN-2)) | `MISA_STD);

        `ifdef EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_r = 32'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_r = 32'(fcsr[read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_r = 32'(fcsr[read_wid]);
        `endif
            `VX_CSR_WARP_ID    : read_data_ro_r = 32'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_r = 32'(CORE_ID % `NUM_CORES);
            `VX_CSR_CLUSTER_ID : read_data_ro_r = 32'(CORE_ID / `NUM_CORES);
            `VX_CSR_TMASK      : read_data_ro_r = 32'(read_tmask);
            `VX_CSR_NUM_THREADS: read_data_ro_r = 32'(`NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_r = 32'(`NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_r = 32'(`NUM_CORES);
            `VX_CSR_NUM_CLUSTERS:read_data_ro_r = 32'(`NUM_CLUSTERS);            
            `VX_CSR_MCYCLE     : read_data_ro_r = 32'(sched_csr_if.cycles[31:0]);
            `VX_CSR_MCYCLE_H   : read_data_ro_r = 32'(sched_csr_if.cycles[`PERF_CTR_BITS-1:32]);
            `VX_CSR_MPM_RESERVED : read_data_ro_r = 'x;
            `VX_CSR_MPM_RESERVED_H : read_data_ro_r = 'x;  
            `VX_CSR_MINSTRET   : read_data_ro_r = 32'(commit_csr_if.instret[31:0]);
            `VX_CSR_MINSTRET_H : read_data_ro_r = 32'(commit_csr_if.instret[`PERF_CTR_BITS-1:32]);       
            
            `VX_CSR_SATP,
            `VX_CSR_MSTATUS,
            `VX_CSR_MNSTATUS,
            `VX_CSR_MEDELEG,
            `VX_CSR_MIDELEG,
            `VX_CSR_MIE,
            `VX_CSR_MTVEC,
            `VX_CSR_MEPC,
            `VX_CSR_PMPCFG0,
            `VX_CSR_PMPADDR0   : read_data_ro_r = 32'(0);

            default: begin
                read_addr_valid_r = 0;
                if ((read_addr >= `VX_CSR_MPM_BASE   && read_addr < (`VX_CSR_MPM_BASE + 32))
                 || (read_addr >= `VX_CSR_MPM_BASE_H && read_addr < (`VX_CSR_MPM_BASE_H + 32))) begin
                    read_addr_valid_r = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `VX_CSR_MPM_IBUF_ST        : read_data_ro_r = pipeline_perf_if.ibf_stalls[31:0];
                        `VX_CSR_MPM_IBUF_ST_H      : read_data_ro_r = 32'(pipeline_perf_if.ibf_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_ST        : read_data_ro_r = pipeline_perf_if.scb_stalls[31:0];
                        `VX_CSR_MPM_SCRB_ST_H      : read_data_ro_r = 32'(pipeline_perf_if.scb_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ALU_ST         : read_data_ro_r = pipeline_perf_if.alu_stalls[31:0];
                        `VX_CSR_MPM_ALU_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.alu_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_LSU_ST         : read_data_ro_r = pipeline_perf_if.lsu_stalls[31:0];
                        `VX_CSR_MPM_LSU_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.lsu_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_CSR_ST         : read_data_ro_r = pipeline_perf_if.csr_stalls[31:0];
                        `VX_CSR_MPM_CSR_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.csr_stalls[`PERF_CTR_BITS-1:32]);
                    `ifdef EXT_F_ENABLE
                        `VX_CSR_MPM_FPU_ST         : read_data_ro_r = pipeline_perf_if.fpu_stalls[31:0];
                        `VX_CSR_MPM_FPU_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.fpu_stalls[`PERF_CTR_BITS-1:32]);
                    `else
                        `VX_CSR_MPM_FPU_ST         : read_data_ro_r = '0;
                        `VX_CSR_MPM_FPU_ST_H       : read_data_ro_r = '0;
                    `endif
                        `VX_CSR_MPM_GPU_ST         : read_data_ro_r = pipeline_perf_if.gpu_stalls[31:0];
                        `VX_CSR_MPM_GPU_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.gpu_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: memory
                        `VX_CSR_MPM_IFETCHES       : read_data_ro_r = pipeline_perf_if.ifetches[31:0];
                        `VX_CSR_MPM_IFETCHES_H     : read_data_ro_r = 32'(pipeline_perf_if.ifetches[`PERF_CTR_BITS-1:32]); 
                        `VX_CSR_MPM_LOADS          : read_data_ro_r = pipeline_perf_if.loads[31:0];
                        `VX_CSR_MPM_LOADS_H        : read_data_ro_r = 32'(pipeline_perf_if.loads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_STORES         : read_data_ro_r = pipeline_perf_if.stores[31:0];
                        `VX_CSR_MPM_STORES_H       : read_data_ro_r = 32'(pipeline_perf_if.stores[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_IFETCH_LAT     : read_data_ro_r = pipeline_perf_if.ifetch_latency[31:0];
                        `VX_CSR_MPM_IFETCH_LAT_H   : read_data_ro_r = 32'(pipeline_perf_if.ifetch_latency[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_LOAD_LAT       : read_data_ro_r = pipeline_perf_if.load_latency[31:0];
                        `VX_CSR_MPM_LOAD_LAT_H     : read_data_ro_r = 32'(pipeline_perf_if.load_latency[`PERF_CTR_BITS-1:32]);                
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `VX_CSR_MPM_ICACHE_READS       : read_data_ro_r = mem_perf_if.icache_reads[31:0];
                        `VX_CSR_MPM_ICACHE_READS_H     : read_data_ro_r = 32'(mem_perf_if.icache_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ICACHE_MISS_R      : read_data_ro_r = mem_perf_if.icache_read_misses[31:0];
                        `VX_CSR_MPM_ICACHE_MISS_R_H    : read_data_ro_r = 32'(mem_perf_if.icache_read_misses[`PERF_CTR_BITS-1:32]);
                        // PERF: dcache
                        `VX_CSR_MPM_DCACHE_READS       : read_data_ro_r = mem_perf_if.dcache_reads[31:0];
                        `VX_CSR_MPM_DCACHE_READS_H     : read_data_ro_r = 32'(mem_perf_if.dcache_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_WRITES      : read_data_ro_r = mem_perf_if.dcache_writes[31:0];
                        `VX_CSR_MPM_DCACHE_WRITES_H    : read_data_ro_r = 32'(mem_perf_if.dcache_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MISS_R      : read_data_ro_r = mem_perf_if.dcache_read_misses[31:0];
                        `VX_CSR_MPM_DCACHE_MISS_R_H    : read_data_ro_r = 32'(mem_perf_if.dcache_read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MISS_W      : read_data_ro_r = mem_perf_if.dcache_write_misses[31:0];
                        `VX_CSR_MPM_DCACHE_MISS_W_H    : read_data_ro_r = 32'(mem_perf_if.dcache_write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_BANK_ST     : read_data_ro_r = mem_perf_if.dcache_bank_stalls[31:0];
                        `VX_CSR_MPM_DCACHE_BANK_ST_H   : read_data_ro_r = 32'(mem_perf_if.dcache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MSHR_ST     : read_data_ro_r = mem_perf_if.dcache_mshr_stalls[31:0];
                        `VX_CSR_MPM_DCACHE_MSHR_ST_H   : read_data_ro_r = 32'(mem_perf_if.dcache_mshr_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: smem          
                        `VX_CSR_MPM_SMEM_READS         : read_data_ro_r = mem_perf_if.smem_reads[31:0];
                        `VX_CSR_MPM_SMEM_READS_H       : read_data_ro_r = 32'(mem_perf_if.smem_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SMEM_WRITES        : read_data_ro_r = mem_perf_if.smem_writes[31:0];
                        `VX_CSR_MPM_SMEM_WRITES_H      : read_data_ro_r = 32'(mem_perf_if.smem_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SMEM_BANK_ST       : read_data_ro_r = mem_perf_if.smem_bank_stalls[31:0];
                        `VX_CSR_MPM_SMEM_BANK_ST_H     : read_data_ro_r = 32'(mem_perf_if.smem_bank_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: l2cache                        
                        `VX_CSR_MPM_L2CACHE_READS      : read_data_ro_r = mem_perf_if.l2cache_reads[31:0];
                        `VX_CSR_MPM_L2CACHE_READS_H    : read_data_ro_r = 32'(mem_perf_if.l2cache_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_WRITES     : read_data_ro_r = mem_perf_if.l2cache_writes[31:0];
                        `VX_CSR_MPM_L2CACHE_WRITES_H   : read_data_ro_r = 32'(mem_perf_if.l2cache_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MISS_R     : read_data_ro_r = mem_perf_if.l2cache_read_misses[31:0];
                        `VX_CSR_MPM_L2CACHE_MISS_R_H   : read_data_ro_r = 32'(mem_perf_if.l2cache_read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MISS_W     : read_data_ro_r = mem_perf_if.l2cache_write_misses[31:0];
                        `VX_CSR_MPM_L2CACHE_MISS_W_H   : read_data_ro_r = 32'(mem_perf_if.l2cache_write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_BANK_ST    : read_data_ro_r = mem_perf_if.l2cache_bank_stalls[31:0];
                        `VX_CSR_MPM_L2CACHE_BANK_ST_H  : read_data_ro_r = 32'(mem_perf_if.l2cache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MSHR_ST    : read_data_ro_r = mem_perf_if.l2cache_mshr_stalls[31:0];
                        `VX_CSR_MPM_L2CACHE_MSHR_ST_H  : read_data_ro_r = 32'(mem_perf_if.l2cache_mshr_stalls[`PERF_CTR_BITS-1:32]);      
                        // PERF: l3cache
                        `VX_CSR_MPM_L3CACHE_READS      : read_data_ro_r = mem_perf_if.l3cache_reads[31:0];
                        `VX_CSR_MPM_L3CACHE_READS_H    : read_data_ro_r = 32'(mem_perf_if.l3cache_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_WRITES     : read_data_ro_r = mem_perf_if.l3cache_writes[31:0];
                        `VX_CSR_MPM_L3CACHE_WRITES_H   : read_data_ro_r = 32'(mem_perf_if.l3cache_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MISS_R     : read_data_ro_r = mem_perf_if.l3cache_read_misses[31:0];
                        `VX_CSR_MPM_L3CACHE_MISS_R_H   : read_data_ro_r = 32'(mem_perf_if.l3cache_read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MISS_W     : read_data_ro_r = mem_perf_if.l3cache_write_misses[31:0];
                        `VX_CSR_MPM_L3CACHE_MISS_W_H   : read_data_ro_r = 32'(mem_perf_if.l3cache_write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_BANK_ST    : read_data_ro_r = mem_perf_if.l3cache_bank_stalls[31:0];
                        `VX_CSR_MPM_L3CACHE_BANK_ST_H  : read_data_ro_r = 32'(mem_perf_if.l3cache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MSHR_ST    : read_data_ro_r = mem_perf_if.l3cache_mshr_stalls[31:0];
                        `VX_CSR_MPM_L3CACHE_MSHR_ST_H  : read_data_ro_r = 32'(mem_perf_if.l3cache_mshr_stalls[`PERF_CTR_BITS-1:32]); 
                        // PERF: memory
                        `VX_CSR_MPM_MEM_READS          : read_data_ro_r = mem_perf_if.mem_reads[31:0];
                        `VX_CSR_MPM_MEM_READS_H        : read_data_ro_r = 32'(mem_perf_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_MEM_WRITES         : read_data_ro_r = mem_perf_if.mem_writes[31:0];
                        `VX_CSR_MPM_MEM_WRITES_H       : read_data_ro_r = 32'(mem_perf_if.mem_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_MEM_LAT            : read_data_ro_r = mem_perf_if.mem_latency[31:0];
                        `VX_CSR_MPM_MEM_LAT_H          : read_data_ro_r = 32'(mem_perf_if.mem_latency[`PERF_CTR_BITS-1:32]);     
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_TEX: begin
                    `ifdef EXT_TEX_ENABLE
                        case (read_addr)
                        `VX_CSR_MPM_TEX_READS      : read_data_ro_r = perf_tex_if.mem_reads[31:0];
                        `VX_CSR_MPM_TEX_READS_H    : read_data_ro_r = 32'(perf_tex_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_TEX_LAT        : read_data_ro_r = perf_tex_if.mem_latency[31:0];
                        `VX_CSR_MPM_TEX_LAT_H      : read_data_ro_r = 32'(perf_tex_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_TEX_STALL      : read_data_ro_r = perf_tex_if.stall_cycles[31:0];
                        `VX_CSR_MPM_TEX_STALL_H    : read_data_ro_r = 32'(perf_tex_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef TCACHE_ENABLE
                        // cache perf counters
                        `VX_CSR_MPM_TCACHE_READS   : read_data_ro_r = perf_tcache_if.reads[31:0];
                        `VX_CSR_MPM_TCACHE_READS_H : read_data_ro_r = 32'(perf_tcache_if.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_TCACHE_MISS_R  : read_data_ro_r = perf_tcache_if.read_misses[31:0];
                        `VX_CSR_MPM_TCACHE_MISS_R_H: read_data_ro_r = 32'(perf_tcache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_TCACHE_BANK_ST : read_data_ro_r = perf_tcache_if.bank_stalls[31:0];
                        `VX_CSR_MPM_TCACHE_BANK_ST_H:read_data_ro_r = 32'(perf_tcache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_TCACHE_MSHR_ST  :read_data_ro_r = perf_tcache_if.mshr_stalls[31:0];
                        `VX_CSR_MPM_TCACHE_MSHR_ST_H:read_data_ro_r = 32'(perf_tcache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `VX_CSR_MPM_TEX_ISSUE_ST   : read_data_ro_r = gpu_perf_if.tex_stalls[31:0];
                        `VX_CSR_MPM_TEX_ISSUE_ST_H : read_data_ro_r = 32'(gpu_perf_if.tex_stalls[`PERF_CTR_BITS-1:32]);
                        default:;
                        endcase
                    `endif
                    end
                    `VX_DCR_MPM_CLASS_RASTER: begin
                    `ifdef EXT_RASTER_ENABLE
                        case (read_addr)
                        `VX_CSR_MPM_RASTER_READS   : read_data_ro_r = perf_raster_if.mem_reads[31:0];
                        `VX_CSR_MPM_RASTER_READS_H : read_data_ro_r = 32'(perf_raster_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_RASTER_LAT     : read_data_ro_r = perf_raster_if.mem_latency[31:0];
                        `VX_CSR_MPM_RASTER_LAT_H   : read_data_ro_r = 32'(perf_raster_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_RASTER_STALL   : read_data_ro_r = perf_raster_if.stall_cycles[31:0];
                        `VX_CSR_MPM_RASTER_STALL_H : read_data_ro_r = 32'(perf_raster_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef RCACHE_ENABLE
                        // cache perf counters
                        `VX_CSR_MPM_RCACHE_READS   : read_data_ro_r = perf_rcache_if.reads[31:0];
                        `VX_CSR_MPM_RCACHE_READS_H : read_data_ro_r = 32'(perf_rcache_if.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_RCACHE_MISS_R  : read_data_ro_r = perf_rcache_if.read_misses[31:0];
                        `VX_CSR_MPM_RCACHE_MISS_R_H: read_data_ro_r = 32'(perf_rcache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_RCACHE_BANK_ST : read_data_ro_r = perf_rcache_if.bank_stalls[31:0];
                        `VX_CSR_MPM_RCACHE_BANK_ST_H:read_data_ro_r = 32'(perf_rcache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_RCACHE_MSHR_ST  :read_data_ro_r = perf_rcache_if.mshr_stalls[31:0];
                        `VX_CSR_MPM_RCACHE_MSHR_ST_H:read_data_ro_r = 32'(perf_rcache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `VX_CSR_MPM_RASTER_ISSUE_ST   : read_data_ro_r = gpu_perf_if.raster_stalls[31:0];
                        `VX_CSR_MPM_RASTER_ISSUE_ST_H : read_data_ro_r = 32'(gpu_perf_if.raster_stalls[`PERF_CTR_BITS-1:32]);
                        default:;
                        endcase
                    `endif
                    end
                    `VX_DCR_MPM_CLASS_ROP: begin
                    `ifdef EXT_ROP_ENABLE
                        case (read_addr)
                        `VX_CSR_MPM_ROP_READS      : read_data_ro_r = perf_rop_if.mem_reads[31:0];
                        `VX_CSR_MPM_ROP_READS_H    : read_data_ro_r = 32'(perf_rop_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ROP_WRITES     : read_data_ro_r = perf_rop_if.mem_writes[31:0];
                        `VX_CSR_MPM_ROP_WRITES_H   : read_data_ro_r = 32'(perf_rop_if.mem_writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ROP_LAT        : read_data_ro_r = perf_rop_if.mem_latency[31:0];
                        `VX_CSR_MPM_ROP_LAT_H      : read_data_ro_r = 32'(perf_rop_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ROP_STALL      : read_data_ro_r = perf_rop_if.stall_cycles[31:0];
                        `VX_CSR_MPM_ROP_STALL_H    : read_data_ro_r = 32'(perf_rop_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef OCACHE_ENABLE
                        // cache perf counters
                        `VX_CSR_MPM_OCACHE_READS   : read_data_ro_r = perf_ocache_if.reads[31:0];
                        `VX_CSR_MPM_OCACHE_READS_H : read_data_ro_r = 32'(perf_ocache_if.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_OCACHE_WRITES  : read_data_ro_r = perf_ocache_if.writes[31:0];
                        `VX_CSR_MPM_OCACHE_WRITES_H: read_data_ro_r = 32'(perf_ocache_if.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_OCACHE_MISS_R  : read_data_ro_r = perf_ocache_if.read_misses[31:0];
                        `VX_CSR_MPM_OCACHE_MISS_R_H: read_data_ro_r = 32'(perf_ocache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_OCACHE_MISS_W  : read_data_ro_r = perf_ocache_if.write_misses[31:0];
                        `VX_CSR_MPM_OCACHE_MISS_W_H: read_data_ro_r = 32'(perf_ocache_if.write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_OCACHE_BANK_ST : read_data_ro_r = perf_ocache_if.bank_stalls[31:0];
                        `VX_CSR_MPM_OCACHE_BANK_ST_H:read_data_ro_r = 32'(perf_ocache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_OCACHE_MSHR_ST  :read_data_ro_r = perf_ocache_if.mshr_stalls[31:0];
                        `VX_CSR_MPM_OCACHE_MSHR_ST_H:read_data_ro_r = 32'(perf_ocache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `VX_CSR_MPM_ROP_ISSUE_ST   : read_data_ro_r = gpu_perf_if.rop_stalls[31:0];
                        `VX_CSR_MPM_ROP_ISSUE_ST_H : read_data_ro_r = 32'(gpu_perf_if.rop_stalls[`PERF_CTR_BITS-1:32]);
                        default:;
                        endcase
                    `endif
                    end
                    default:;
                    endcase
                `endif
                end
            end
        endcase
    end

    assign read_data_ro = read_data_ro_r;
    assign read_data_rw = read_data_rw_r;

    `UNUSED_VAR (base_dcrs)

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_r, ("%t: *** invalid CSR read address: 0x%0h (#%0d)", $time, read_addr, read_uuid))

`ifdef EXT_F_ENABLE    
    assign fpu_to_csr_if.read_frm = fcsr[fpu_to_csr_if.read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS];
`endif

`ifdef PERF_ENABLE
    wire [`PERF_CTR_BITS-1:0] perf_wctl_stalls = gpu_perf_if.wctl_stalls;
    `UNUSED_VAR (perf_wctl_stalls);
`endif

endmodule
