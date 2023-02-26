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
    input wire clk,
    input wire reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave             perf_memsys_if,
    VX_perf_pipeline_if.slave           perf_pipeline_if,
    VX_perf_gpu_if.slave                perf_gpu_if,
`ifdef EXT_TEX_ENABLE
    VX_tex_perf_if.slave                perf_tex_if,
    VX_perf_cache_if.slave              perf_tcache_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_perf_if.slave             perf_raster_if,
    VX_perf_cache_if.slave              perf_rcache_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_perf_if.slave                perf_rop_if,
    VX_perf_cache_if.slave              perf_ocache_if,
`endif
`endif

    VX_cmt_to_csr_if.slave              cmt_to_csr_if,
    VX_fetch_to_csr_if.slave            fetch_to_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave              fpu_to_csr_if,
`endif

    input wire                          read_enable,
    input wire [`UP(`UUID_BITS)-1:0]    read_uuid,
    input wire [`UP(`NW_BITS)-1:0]      read_wid,
    input wire [`NUM_THREADS-1:0]       read_tmask,
    input wire [`CSR_ADDR_BITS-1:0]     read_addr,
    output wire [`XLEN-1:0]             read_data_ro,
    output wire [`XLEN-1:0]             read_data_rw,

    input wire                          write_enable, 
    input wire [`UP(`UUID_BITS)-1:0]    write_uuid,
    input wire [`UP(`NW_BITS)-1:0]      write_wid,
    input wire [`CSR_ADDR_BITS-1:0]     write_addr,
    input wire [`XLEN-1:0]              write_data
);

    `UNUSED_VAR (reset)
    `UNUSED_VAR (write_wid)

    // CSRs Write /////////////////////////////////////////////////////////////

`ifdef EXT_F_ENABLE    
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr;
`endif

    reg [`XLEN-1:0] csr_satp;
    reg [`XLEN-1:0] csr_mstatus;
    reg [`XLEN-1:0] csr_medeleg;
    reg [`XLEN-1:0] csr_mideleg;
    reg [`XLEN-1:0] csr_mie;
    reg [`XLEN-1:0] csr_mtvec;
    reg [`XLEN-1:0] csr_mepc;    
    reg [`XLEN-1:0] csr_pmpcfg;
    reg [`XLEN-1:0] csr_pmpaddr;

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
                `CSR_FFLAGS:   fcsr[write_wid][`FP_FLAGS_BITS-1:0] <= write_data[`FP_FLAGS_BITS-1:0];
                `CSR_FRM:      fcsr[write_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] <= write_data[`INST_FRM_BITS-1:0];
                `CSR_FCSR:     fcsr[write_wid] <= write_data[`FP_FLAGS_BITS+`INST_FRM_BITS-1:0];
            `endif
                `CSR_SATP:     csr_satp     <= write_data;
                `CSR_MSTATUS:  csr_mstatus  <= write_data;
                `CSR_MEDELEG:  csr_medeleg  <= write_data;
                `CSR_MIDELEG:  csr_mideleg  <= write_data;
                `CSR_MIE:      csr_mie      <= write_data;
                `CSR_MTVEC:    csr_mtvec    <= write_data;
                `CSR_MEPC:     csr_mepc     <= write_data;
                `CSR_PMPCFG0:  csr_pmpcfg   <= write_data;
                `CSR_PMPADDR0: csr_pmpaddr  <= write_data;
                default: begin
                    `ASSERT(0, ("%t: *** invalid CSR write address: %0h (#%0d)", $time, write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] read_data_ro_r;
    reg [`XLEN-1:0] read_data_rw_r;
    reg read_addr_valid_r;

    always @(*) begin
        read_data_ro_r    = '0;
        read_data_rw_r    = '0;
        read_addr_valid_r = 1;
        case (read_addr)
        `ifdef EXT_F_ENABLE
            `CSR_FFLAGS     : read_data_rw_r = `XLEN'(fcsr[read_wid][`FFLAGS_BITS-1:0]);
            `CSR_FRM        : read_data_rw_r = `XLEN'(fcsr[read_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS]);
            `CSR_FCSR       : read_data_rw_r = `XLEN'(fcsr[read_wid]);
        `endif    
            `CSR_LWID       : read_data_ro_r = `XLEN'(read_wid);
            /*`CSR_MHARTID ,*/
            `CSR_GWID       : read_data_ro_r = (`XLEN'(CORE_ID) << `NW_BITS) + `XLEN'(read_wid);
            `CSR_GCID       : read_data_ro_r = `XLEN'(CORE_ID);

            `CSR_TMASK      : read_data_ro_r = `XLEN'(read_tmask);

            `CSR_NT         : read_data_ro_r = `XLEN'd`NUM_THREADS;
            `CSR_NW         : read_data_ro_r = `XLEN'd`NUM_WARPS;
            `CSR_NC         : read_data_ro_r = `XLEN'(`NUM_CORES * `NUM_CLUSTERS);
            
            `CSR_MCYCLE     : read_data_ro_r = `XLEN'(fetch_to_csr_if.cycles[31:0]); // TODO: 64b_extensions Check if range of cycles also needs to be increased.
            `CSR_MCYCLE_H   : read_data_ro_r = `XLEN'(fetch_to_csr_if.cycles[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_RESERVED : read_data_ro_r = 'x;
            `CSR_MPM_RESERVED_H : read_data_ro_r = 'x;  
            `CSR_MINSTRET   : read_data_ro_r = `XLEN'(cmt_to_csr_if.instret[31:0]); // TODO: 64b_extensions Check if range of cycles also needs to be increased.
            `CSR_MINSTRET_H : read_data_ro_r = `XLEN'(cmt_to_csr_if.instret[`PERF_CTR_BITS-1:32]);       
            
            `CSR_SATP       : read_data_ro_r = `XLEN'(csr_satp);
            
            `CSR_MSTATUS    : read_data_ro_r = `XLEN'(csr_mstatus);
            `CSR_MISA       : read_data_ro_r = (((`XLEN'($clog2(`XLEN))-4) << (`XLEN-2)) | `MISA_STD);
            `CSR_MEDELEG    : read_data_ro_r = `XLEN'(csr_medeleg);
            `CSR_MIDELEG    : read_data_ro_r = `XLEN'(csr_mideleg);
            `CSR_MIE        : read_data_ro_r = `XLEN'(csr_mie);
            `CSR_MTVEC      : read_data_ro_r = `XLEN'(csr_mtvec);

            `CSR_MEPC       : read_data_ro_r = `XLEN'(csr_mepc);

            `CSR_PMPCFG0    : read_data_ro_r = `XLEN'(csr_pmpcfg);
            `CSR_PMPADDR0   : read_data_ro_r = `XLEN'(csr_pmpaddr);
            
            `CSR_MVENDORID  : read_data_ro_r = `XLEN'd`VENDOR_ID;
            `CSR_MARCHID    : read_data_ro_r = `XLEN'd`ARCHITECTURE_ID;
            `CSR_MIMPID     : read_data_ro_r = `XLEN'd`IMPLEMENTATION_ID;

            default: begin
                read_addr_valid_r = 0;
                if ((read_addr >= `CSR_MPM_BASE   && read_addr < (`CSR_MPM_BASE + 32))
                 || (read_addr >= `CSR_MPM_BASE_H && read_addr < (`CSR_MPM_BASE_H + 32))) begin
                    read_addr_valid_r = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `CSR_MPM_IBUF_ST        : read_data_ro_r = perf_pipeline_if.ibf_stalls[31:0];
                        `CSR_MPM_IBUF_ST_H      : read_data_ro_r = `XLEN'(perf_pipeline_if.ibf_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_SCRB_ST        : read_data_ro_r = perf_pipeline_if.scb_stalls[31:0];
                        `CSR_MPM_SCRB_ST_H      : read_data_ro_r = `XLEN'(perf_pipeline_if.scb_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_ALU_ST         : read_data_ro_r = perf_pipeline_if.alu_stalls[31:0];
                        `CSR_MPM_ALU_ST_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.alu_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_LSU_ST         : read_data_ro_r = perf_pipeline_if.lsu_stalls[31:0];
                        `CSR_MPM_LSU_ST_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.lsu_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_CSR_ST         : read_data_ro_r = perf_pipeline_if.csr_stalls[31:0];
                        `CSR_MPM_CSR_ST_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.csr_stalls[`PERF_CTR_BITS-1:32]);
                    `ifdef EXT_F_ENABLE
                        `CSR_MPM_FPU_ST         : read_data_ro_r = perf_pipeline_if.fpu_stalls[31:0];
                        `CSR_MPM_FPU_ST_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.fpu_stalls[`PERF_CTR_BITS-1:32]);
                    `else
                        `CSR_MPM_FPU_ST         : read_data_ro_r = '0;
                        `CSR_MPM_FPU_ST_H       : read_data_ro_r = '0;
                    `endif
                        `CSR_MPM_GPU_ST         : read_data_ro_r = perf_pipeline_if.gpu_stalls[31:0];
                        `CSR_MPM_GPU_ST_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.gpu_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: memory
                        `CSR_MPM_IFETCHES       : read_data_ro_r = perf_pipeline_if.ifetches[31:0];
                        `CSR_MPM_IFETCHES_H     : read_data_ro_r = `XLEN'(perf_pipeline_if.ifetches[`PERF_CTR_BITS-1:32]); 
                        `CSR_MPM_LOADS          : read_data_ro_r = perf_pipeline_if.loads[31:0];
                        `CSR_MPM_LOADS_H        : read_data_ro_r = `XLEN'(perf_pipeline_if.loads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_STORES         : read_data_ro_r = perf_pipeline_if.stores[31:0];
                        `CSR_MPM_STORES_H       : read_data_ro_r = `XLEN'(perf_pipeline_if.stores[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_IFETCH_LAT     : read_data_ro_r = perf_pipeline_if.ifetch_latency[31:0];
                        `CSR_MPM_IFETCH_LAT_H   : read_data_ro_r = `XLEN'(perf_pipeline_if.ifetch_latency[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_LOAD_LAT       : read_data_ro_r = perf_pipeline_if.load_latency[31:0];
                        `CSR_MPM_LOAD_LAT_H     : read_data_ro_r = `XLEN'(perf_pipeline_if.load_latency[`PERF_CTR_BITS-1:32]);                
                        default:;
                        endcase
                    end
                    `DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `CSR_MPM_ICACHE_READS       : read_data_ro_r = perf_memsys_if.icache_reads[31:0];
                        `CSR_MPM_ICACHE_READS_H     : read_data_ro_r = `XLEN'(perf_memsys_if.icache_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_ICACHE_MISS_R      : read_data_ro_r = perf_memsys_if.icache_read_misses[31:0];
                        `CSR_MPM_ICACHE_MISS_R_H    : read_data_ro_r = `XLEN'(perf_memsys_if.icache_read_misses[`PERF_CTR_BITS-1:32]);
                        // PERF: dcache
                        `CSR_MPM_DCACHE_READS       : read_data_ro_r = perf_memsys_if.dcache_reads[31:0];
                        `CSR_MPM_DCACHE_READS_H     : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_DCACHE_WRITES      : read_data_ro_r = perf_memsys_if.dcache_writes[31:0];
                        `CSR_MPM_DCACHE_WRITES_H    : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_DCACHE_MISS_R      : read_data_ro_r = perf_memsys_if.dcache_read_misses[31:0];
                        `CSR_MPM_DCACHE_MISS_R_H    : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_DCACHE_MISS_W      : read_data_ro_r = perf_memsys_if.dcache_write_misses[31:0];
                        `CSR_MPM_DCACHE_MISS_W_H    : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_write_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_DCACHE_BANK_ST     : read_data_ro_r = perf_memsys_if.dcache_bank_stalls[31:0];
                        `CSR_MPM_DCACHE_BANK_ST_H   : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_DCACHE_MSHR_ST     : read_data_ro_r = perf_memsys_if.dcache_mshr_stalls[31:0];
                        `CSR_MPM_DCACHE_MSHR_ST_H   : read_data_ro_r = `XLEN'(perf_memsys_if.dcache_mshr_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: smem          
                        `CSR_MPM_SMEM_READS         : read_data_ro_r = perf_memsys_if.smem_reads[31:0];
                        `CSR_MPM_SMEM_READS_H       : read_data_ro_r = `XLEN'(perf_memsys_if.smem_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_SMEM_WRITES        : read_data_ro_r = perf_memsys_if.smem_writes[31:0];
                        `CSR_MPM_SMEM_WRITES_H      : read_data_ro_r = `XLEN'(perf_memsys_if.smem_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_SMEM_BANK_ST       : read_data_ro_r = perf_memsys_if.smem_bank_stalls[31:0];
                        `CSR_MPM_SMEM_BANK_ST_H     : read_data_ro_r = `XLEN'(perf_memsys_if.smem_bank_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: l2cache                        
                        `CSR_MPM_L2CACHE_READS      : read_data_ro_r = perf_memsys_if.l2cache_reads[31:0];
                        `CSR_MPM_L2CACHE_READS_H    : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L2CACHE_WRITES     : read_data_ro_r = perf_memsys_if.l2cache_writes[31:0];
                        `CSR_MPM_L2CACHE_WRITES_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L2CACHE_MISS_R     : read_data_ro_r = perf_memsys_if.l2cache_read_misses[31:0];
                        `CSR_MPM_L2CACHE_MISS_R_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L2CACHE_MISS_W     : read_data_ro_r = perf_memsys_if.l2cache_write_misses[31:0];
                        `CSR_MPM_L2CACHE_MISS_W_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_write_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L2CACHE_BANK_ST    : read_data_ro_r = perf_memsys_if.l2cache_bank_stalls[31:0];
                        `CSR_MPM_L2CACHE_BANK_ST_H  : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L2CACHE_MSHR_ST    : read_data_ro_r = perf_memsys_if.l2cache_mshr_stalls[31:0];
                        `CSR_MPM_L2CACHE_MSHR_ST_H  : read_data_ro_r = `XLEN'(perf_memsys_if.l2cache_mshr_stalls[`PERF_CTR_BITS-1:32]);      
                        // PERF: l3cache
                        `CSR_MPM_L3CACHE_READS      : read_data_ro_r = perf_memsys_if.l3cache_reads[31:0];
                        `CSR_MPM_L3CACHE_READS_H    : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L3CACHE_WRITES     : read_data_ro_r = perf_memsys_if.l3cache_writes[31:0];
                        `CSR_MPM_L3CACHE_WRITES_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L3CACHE_MISS_R     : read_data_ro_r = perf_memsys_if.l3cache_read_misses[31:0];
                        `CSR_MPM_L3CACHE_MISS_R_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L3CACHE_MISS_W     : read_data_ro_r = perf_memsys_if.l3cache_write_misses[31:0];
                        `CSR_MPM_L3CACHE_MISS_W_H   : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_write_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L3CACHE_BANK_ST    : read_data_ro_r = perf_memsys_if.l3cache_bank_stalls[31:0];
                        `CSR_MPM_L3CACHE_BANK_ST_H  : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_L3CACHE_MSHR_ST    : read_data_ro_r = perf_memsys_if.l3cache_mshr_stalls[31:0];
                        `CSR_MPM_L3CACHE_MSHR_ST_H  : read_data_ro_r = `XLEN'(perf_memsys_if.l3cache_mshr_stalls[`PERF_CTR_BITS-1:32]); 
                        // PERF: memory
                        `CSR_MPM_MEM_READS          : read_data_ro_r = perf_memsys_if.mem_reads[31:0];
                        `CSR_MPM_MEM_READS_H        : read_data_ro_r = `XLEN'(perf_memsys_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_MEM_WRITES         : read_data_ro_r = perf_memsys_if.mem_writes[31:0];
                        `CSR_MPM_MEM_WRITES_H       : read_data_ro_r = `XLEN'(perf_memsys_if.mem_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_MEM_LAT            : read_data_ro_r = perf_memsys_if.mem_latency[31:0];
                        `CSR_MPM_MEM_LAT_H          : read_data_ro_r = `XLEN'(perf_memsys_if.mem_latency[`PERF_CTR_BITS-1:32]);     
                        default:;
                        endcase
                    end
                    `DCR_MPM_CLASS_TEX: begin
                    `ifdef EXT_TEX_ENABLE
                        case (read_addr)
                        `CSR_MPM_TEX_READS      : read_data_ro_r = perf_tex_if.mem_reads[31:0];
                        `CSR_MPM_TEX_READS_H    : read_data_ro_r = `XLEN'(perf_tex_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_TEX_LAT        : read_data_ro_r = perf_tex_if.mem_latency[31:0];
                        `CSR_MPM_TEX_LAT_H      : read_data_ro_r = `XLEN'(perf_tex_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_TEX_STALL      : read_data_ro_r = perf_tex_if.stall_cycles[31:0];
                        `CSR_MPM_TEX_STALL_H    : read_data_ro_r = `XLEN'(perf_tex_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef TCACHE_ENABLE
                        // cache perf counters
                        `CSR_MPM_TCACHE_READS   : read_data_ro_r = perf_tcache_if.reads[31:0];
                        `CSR_MPM_TCACHE_READS_H : read_data_ro_r = `XLEN'(perf_tcache_if.reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_TCACHE_MISS_R  : read_data_ro_r = perf_tcache_if.read_misses[31:0];
                        `CSR_MPM_TCACHE_MISS_R_H: read_data_ro_r = `XLEN'(perf_tcache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_TCACHE_BANK_ST : read_data_ro_r = perf_tcache_if.bank_stalls[31:0];
                        `CSR_MPM_TCACHE_BANK_ST_H:read_data_ro_r = `XLEN'(perf_tcache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_TCACHE_MSHR_ST  :read_data_ro_r = perf_tcache_if.mshr_stalls[31:0];
                        `CSR_MPM_TCACHE_MSHR_ST_H:read_data_ro_r = `XLEN'(perf_tcache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `CSR_MPM_TEX_ISSUE_ST   : read_data_ro_r = perf_gpu_if.tex_stalls[31:0];
                        `CSR_MPM_TEX_ISSUE_ST_H : read_data_ro_r = `XLEN'(perf_gpu_if.tex_stalls[`PERF_CTR_BITS-1:32]);
                        default:;
                        endcase
                    `endif
                    end
                    `DCR_MPM_CLASS_RASTER: begin
                    `ifdef EXT_RASTER_ENABLE
                        case (read_addr)
                        `CSR_MPM_RASTER_READS   : read_data_ro_r = perf_raster_if.mem_reads[31:0];
                        `CSR_MPM_RASTER_READS_H : read_data_ro_r = `XLEN'(perf_raster_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_RASTER_LAT     : read_data_ro_r = perf_raster_if.mem_latency[31:0];
                        `CSR_MPM_RASTER_LAT_H   : read_data_ro_r = `XLEN'(perf_raster_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_RASTER_STALL   : read_data_ro_r = perf_raster_if.stall_cycles[31:0];
                        `CSR_MPM_RASTER_STALL_H : read_data_ro_r = `XLEN'(perf_raster_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef RCACHE_ENABLE
                        // cache perf counters
                        `CSR_MPM_RCACHE_READS   : read_data_ro_r = perf_rcache_if.reads[31:0];
                        `CSR_MPM_RCACHE_READS_H : read_data_ro_r = `XLEN'(perf_rcache_if.reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_RCACHE_MISS_R  : read_data_ro_r = perf_rcache_if.read_misses[31:0];
                        `CSR_MPM_RCACHE_MISS_R_H: read_data_ro_r = `XLEN'(perf_rcache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_RCACHE_BANK_ST : read_data_ro_r = perf_rcache_if.bank_stalls[31:0];
                        `CSR_MPM_RCACHE_BANK_ST_H:read_data_ro_r = `XLEN'(perf_rcache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_RCACHE_MSHR_ST  :read_data_ro_r = perf_rcache_if.mshr_stalls[31:0];
                        `CSR_MPM_RCACHE_MSHR_ST_H:read_data_ro_r = `XLEN'(perf_rcache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `CSR_MPM_RASTER_ISSUE_ST   : read_data_ro_r = perf_gpu_if.raster_stalls[31:0];
                        `CSR_MPM_RASTER_ISSUE_ST_H : read_data_ro_r = `XLEN'(perf_gpu_if.raster_stalls[`PERF_CTR_BITS-1:32]);
                        default:;
                        endcase
                    `endif
                    end
                    `DCR_MPM_CLASS_ROP: begin
                    `ifdef EXT_ROP_ENABLE
                        case (read_addr)
                        `CSR_MPM_ROP_READS      : read_data_ro_r = perf_rop_if.mem_reads[31:0];
                        `CSR_MPM_ROP_READS_H    : read_data_ro_r = `XLEN'(perf_rop_if.mem_reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_ROP_WRITES     : read_data_ro_r = perf_rop_if.mem_writes[31:0];
                        `CSR_MPM_ROP_WRITES_H   : read_data_ro_r = `XLEN'(perf_rop_if.mem_writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_ROP_LAT        : read_data_ro_r = perf_rop_if.mem_latency[31:0];
                        `CSR_MPM_ROP_LAT_H      : read_data_ro_r = `XLEN'(perf_rop_if.mem_latency[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_ROP_STALL      : read_data_ro_r = perf_rop_if.stall_cycles[31:0];
                        `CSR_MPM_ROP_STALL_H    : read_data_ro_r = `XLEN'(perf_rop_if.stall_cycles[`PERF_CTR_BITS-1:32]);
                    `ifdef OCACHE_ENABLE
                        // cache perf counters
                        `CSR_MPM_OCACHE_READS   : read_data_ro_r = perf_ocache_if.reads[31:0];
                        `CSR_MPM_OCACHE_READS_H : read_data_ro_r = `XLEN'(perf_ocache_if.reads[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_OCACHE_WRITES  : read_data_ro_r = perf_ocache_if.writes[31:0];
                        `CSR_MPM_OCACHE_WRITES_H: read_data_ro_r = `XLEN'(perf_ocache_if.writes[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_OCACHE_MISS_R  : read_data_ro_r = perf_ocache_if.read_misses[31:0];
                        `CSR_MPM_OCACHE_MISS_R_H: read_data_ro_r = `XLEN'(perf_ocache_if.read_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_OCACHE_MISS_W  : read_data_ro_r = perf_ocache_if.write_misses[31:0];
                        `CSR_MPM_OCACHE_MISS_W_H: read_data_ro_r = `XLEN'(perf_ocache_if.write_misses[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_OCACHE_BANK_ST : read_data_ro_r = perf_ocache_if.bank_stalls[31:0];
                        `CSR_MPM_OCACHE_BANK_ST_H:read_data_ro_r = `XLEN'(perf_ocache_if.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `CSR_MPM_OCACHE_MSHR_ST  :read_data_ro_r = perf_ocache_if.mshr_stalls[31:0];
                        `CSR_MPM_OCACHE_MSHR_ST_H:read_data_ro_r = `XLEN'(perf_ocache_if.mshr_stalls[`PERF_CTR_BITS-1:32]);
                    `endif
                        `CSR_MPM_ROP_ISSUE_ST   : read_data_ro_r = perf_gpu_if.rop_stalls[31:0];
                        `CSR_MPM_ROP_ISSUE_ST_H : read_data_ro_r = `XLEN'(perf_gpu_if.rop_stalls[`PERF_CTR_BITS-1:32]);
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
`ifdef EXT_IMADD_ENABLE
    wire [`PERF_CTR_BITS-1:0] perf_imadd_stalls = perf_gpu_if.imadd_stalls;
    `UNUSED_VAR (perf_imadd_stalls);
`endif
    wire [`PERF_CTR_BITS-1:0] perf_wctl_stalls = perf_gpu_if.wctl_stalls;
    `UNUSED_VAR (perf_wctl_stalls);
`endif

endmodule
