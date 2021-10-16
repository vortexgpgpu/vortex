`include "VX_define.vh"

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave         perf_memsys_if,
    VX_perf_pipeline_if.slave       perf_pipeline_if,
`endif

    VX_cmt_to_csr_if.slave          cmt_to_csr_if,
    VX_fetch_to_csr_if.slave        fetch_to_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave          fpu_to_csr_if,
`endif
`ifdef EXT_TEX_ENABLE
    VX_tex_csr_if.master            tex_csr_if,
`endif 

    input wire                      read_enable,
    input wire[`CSR_ADDR_BITS-1:0]  read_addr,
    input wire[`NW_BITS-1:0]        read_wid,
    output wire[31:0]               read_data,

    input wire                      write_enable, 
    input wire[`CSR_ADDR_BITS-1:0]  write_addr,
    input wire[`NW_BITS-1:0]        write_wid,
    input wire[31:0]                write_data,
    
    input wire                      busy
);
    import fpu_types::*;
    
    reg [`CSR_WIDTH-1:0] csr_satp;
    reg [`CSR_WIDTH-1:0] csr_mstatus;
    reg [`CSR_WIDTH-1:0] csr_medeleg;
    reg [`CSR_WIDTH-1:0] csr_mideleg;
    reg [`CSR_WIDTH-1:0] csr_mie;
    reg [`CSR_WIDTH-1:0] csr_mtvec;
    reg [`CSR_WIDTH-1:0] csr_mepc;    
    reg [`CSR_WIDTH-1:0] csr_pmpcfg [0:0];
    reg [`CSR_WIDTH-1:0] csr_pmpaddr [0:0];
    reg [63:0] csr_cycle;
    reg [63:0] csr_instret;
    
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FFLAGS_BITS-1:0] fcsr;

    always @(posedge clk) begin
    `ifdef EXT_F_ENABLE
        if (reset) begin
            fcsr <= '0;
        end
        if (fpu_to_csr_if.write_enable) begin
            fcsr[fpu_to_csr_if.write_wid][`FFLAGS_BITS-1:0] <= fcsr[fpu_to_csr_if.write_wid][`FFLAGS_BITS-1:0]
                                                             | fpu_to_csr_if.write_fflags;
        end
    `endif
        if (write_enable) begin
            case (write_addr)
                `CSR_FFLAGS:   fcsr[write_wid][`FFLAGS_BITS-1:0] <= write_data[`FFLAGS_BITS-1:0];
                `CSR_FRM:      fcsr[write_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS] <= write_data[`INST_FRM_BITS-1:0];
                `CSR_FCSR:     fcsr[write_wid] <= write_data[`FFLAGS_BITS+`INST_FRM_BITS-1:0];
                `CSR_SATP:     csr_satp       <= write_data[`CSR_WIDTH-1:0];
                `CSR_MSTATUS:  csr_mstatus    <= write_data[`CSR_WIDTH-1:0];
                `CSR_MEDELEG:  csr_medeleg    <= write_data[`CSR_WIDTH-1:0];
                `CSR_MIDELEG:  csr_mideleg    <= write_data[`CSR_WIDTH-1:0];
                `CSR_MIE:      csr_mie        <= write_data[`CSR_WIDTH-1:0];
                `CSR_MTVEC:    csr_mtvec      <= write_data[`CSR_WIDTH-1:0];
                `CSR_MEPC:     csr_mepc       <= write_data[`CSR_WIDTH-1:0];
                `CSR_PMPCFG0:  csr_pmpcfg[0]  <= write_data[`CSR_WIDTH-1:0];
                `CSR_PMPADDR0: csr_pmpaddr[0] <= write_data[`CSR_WIDTH-1:0];
                default: begin
                    `ASSERT(write_addr >= `CSR_TEX_BEGIN(0)
                         && write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES),
                            ("%t: invalid CSR write address: %0h", $time, write_addr));
                end
            endcase
        end
    end

    `UNUSED_VAR (write_data)

    // TEX CSRs
`ifdef EXT_TEX_ENABLE    
    assign tex_csr_if.write_enable = write_enable;
    assign tex_csr_if.write_addr   = write_addr;
    assign tex_csr_if.write_data   = write_data;
`endif

    always @(posedge clk) begin
       if (reset) begin
            csr_cycle   <= 0;
            csr_instret <= 0;
        end else begin
            if (busy) begin
                csr_cycle <= csr_cycle + 1;
            end
            if (cmt_to_csr_if.valid) begin
                csr_instret <= csr_instret + 64'(cmt_to_csr_if.commit_size);
            end
        end
    end

    reg [31:0] read_data_r;
    reg read_addr_valid_r;

    always @(*) begin
        read_data_r = 'x;
        read_addr_valid_r = 1;
        case (read_addr)
            `CSR_FFLAGS     : read_data_r = 32'(fcsr[read_wid][`FFLAGS_BITS-1:0]);
            `CSR_FRM        : read_data_r = 32'(fcsr[read_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS]);
            `CSR_FCSR       : read_data_r = 32'(fcsr[read_wid]);

            `CSR_WTID       ,            
            `CSR_LTID       ,
            `CSR_LWID       : read_data_r = 32'(read_wid);            
            `CSR_GTID       ,
            /*`CSR_MHARTID ,*/
            `CSR_GWID       : read_data_r = CORE_ID * `NUM_WARPS + 32'(read_wid);
            `CSR_GCID       : read_data_r = CORE_ID;

            `CSR_TMASK      : read_data_r = 32'(fetch_to_csr_if.thread_masks[read_wid]);

            `CSR_NT         : read_data_r = `NUM_THREADS;
            `CSR_NW         : read_data_r = `NUM_WARPS;
            `CSR_NC         : read_data_r = `NUM_CORES * `NUM_CLUSTERS;
            
            `CSR_MCYCLE     : read_data_r = csr_cycle[31:0];
            `CSR_MCYCLE_H   : read_data_r = 32'(csr_cycle[`PERF_CTR_BITS-1:32]);
            `CSR_MINSTRET   : read_data_r = csr_instret[31:0];
            `CSR_MINSTRET_H : read_data_r = 32'(csr_instret[`PERF_CTR_BITS-1:32]);
            
        `ifdef PERF_ENABLE
            // PERF: pipeline
            `CSR_MPM_IBUF_ST    : read_data_r = perf_pipeline_if.ibf_stalls[31:0];
            `CSR_MPM_IBUF_ST_H  : read_data_r = 32'(perf_pipeline_if.ibf_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_SCRB_ST    : read_data_r = perf_pipeline_if.scb_stalls[31:0];
            `CSR_MPM_SCRB_ST_H  : read_data_r = 32'(perf_pipeline_if.scb_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_ALU_ST     : read_data_r = perf_pipeline_if.alu_stalls[31:0];
            `CSR_MPM_ALU_ST_H   : read_data_r = 32'(perf_pipeline_if.alu_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_LSU_ST     : read_data_r = perf_pipeline_if.lsu_stalls[31:0];
            `CSR_MPM_LSU_ST_H   : read_data_r = 32'(perf_pipeline_if.lsu_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_CSR_ST     : read_data_r = perf_pipeline_if.csr_stalls[31:0];
            `CSR_MPM_CSR_ST_H   : read_data_r = 32'(perf_pipeline_if.csr_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_FPU_ST     : read_data_r = perf_pipeline_if.fpu_stalls[31:0];
            `CSR_MPM_FPU_ST_H   : read_data_r = 32'(perf_pipeline_if.fpu_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_GPU_ST     : read_data_r = perf_pipeline_if.gpu_stalls[31:0];
            `CSR_MPM_GPU_ST_H   : read_data_r = 32'(perf_pipeline_if.gpu_stalls[`PERF_CTR_BITS-1:32]);
            // PERF: icache
            `CSR_MPM_ICACHE_READS       : read_data_r = perf_memsys_if.icache_reads[31:0];
            `CSR_MPM_ICACHE_READS_H     : read_data_r = 32'(perf_memsys_if.icache_reads[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_ICACHE_MISS_R      : read_data_r = perf_memsys_if.icache_read_misses[31:0];
            `CSR_MPM_ICACHE_MISS_R_H    : read_data_r = 32'(perf_memsys_if.icache_read_misses[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_ICACHE_PIPE_ST     : read_data_r = perf_memsys_if.icache_pipe_stalls[31:0];
            `CSR_MPM_ICACHE_PIPE_ST_H   : read_data_r = 32'(perf_memsys_if.icache_pipe_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_ICACHE_CRSP_ST     : read_data_r = perf_memsys_if.icache_crsp_stalls[31:0];
            `CSR_MPM_ICACHE_CRSP_ST_H   : read_data_r = 32'(perf_memsys_if.icache_crsp_stalls[`PERF_CTR_BITS-1:32]);
            // PERF: dcache            
            `CSR_MPM_DCACHE_READS       : read_data_r = perf_memsys_if.dcache_reads[31:0];
            `CSR_MPM_DCACHE_READS_H     : read_data_r = 32'(perf_memsys_if.dcache_reads[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_WRITES      : read_data_r = perf_memsys_if.dcache_writes[31:0];
            `CSR_MPM_DCACHE_WRITES_H    : read_data_r = 32'(perf_memsys_if.dcache_writes[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_MISS_R      : read_data_r = perf_memsys_if.dcache_read_misses[31:0];
            `CSR_MPM_DCACHE_MISS_R_H    : read_data_r = 32'(perf_memsys_if.dcache_read_misses[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_MISS_W      : read_data_r = perf_memsys_if.dcache_write_misses[31:0];
            `CSR_MPM_DCACHE_MISS_W_H    : read_data_r = 32'(perf_memsys_if.dcache_write_misses[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_BANK_ST     : read_data_r = perf_memsys_if.dcache_bank_stalls[31:0];
            `CSR_MPM_DCACHE_BANK_ST_H   : read_data_r = 32'(perf_memsys_if.dcache_bank_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_MSHR_ST     : read_data_r = perf_memsys_if.dcache_mshr_stalls[31:0];
            `CSR_MPM_DCACHE_MSHR_ST_H   : read_data_r = 32'(perf_memsys_if.dcache_mshr_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_PIPE_ST     : read_data_r = perf_memsys_if.dcache_pipe_stalls[31:0];
            `CSR_MPM_DCACHE_PIPE_ST_H   : read_data_r = 32'(perf_memsys_if.dcache_pipe_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_DCACHE_CRSP_ST     : read_data_r = perf_memsys_if.dcache_crsp_stalls[31:0];
            `CSR_MPM_DCACHE_CRSP_ST_H   : read_data_r = 32'(perf_memsys_if.dcache_crsp_stalls[`PERF_CTR_BITS-1:32]);
            // PERF: smem            
            `CSR_MPM_SMEM_READS     : read_data_r = perf_memsys_if.smem_reads[31:0];
            `CSR_MPM_SMEM_READS_H   : read_data_r = 32'(perf_memsys_if.smem_reads[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_SMEM_WRITES    : read_data_r = perf_memsys_if.smem_writes[31:0];
            `CSR_MPM_SMEM_WRITES_H  : read_data_r = 32'(perf_memsys_if.smem_writes[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_SMEM_BANK_ST   : read_data_r = perf_memsys_if.smem_bank_stalls[31:0];
            `CSR_MPM_SMEM_BANK_ST_H : read_data_r = 32'(perf_memsys_if.smem_bank_stalls[`PERF_CTR_BITS-1:32]);
            // PERF: MEM
            `CSR_MPM_MEM_READS      : read_data_r = perf_memsys_if.mem_reads[31:0];
            `CSR_MPM_MEM_READS_H    : read_data_r = 32'(perf_memsys_if.mem_reads[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_MEM_WRITES     : read_data_r = perf_memsys_if.mem_writes[31:0];
            `CSR_MPM_MEM_WRITES_H   : read_data_r = 32'(perf_memsys_if.mem_writes[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_MEM_ST         : read_data_r = perf_memsys_if.mem_stalls[31:0];
            `CSR_MPM_MEM_ST_H       : read_data_r = 32'(perf_memsys_if.mem_stalls[`PERF_CTR_BITS-1:32]);
            `CSR_MPM_MEM_LAT        : read_data_r = perf_memsys_if.mem_latency[31:0];
            `CSR_MPM_MEM_LAT_H      : read_data_r = 32'(perf_memsys_if.mem_latency[`PERF_CTR_BITS-1:32]);
            // PERF: reserved            
            `CSR_MPM_RESERVED       : read_data_r = '0;
            `CSR_MPM_RESERVED_H     : read_data_r = '0;
        `endif
            
            `CSR_SATP      : read_data_r = 32'(csr_satp);
            
            `CSR_MSTATUS   : read_data_r = 32'(csr_mstatus);
            `CSR_MISA      : read_data_r = `ISA_CODE;
            `CSR_MEDELEG   : read_data_r = 32'(csr_medeleg);
            `CSR_MIDELEG   : read_data_r = 32'(csr_mideleg);
            `CSR_MIE       : read_data_r = 32'(csr_mie);
            `CSR_MTVEC     : read_data_r = 32'(csr_mtvec);

            `CSR_MEPC      : read_data_r = 32'(csr_mepc);

            `CSR_PMPCFG0   : read_data_r = 32'(csr_pmpcfg[0]);
            `CSR_PMPADDR0  : read_data_r = 32'(csr_pmpaddr[0]);
            
            `CSR_MVENDORID : read_data_r = `VENDOR_ID;
            `CSR_MARCHID   : read_data_r = `ARCHITECTURE_ID;
            `CSR_MIMPID    : read_data_r = `IMPLEMENTATION_ID;

            default: begin
                if (!((read_addr >= `CSR_MPM_BASE && read_addr < (`CSR_MPM_BASE + 32))
                   || (read_addr >= `CSR_MPM_BASE_H && read_addr < (`CSR_MPM_BASE_H + 32)
                   || (read_addr >= `CSR_TEX_BEGIN(0) && read_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES))))) begin
                    read_addr_valid_r = 0;
                end
            end
        endcase
    end 

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_r, ("invalid CSR read address: %0h", read_addr))

    assign read_data = read_data_r;

`ifdef EXT_F_ENABLE    
    assign fpu_to_csr_if.read_frm = fcsr[fpu_to_csr_if.read_wid][`INST_FRM_BITS+`FFLAGS_BITS-1:`FFLAGS_BITS];
`endif

endmodule