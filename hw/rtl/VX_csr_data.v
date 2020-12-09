`include "VX_define.vh"

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if                perf_memsys_if,
    VX_perf_pipeline_if             perf_pipeline_if,
`endif

    VX_cmt_to_csr_if                cmt_to_csr_if,
    VX_fpu_to_csr_if                fpu_to_csr_if,  

    input wire                      read_enable,
    input wire[`CSR_ADDR_BITS-1:0]  read_addr,
    input wire[`NW_BITS-1:0]        read_wid,
    output wire[31:0]               read_data,

    input wire                      write_enable, 
    input wire[`CSR_ADDR_BITS-1:0]  write_addr,
    input wire[`NW_BITS-1:0]        write_wid,
    input wire[`CSR_WIDTH-1:0]      write_data,
    
    input wire                      busy
);
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
    
    reg [`FFG_BITS-1:0]           csr_fflags [`NUM_WARPS-1:0];
    reg [`FRM_BITS-1:0]           csr_frm [`NUM_WARPS-1:0];
    reg [`FRM_BITS+`FFG_BITS-1:0] csr_fcsr [`NUM_WARPS-1:0];  // fflags + frm

    reg [31:0] read_data_r;

    always @(posedge clk) begin
        if (fpu_to_csr_if.write_enable) begin
            csr_fflags[fpu_to_csr_if.write_wid]               <= fpu_to_csr_if.write_fflags;
            csr_fcsr[fpu_to_csr_if.write_wid][`FFG_BITS-1:0]  <= fpu_to_csr_if.write_fflags;
        end

        if (write_enable) begin
            case (write_addr)
                `CSR_FFLAGS: begin
                    csr_fcsr[write_wid][`FFG_BITS-1:0]  <= write_data[`FFG_BITS-1:0];
                    csr_fflags[write_wid]               <= write_data[`FFG_BITS-1:0];
                end

                `CSR_FRM: begin
                    csr_fcsr[write_wid][`FFG_BITS+`FRM_BITS-1:`FFG_BITS] <= write_data[`FRM_BITS-1:0];
                    csr_frm[write_wid]                                   <= write_data[`FRM_BITS-1:0];
                end

                `CSR_FCSR: begin
                    csr_fcsr[write_wid]   <= write_data[`FFG_BITS+`FRM_BITS-1:0];
                    csr_frm[write_wid]    <= write_data[`FFG_BITS+`FRM_BITS-1:`FFG_BITS];
                    csr_fflags[write_wid] <= write_data[`FFG_BITS-1:0];
                end

                `CSR_SATP:     csr_satp   <= write_data;
                
                `CSR_MSTATUS:  csr_mstatus <= write_data;
                `CSR_MEDELEG:  csr_medeleg <= write_data;
                `CSR_MIDELEG:  csr_mideleg <= write_data;
                `CSR_MIE:      csr_mie     <= write_data;
                `CSR_MTVEC:    csr_mtvec   <= write_data;

                `CSR_MEPC:     csr_mepc    <= write_data;

                `CSR_PMPCFG0:  csr_pmpcfg[0]  <= write_data;
                `CSR_PMPADDR0: csr_pmpaddr[0] <= write_data;

                default: begin           
                    assert(~write_enable) else $error("%t: invalid CSR write address: %0h", $time, write_addr);
                end
            endcase                
        end
    end

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

    always @(*) begin
        read_data_r = 'x;
        case (read_addr)
            `CSR_FFLAGS    : read_data_r = 32'(csr_fflags[read_wid]);
            `CSR_FRM       : read_data_r = 32'(csr_frm[read_wid]);
            `CSR_FCSR      : read_data_r = 32'(csr_fcsr[read_wid]);

            `CSR_LWID      : read_data_r = 32'(read_wid);
            `CSR_LTID      ,
            `CSR_GTID      ,
            `CSR_MHARTID   ,
            `CSR_GWID      : read_data_r = CORE_ID * `NUM_WARPS + 32'(read_wid);
            `CSR_GCID      : read_data_r = CORE_ID;
            `CSR_NT        : read_data_r = `NUM_THREADS;
            `CSR_NW        : read_data_r = `NUM_WARPS;
            `CSR_NC        : read_data_r = `NUM_CORES * `NUM_CLUSTERS;
            
        `ifdef PERF_ENABLE
            // PERF: pipeline
            `CSR_MPM_ICACHE_ST    : read_data_r = perf_pipeline_if.icache_stalls[31:0];
            `CSR_MPM_ICACHE_ST_H  : read_data_r = perf_pipeline_if.icache_stalls[63:32];
            `CSR_MPM_IBUF_ST      : read_data_r = perf_pipeline_if.ibuffer_stalls[31:0];
            `CSR_MPM_IBUF_ST_H    : read_data_r = perf_pipeline_if.ibuffer_stalls[63:32];
            `CSR_MPM_SCRB_ST      : read_data_r = perf_pipeline_if.scoreboard_stalls[31:0];
            `CSR_MPM_SCRB_ST_H    : read_data_r = perf_pipeline_if.scoreboard_stalls[63:32];
            `CSR_MPM_ALU_ST       : read_data_r = perf_pipeline_if.alu_stalls[31:0];
            `CSR_MPM_ALU_ST_H     : read_data_r = perf_pipeline_if.alu_stalls[63:32];
            `CSR_MPM_LSU_ST       : read_data_r = perf_pipeline_if.lsu_stalls[31:0];
            `CSR_MPM_LSU_ST_H     : read_data_r = perf_pipeline_if.lsu_stalls[63:32];
            `CSR_MPM_CSR_ST       : read_data_r = perf_pipeline_if.csr_stalls[31:0];
            `CSR_MPM_CSR_ST_H     : read_data_r = perf_pipeline_if.csr_stalls[63:32];
            `CSR_MPM_MUL_ST       : read_data_r = perf_pipeline_if.mul_stalls[31:0];
            `CSR_MPM_MUL_ST_H     : read_data_r = perf_pipeline_if.mul_stalls[63:32];
            `CSR_MPM_FPU_ST       : read_data_r = perf_pipeline_if.fpu_stalls[31:0];
            `CSR_MPM_FPU_ST_H     : read_data_r = perf_pipeline_if.fpu_stalls[63:32];
            `CSR_MPM_GPU_ST       : read_data_r = perf_pipeline_if.gpu_stalls[31:0];
            `CSR_MPM_GPU_ST_H     : read_data_r = perf_pipeline_if.gpu_stalls[63:32];
            // PERF: icache
            `CSR_MPM_ICACHE_MISS_R      : read_data_r = perf_memsys_if.icache_read_misses[31:0];
            `CSR_MPM_ICACHE_MISS_R_H    : read_data_r = perf_memsys_if.icache_read_misses[63:32];
            `CSR_MPM_ICACHE_DREQ_ST     : read_data_r = perf_memsys_if.icache_dreq_stalls[31:0];
            `CSR_MPM_ICACHE_DREQ_ST_H   : read_data_r = perf_memsys_if.icache_dreq_stalls[63:32];
            `CSR_MPM_ICACHE_CRSP_ST     : read_data_r = perf_memsys_if.icache_crsp_stalls[31:0];
            `CSR_MPM_ICACHE_CRSP_ST_H   : read_data_r = perf_memsys_if.icache_crsp_stalls[63:32];
            `CSR_MPM_ICACHE_MSHR_ST     : read_data_r = perf_memsys_if.icache_mshr_stalls[31:0];
            `CSR_MPM_ICACHE_MSHR_ST_H   : read_data_r = perf_memsys_if.icache_mshr_stalls[63:32];
            `CSR_MPM_ICACHE_PIPE_ST     : read_data_r = perf_memsys_if.icache_pipe_stalls[31:0];
            `CSR_MPM_ICACHE_PIPE_ST_H   : read_data_r = perf_memsys_if.icache_pipe_stalls[63:32];
            `CSR_MPM_ICACHE_READS       : read_data_r = perf_memsys_if.icache_reads[31:0];
            `CSR_MPM_ICACHE_READS_H     : read_data_r = perf_memsys_if.icache_reads[63:32];
            // PERF: dcache
            `CSR_MPM_DCACHE_MISS_R      : read_data_r = perf_memsys_if.dcache_read_misses[31:0];
            `CSR_MPM_DCACHE_MISS_R_H    : read_data_r = perf_memsys_if.dcache_read_misses[63:32];
            `CSR_MPM_DCACHE_MISS_W      : read_data_r = perf_memsys_if.dcache_write_misses[31:0];
            `CSR_MPM_DCACHE_MISS_W_H    : read_data_r = perf_memsys_if.dcache_write_misses[63:32];
            `CSR_MPM_DCACHE_DREQ_ST     : read_data_r = perf_memsys_if.dcache_dreq_stalls[31:0];
            `CSR_MPM_DCACHE_DREQ_ST_H   : read_data_r = perf_memsys_if.dcache_dreq_stalls[63:32];
            `CSR_MPM_DCACHE_CRSP_ST     : read_data_r = perf_memsys_if.dcache_crsp_stalls[31:0];
            `CSR_MPM_DCACHE_CRSP_ST_H   : read_data_r = perf_memsys_if.dcache_crsp_stalls[63:32];
            `CSR_MPM_DCACHE_MSHR_ST     : read_data_r = perf_memsys_if.dcache_mshr_stalls[31:0];
            `CSR_MPM_DCACHE_MSHR_ST_H   : read_data_r = perf_memsys_if.dcache_mshr_stalls[63:32];
            `CSR_MPM_DCACHE_PIPE_ST     : read_data_r = perf_memsys_if.dcache_pipe_stalls[31:0];
            `CSR_MPM_DCACHE_PIPE_ST_H   : read_data_r = perf_memsys_if.dcache_pipe_stalls[63:32];
            `CSR_MPM_DCACHE_READS       : read_data_r = perf_memsys_if.dcache_reads[31:0];
            `CSR_MPM_DCACHE_READS_H     : read_data_r = perf_memsys_if.dcache_reads[63:32];
            `CSR_MPM_DCACHE_WRITES      : read_data_r = perf_memsys_if.dcache_writes[31:0];
            `CSR_MPM_DCACHE_WRITES_H    : read_data_r = perf_memsys_if.dcache_writes[63:32];
            `CSR_MPM_DCACHE_EVICTS      : read_data_r = perf_memsys_if.dcache_evictions[31:0];
            `CSR_MPM_DCACHE_EVICTS_H    : read_data_r = perf_memsys_if.dcache_evictions[63:32];
            // PERF: memory
            `CSR_MPM_DRAM_LAT     : read_data_r = perf_memsys_if.dram_latency[31:0];
            `CSR_MPM_DRAM_LAT_H   : read_data_r = perf_memsys_if.dram_latency[63:32];
            `CSR_MPM_DRAM_REQ     : read_data_r = perf_memsys_if.dram_requests[31:0];
            `CSR_MPM_DRAM_REQ_H   : read_data_r = perf_memsys_if.dram_requests[63:32];
            `CSR_MPM_DRAM_RSP     : read_data_r = perf_memsys_if.dram_responses[31:0];
            `CSR_MPM_DRAM_RSP_H   : read_data_r = perf_memsys_if.dram_responses[63:32];
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
            
            `CSR_MCYCLE    : read_data_r = csr_cycle[31:0];
            `CSR_MCYCLE_H  : read_data_r = csr_cycle[63:32];
            `CSR_MINSTRET  : read_data_r = csr_instret[31:0];
            `CSR_MINSTRET_H: read_data_r = csr_instret[63:32];
            
            `CSR_MVENDORID : read_data_r = `VENDOR_ID;
            `CSR_MARCHID   : read_data_r = `ARCHITECTURE_ID;
            `CSR_MIMPID    : read_data_r = `IMPLEMENTATION_ID;

            default: begin                      
                assert(~read_enable) else $error("%t: invalid CSR read address: %0h", $time, read_addr);
            end
        endcase
    end 

    assign read_data = read_data_r;
    assign fpu_to_csr_if.read_frm = csr_frm[fpu_to_csr_if.read_wid];

endmodule