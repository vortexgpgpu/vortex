// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_csr_data
import VX_gpu_pkg::*;
`ifdef EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
#(
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave                mem_perf_if,
    VX_pipeline_perf_if.slave           pipeline_perf_if,
`endif

    VX_commit_csr_if.slave              commit_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave              fpu_to_csr_if [`NUM_FPU_BLOCKS],
`endif

    input wire [`PERF_CTR_BITS-1:0]     cycles,
    input wire [`NUM_WARPS-1:0]         active_warps,
    input wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks,

    input wire                          read_enable,
    input wire [`UUID_WIDTH-1:0]        read_uuid,
    input wire [`NW_WIDTH-1:0]          read_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [31:0]                  read_data_ro,
    output wire [31:0]                  read_data_rw,

    input wire                          write_enable, 
    input wire [`UUID_WIDTH-1:0]        write_uuid,
    input wire [`NW_WIDTH-1:0]          write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [31:0]                   write_data
);

    `UNUSED_VAR (reset)
    `UNUSED_VAR (write_wid)
    `UNUSED_VAR (write_data)

    // CSRs Write /////////////////////////////////////////////////////////////

`ifdef EXT_F_ENABLE    
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr, fcsr_n;
    wire [`NUM_FPU_BLOCKS-1:0]              fpu_write_enable;
    wire [`NUM_FPU_BLOCKS-1:0][`NW_WIDTH-1:0] fpu_write_wid;
    fflags_t [`NUM_FPU_BLOCKS-1:0]          fpu_write_fflags;
    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
        assign fpu_write_enable[i] = fpu_to_csr_if[i].write_enable;
        assign fpu_write_wid[i]    = fpu_to_csr_if[i].write_wid;
        assign fpu_write_fflags[i] = fpu_to_csr_if[i].write_fflags;
    end
    always @(*) begin
        fcsr_n = fcsr;
        for (integer i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
            if (fpu_write_enable[i]) begin
                fcsr_n[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0] = fcsr[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0]
                                                             | fpu_write_fflags[i];
            end
        end            
        if (write_enable) begin
            case (write_addr)
                `VX_CSR_FFLAGS: fcsr_n[write_wid][`FP_FLAGS_BITS-1:0] = write_data[`FP_FLAGS_BITS-1:0];
                `VX_CSR_FRM:    fcsr_n[write_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] = write_data[`INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:   fcsr_n[write_wid] = write_data[`FP_FLAGS_BITS+`INST_FRM_BITS-1:0];
            default:;
            endcase
        end
    end
    
    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
        assign fpu_to_csr_if[i].read_frm = fcsr[fpu_to_csr_if[i].read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS];
    end

    always @(posedge clk) begin
        if (reset) begin
            fcsr <= '0;
        end else begin
            fcsr <= fcsr_n;
        end
    end
`endif

    always @(posedge clk) begin
        if (write_enable) begin
            case (write_addr)
            `ifdef EXT_F_ENABLE
                `VX_CSR_FFLAGS,
                `VX_CSR_FRM,
                `VX_CSR_FCSR,
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
            `VX_CSR_MISA       : read_data_ro_r = (((`CLOG2(`XLEN)-4) << (`XLEN-2)) | `MISA_STD);
        `ifdef EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_r = 32'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_r = 32'(fcsr[read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_r = 32'(fcsr[read_wid]);
        `endif
            `VX_CSR_WARP_ID    : read_data_ro_r = 32'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_r = 32'(CORE_ID);
            `VX_CSR_THREAD_MASK: read_data_ro_r = 32'(thread_masks[read_wid]);
            `VX_CSR_WARP_MASK  : read_data_ro_r = 32'(active_warps);
            `VX_CSR_NUM_THREADS: read_data_ro_r = 32'(`NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_r = 32'(`NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_r = 32'(`NUM_CORES * `NUM_CLUSTERS);           
            `VX_CSR_MCYCLE     : read_data_ro_r = 32'(cycles[31:0]);
            `VX_CSR_MCYCLE_H   : read_data_ro_r = 32'(cycles[`PERF_CTR_BITS-1:32]);
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
                if ((read_addr >= `VX_CSR_MPM_USER   && read_addr < (`VX_CSR_MPM_USER + 32))
                 || (read_addr >= `VX_CSR_MPM_USER_H && read_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    read_addr_valid_r = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `VX_CSR_MPM_SCHED_ID        : read_data_ro_r = pipeline_perf_if.sched_idles[31:0];
                        `VX_CSR_MPM_SCHED_ID_H      : read_data_ro_r = 32'(pipeline_perf_if.sched_idles[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCHED_ST        : read_data_ro_r = pipeline_perf_if.sched_stalls[31:0];
                        `VX_CSR_MPM_SCHED_ST_H      : read_data_ro_r = 32'(pipeline_perf_if.sched_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_IBUF_ST         : read_data_ro_r = pipeline_perf_if.ibf_stalls[31:0];
                        `VX_CSR_MPM_IBUF_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.ibf_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_ST         : read_data_ro_r = pipeline_perf_if.scb_stalls[31:0];
                        `VX_CSR_MPM_SCRB_ST_H       : read_data_ro_r = 32'(pipeline_perf_if.scb_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_ALU        : read_data_ro_r = pipeline_perf_if.units_uses[`EX_ALU][31:0];
                        `VX_CSR_MPM_SCRB_ALU_H      : read_data_ro_r = 32'(pipeline_perf_if.units_uses[`EX_ALU][`PERF_CTR_BITS-1:32]);
                    `ifdef EXT_F_ENABLE
                        `VX_CSR_MPM_SCRB_FPU        : read_data_ro_r = pipeline_perf_if.units_uses[`EX_FPU][31:0];
                        `VX_CSR_MPM_SCRB_FPU_H      : read_data_ro_r = 32'(pipeline_perf_if.units_uses[`EX_FPU][`PERF_CTR_BITS-1:32]);
                    `else
                        `VX_CSR_MPM_SCRB_FPU        : read_data_ro_r = '0;
                        `VX_CSR_MPM_SCRB_FPU_H      : read_data_ro_r = '0;
                    `endif
                        `VX_CSR_MPM_SCRB_LSU        : read_data_ro_r = pipeline_perf_if.units_uses[`EX_LSU][31:0];
                        `VX_CSR_MPM_SCRB_LSU_H      : read_data_ro_r = 32'(pipeline_perf_if.units_uses[`EX_LSU][`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_SFU        : read_data_ro_r = pipeline_perf_if.units_uses[`EX_SFU][31:0];
                        `VX_CSR_MPM_SCRB_SFU_H      : read_data_ro_r = 32'(pipeline_perf_if.units_uses[`EX_SFU][`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_CSRS       : read_data_ro_r = pipeline_perf_if.sfu_uses[`SFU_CSRS][31:0];
                        `VX_CSR_MPM_SCRB_CSRS_H     : read_data_ro_r = 32'(pipeline_perf_if.sfu_uses[`SFU_CSRS][`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SCRB_WCTL       : read_data_ro_r = pipeline_perf_if.sfu_uses[`SFU_WCTL][31:0];
                        `VX_CSR_MPM_SCRB_WCTL_H     : read_data_ro_r = 32'(pipeline_perf_if.sfu_uses[`SFU_WCTL][`PERF_CTR_BITS-1:32]);
                        // PERF: memory
                        `VX_CSR_MPM_IFETCHES        : read_data_ro_r = pipeline_perf_if.ifetches[31:0];
                        `VX_CSR_MPM_IFETCHES_H      : read_data_ro_r = 32'(pipeline_perf_if.ifetches[`PERF_CTR_BITS-1:32]); 
                        `VX_CSR_MPM_LOADS           : read_data_ro_r = pipeline_perf_if.loads[31:0];
                        `VX_CSR_MPM_LOADS_H         : read_data_ro_r = 32'(pipeline_perf_if.loads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_STORES          : read_data_ro_r = pipeline_perf_if.stores[31:0];
                        `VX_CSR_MPM_STORES_H        : read_data_ro_r = 32'(pipeline_perf_if.stores[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_IFETCH_LT       : read_data_ro_r = pipeline_perf_if.ifetch_latency[31:0];
                        `VX_CSR_MPM_IFETCH_LT_H     : read_data_ro_r = 32'(pipeline_perf_if.ifetch_latency[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_LOAD_LT         : read_data_ro_r = pipeline_perf_if.load_latency[31:0];
                        `VX_CSR_MPM_LOAD_LT_H       : read_data_ro_r = 32'(pipeline_perf_if.load_latency[`PERF_CTR_BITS-1:32]);            
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `VX_CSR_MPM_ICACHE_READS    : read_data_ro_r = mem_perf_if.icache.reads[31:0];
                        `VX_CSR_MPM_ICACHE_READS_H  : read_data_ro_r = 32'(mem_perf_if.icache.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ICACHE_MISS_R   : read_data_ro_r = mem_perf_if.icache.read_misses[31:0];
                        `VX_CSR_MPM_ICACHE_MISS_R_H : read_data_ro_r = 32'(mem_perf_if.icache.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_ICACHE_MSHR_ST  : read_data_ro_r = mem_perf_if.icache.mshr_stalls[31:0];
                        `VX_CSR_MPM_ICACHE_MSHR_ST_H: read_data_ro_r = 32'(mem_perf_if.icache.mshr_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: dcache
                        `VX_CSR_MPM_DCACHE_READS    : read_data_ro_r = mem_perf_if.dcache.reads[31:0];
                        `VX_CSR_MPM_DCACHE_READS_H  : read_data_ro_r = 32'(mem_perf_if.dcache.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_WRITES   : read_data_ro_r = mem_perf_if.dcache.writes[31:0];
                        `VX_CSR_MPM_DCACHE_WRITES_H : read_data_ro_r = 32'(mem_perf_if.dcache.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MISS_R   : read_data_ro_r = mem_perf_if.dcache.read_misses[31:0];
                        `VX_CSR_MPM_DCACHE_MISS_R_H : read_data_ro_r = 32'(mem_perf_if.dcache.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MISS_W   : read_data_ro_r = mem_perf_if.dcache.write_misses[31:0];
                        `VX_CSR_MPM_DCACHE_MISS_W_H : read_data_ro_r = 32'(mem_perf_if.dcache.write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_BANK_ST  : read_data_ro_r = mem_perf_if.dcache.bank_stalls[31:0];
                        `VX_CSR_MPM_DCACHE_BANK_ST_H: read_data_ro_r = 32'(mem_perf_if.dcache.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_DCACHE_MSHR_ST  : read_data_ro_r = mem_perf_if.dcache.mshr_stalls[31:0];
                        `VX_CSR_MPM_DCACHE_MSHR_ST_H: read_data_ro_r = 32'(mem_perf_if.dcache.mshr_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: smem          
                        `VX_CSR_MPM_SMEM_READS      : read_data_ro_r = mem_perf_if.smem.reads[31:0];
                        `VX_CSR_MPM_SMEM_READS_H    : read_data_ro_r = 32'(mem_perf_if.smem.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SMEM_WRITES     : read_data_ro_r = mem_perf_if.smem.writes[31:0];
                        `VX_CSR_MPM_SMEM_WRITES_H   : read_data_ro_r = 32'(mem_perf_if.smem.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_SMEM_BANK_ST    : read_data_ro_r = mem_perf_if.smem.bank_stalls[31:0];
                        `VX_CSR_MPM_SMEM_BANK_ST_H  : read_data_ro_r = 32'(mem_perf_if.smem.bank_stalls[`PERF_CTR_BITS-1:32]);
                        // PERF: l2cache                        
                        `VX_CSR_MPM_L2CACHE_READS   : read_data_ro_r = mem_perf_if.l2cache.reads[31:0];
                        `VX_CSR_MPM_L2CACHE_READS_H : read_data_ro_r = 32'(mem_perf_if.l2cache.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_WRITES  : read_data_ro_r = mem_perf_if.l2cache.writes[31:0];
                        `VX_CSR_MPM_L2CACHE_WRITES_H: read_data_ro_r = 32'(mem_perf_if.l2cache.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MISS_R  : read_data_ro_r = mem_perf_if.l2cache.read_misses[31:0];
                        `VX_CSR_MPM_L2CACHE_MISS_R_H: read_data_ro_r = 32'(mem_perf_if.l2cache.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MISS_W  : read_data_ro_r = mem_perf_if.l2cache.write_misses[31:0];
                        `VX_CSR_MPM_L2CACHE_MISS_W_H: read_data_ro_r = 32'(mem_perf_if.l2cache.write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_BANK_ST : read_data_ro_r = mem_perf_if.l2cache.bank_stalls[31:0];
                        `VX_CSR_MPM_L2CACHE_BANK_ST_H: read_data_ro_r = 32'(mem_perf_if.l2cache.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L2CACHE_MSHR_ST : read_data_ro_r = mem_perf_if.l2cache.mshr_stalls[31:0];
                        `VX_CSR_MPM_L2CACHE_MSHR_ST_H: read_data_ro_r = 32'(mem_perf_if.l2cache.mshr_stalls[`PERF_CTR_BITS-1:32]);      
                        // PERF: l3cache
                        `VX_CSR_MPM_L3CACHE_READS   : read_data_ro_r = mem_perf_if.l3cache.reads[31:0];
                        `VX_CSR_MPM_L3CACHE_READS_H : read_data_ro_r = 32'(mem_perf_if.l3cache.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_WRITES  : read_data_ro_r = mem_perf_if.l3cache.writes[31:0];
                        `VX_CSR_MPM_L3CACHE_WRITES_H: read_data_ro_r = 32'(mem_perf_if.l3cache.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MISS_R  : read_data_ro_r = mem_perf_if.l3cache.read_misses[31:0];
                        `VX_CSR_MPM_L3CACHE_MISS_R_H: read_data_ro_r = 32'(mem_perf_if.l3cache.read_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MISS_W  : read_data_ro_r = mem_perf_if.l3cache.write_misses[31:0];
                        `VX_CSR_MPM_L3CACHE_MISS_W_H: read_data_ro_r = 32'(mem_perf_if.l3cache.write_misses[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_BANK_ST : read_data_ro_r = mem_perf_if.l3cache.bank_stalls[31:0];
                        `VX_CSR_MPM_L3CACHE_BANK_ST_H: read_data_ro_r = 32'(mem_perf_if.l3cache.bank_stalls[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_L3CACHE_MSHR_ST : read_data_ro_r = mem_perf_if.l3cache.mshr_stalls[31:0];
                        `VX_CSR_MPM_L3CACHE_MSHR_ST_H: read_data_ro_r = 32'(mem_perf_if.l3cache.mshr_stalls[`PERF_CTR_BITS-1:32]); 
                        // PERF: memory
                        `VX_CSR_MPM_MEM_READS       : read_data_ro_r = mem_perf_if.mem.reads[31:0];
                        `VX_CSR_MPM_MEM_READS_H     : read_data_ro_r = 32'(mem_perf_if.mem.reads[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_MEM_WRITES      : read_data_ro_r = mem_perf_if.mem.writes[31:0];
                        `VX_CSR_MPM_MEM_WRITES_H    : read_data_ro_r = 32'(mem_perf_if.mem.writes[`PERF_CTR_BITS-1:32]);
                        `VX_CSR_MPM_MEM_LT          : read_data_ro_r = mem_perf_if.mem.latency[31:0];
                        `VX_CSR_MPM_MEM_LT_H        : read_data_ro_r = 32'(mem_perf_if.mem.latency[`PERF_CTR_BITS-1:32]);     
                        default:;
                        endcase
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

`ifdef PERF_ENABLE
    `UNUSED_VAR (mem_perf_if.icache);
    `UNUSED_VAR (mem_perf_if.smem);
`endif

endmodule
