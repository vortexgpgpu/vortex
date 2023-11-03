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

module VX_core import VX_gpu_pkg::*; #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_DECL
    
    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
`endif

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    icache_bus_if,

`ifdef GBAR_ENABLE
    VX_gbar_bus_if.master   gbar_bus_if,
`endif

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire             busy
);
    VX_schedule_if      schedule_if();
    VX_fetch_if         fetch_if();
    VX_decode_if        decode_if();
    VX_sched_csr_if     sched_csr_if();
    VX_decode_sched_if  decode_sched_if();
    VX_commit_sched_if  commit_sched_if();
    VX_commit_csr_if    commit_csr_if();
    VX_branch_ctl_if    branch_ctl_if[`NUM_ALU_BLOCKS]();
    VX_warp_ctl_if      warp_ctl_if();    
    
    VX_dispatch_if      alu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        alu_commit_if[`ISSUE_WIDTH]();

    VX_dispatch_if      lsu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        lsu_commit_if[`ISSUE_WIDTH]();
`ifdef EXT_F_ENABLE 
    VX_dispatch_if      fpu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        fpu_commit_if[`ISSUE_WIDTH]();
`endif
    VX_dispatch_if      sfu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        sfu_commit_if[`ISSUE_WIDTH]();    
    
    VX_writeback_if     writeback_if[`ISSUE_WIDTH]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_bus_tmp_if[DCACHE_NUM_REQS]();

`ifdef PERF_ENABLE
    VX_pipeline_perf_if pipeline_perf_if();
    VX_mem_perf_if mem_perf_tmp_if();

    assign mem_perf_tmp_if.icache = mem_perf_if.icache;
    assign mem_perf_tmp_if.dcache = mem_perf_if.dcache;
    assign mem_perf_tmp_if.l2cache = mem_perf_if.l2cache;
    assign mem_perf_tmp_if.l3cache = mem_perf_if.l3cache;
`ifdef SM_ENABLE
    cache_perf_t smem_perf;
    assign mem_perf_tmp_if.smem = smem_perf;
`else
    assign mem_perf_tmp_if.smem = '0;
`endif
    assign mem_perf_tmp_if.mem = mem_perf_if.mem;
`endif

    `RESET_RELAY (dcr_data_reset, reset);
    `RESET_RELAY (schedule_reset, reset);
    `RESET_RELAY (fetch_reset, reset);
    `RESET_RELAY (decode_reset, reset);
    `RESET_RELAY (issue_reset, reset);
    `RESET_RELAY (execute_reset, reset);
    `RESET_RELAY (commit_reset, reset);

    base_dcrs_t base_dcrs;

    VX_dcr_data dcr_data (
        .clk        (clk),
        .reset      (dcr_data_reset),
        .dcr_bus_if (dcr_bus_if),
        .base_dcrs  (base_dcrs)
    );

    `SCOPE_IO_SWITCH (3)

    VX_schedule #(
        .CORE_ID (CORE_ID)
    ) schedule (
        .clk            (clk),
        .reset          (schedule_reset),   

        .base_dcrs      (base_dcrs),  

        .warp_ctl_if    (warp_ctl_if),        
        .branch_ctl_if  (branch_ctl_if),
        .decode_sched_if(decode_sched_if),
        .commit_sched_if(commit_sched_if),

        .schedule_if    (schedule_if),
    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif
        .sched_csr_if   (sched_csr_if),        

        .busy           (busy)
    );

    VX_fetch #(
        .CORE_ID (CORE_ID)
    ) fetch (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (fetch_reset),
        .icache_bus_if  (icache_bus_if),
        .schedule_if    (schedule_if),
        .fetch_if       (fetch_if)
    );

    VX_decode #(
        .CORE_ID (CORE_ID)
    ) decode (
        .clk            (clk),
        .reset          (decode_reset),
        .fetch_if       (fetch_if),
        .decode_if      (decode_if),
        .decode_sched_if(decode_sched_if)
    );

    VX_issue #(
        .CORE_ID (CORE_ID)
    ) issue (
        `SCOPE_IO_BIND  (1)

        .clk            (clk),
        .reset          (issue_reset),

    `ifdef PERF_ENABLE
        .perf_issue_if  (pipeline_perf_if.issue),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),

        .alu_dispatch_if(alu_dispatch_if),
        .lsu_dispatch_if(lsu_dispatch_if),
    `ifdef EXT_F_ENABLE
        .fpu_dispatch_if(fpu_dispatch_if),
    `endif
        .sfu_dispatch_if(sfu_dispatch_if)
    );

    VX_execute #(
        .CORE_ID (CORE_ID)
    ) execute (
        `SCOPE_IO_BIND  (2)
        
        .clk            (clk),
        .reset          (execute_reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_tmp_if),        
        .pipeline_perf_if(pipeline_perf_if),
    `endif 

        .dcache_bus_if  (dcache_bus_tmp_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_dispatch_if(fpu_dispatch_if),
        .fpu_commit_if  (fpu_commit_if),
    `endif

        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),
        
        .alu_dispatch_if(alu_dispatch_if),
        .lsu_dispatch_if(lsu_dispatch_if),
        .sfu_dispatch_if(sfu_dispatch_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),
        .sfu_commit_if  (sfu_commit_if),

        .sim_ebreak     (sim_ebreak)
    );    

    VX_commit #(
        .CORE_ID (CORE_ID)
    ) commit (
        .clk            (clk),
        .reset          (commit_reset),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),
    `ifdef EXT_F_ENABLE
        .fpu_commit_if  (fpu_commit_if),
    `endif
        .sfu_commit_if  (sfu_commit_if),
        
        .writeback_if   (writeback_if),
        
        .commit_csr_if  (commit_csr_if),
        .commit_sched_if(commit_sched_if),

        .sim_wb_value   (sim_wb_value)
    );

`ifdef SM_ENABLE

    VX_smem_unit #(
        .CORE_ID (CORE_ID)
    ) smem_unit (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .cache_perf         (smem_perf),
    `endif
        .dcache_bus_in_if   (dcache_bus_tmp_if),
        .dcache_bus_out_if  (dcache_bus_if)
    );

`else

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i], dcache_bus_tmp_if[i]);
    end

`endif

`ifdef PERF_ENABLE

    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_rd_req_per_cycle;
    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_wr_req_per_cycle;

    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_rsp_per_cycle;    

    wire perf_icache_pending_read_cycle;
    wire [`CLOG2(DCACHE_NUM_REQS+1)+1-1:0] perf_dcache_pending_read_cycle;

    reg  [`PERF_CTR_BITS-1:0] perf_icache_pending_reads;
    reg  [`PERF_CTR_BITS-1:0] perf_dcache_pending_reads;

    reg  [`PERF_CTR_BITS-1:0] perf_ifetches;
    reg  [`PERF_CTR_BITS-1:0] perf_loads;
    reg  [`PERF_CTR_BITS-1:0] perf_stores;

    wire  perf_icache_req_fire = icache_bus_if.req_valid & icache_bus_if.req_ready;
    wire  perf_icache_rsp_fire = icache_bus_if.rsp_valid & icache_bus_if.rsp_ready;

    wire [DCACHE_NUM_REQS-1:0] perf_dcache_rd_req_fire, perf_dcache_wr_req_fire, perf_dcache_rsp_fire;

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign perf_dcache_rd_req_fire[i] = dcache_bus_if[i].req_valid && ~dcache_bus_if[i].req_data.rw && dcache_bus_if[i].req_ready;
        assign perf_dcache_wr_req_fire[i] = dcache_bus_if[i].req_valid && dcache_bus_if[i].req_data.rw && dcache_bus_if[i].req_ready;
        assign perf_dcache_rsp_fire[i] = dcache_bus_if[i].rsp_valid && dcache_bus_if[i].rsp_ready;
    end

    `POP_COUNT(perf_dcache_rd_req_per_cycle, perf_dcache_rd_req_fire);
    `POP_COUNT(perf_dcache_wr_req_per_cycle, perf_dcache_wr_req_fire);
    `POP_COUNT(perf_dcache_rsp_per_cycle, perf_dcache_rsp_fire);
      
    assign perf_icache_pending_read_cycle = perf_icache_req_fire - perf_icache_rsp_fire;
    assign perf_dcache_pending_read_cycle = perf_dcache_rd_req_per_cycle - perf_dcache_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_icache_pending_reads <= '0;
            perf_dcache_pending_reads <= '0;
        end else begin
            perf_icache_pending_reads <= $signed(perf_icache_pending_reads) + `PERF_CTR_BITS'($signed(perf_icache_pending_read_cycle));
            perf_dcache_pending_reads <= $signed(perf_dcache_pending_reads) + `PERF_CTR_BITS'($signed(perf_dcache_pending_read_cycle));
        end
    end
    
    reg [`PERF_CTR_BITS-1:0] perf_icache_lat;
    reg [`PERF_CTR_BITS-1:0] perf_dcache_lat;

    always @(posedge clk) begin
        if (reset) begin
            perf_ifetches   <= '0;
            perf_loads      <= '0;
            perf_stores     <= '0;
            perf_icache_lat <= '0;
            perf_dcache_lat <= '0;
        end else begin
            perf_ifetches   <= perf_ifetches   + `PERF_CTR_BITS'(perf_icache_req_fire);
            perf_loads      <= perf_loads      + `PERF_CTR_BITS'(perf_dcache_rd_req_per_cycle);
            perf_stores     <= perf_stores     + `PERF_CTR_BITS'(perf_dcache_wr_req_per_cycle);
            perf_icache_lat <= perf_icache_lat + perf_icache_pending_reads;
            perf_dcache_lat <= perf_dcache_lat + perf_dcache_pending_reads;
        end
    end

    assign pipeline_perf_if.ifetches = perf_ifetches;
    assign pipeline_perf_if.loads = perf_loads;
    assign pipeline_perf_if.stores = perf_stores;
    assign pipeline_perf_if.load_latency = perf_dcache_lat;
    assign pipeline_perf_if.ifetch_latency = perf_icache_lat;
    assign pipeline_perf_if.load_latency = perf_dcache_lat;

`endif
    
endmodule

///////////////////////////////////////////////////////////////////////////////

module VX_core_top
import VX_gpu_pkg::*;
#( 
    parameter CORE_ID = 0
) (  
    // Clock
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [`VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [`VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    output wire [DCACHE_NUM_REQS-1:0]       dcache_req_valid,
    output wire [DCACHE_NUM_REQS-1:0]       dcache_req_rw,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE-1:0] dcache_req_byteen,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0] dcache_req_addr,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] dcache_req_data,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_NOSM_TAG_WIDTH-1:0] dcache_req_tag,
    input  wire [DCACHE_NUM_REQS-1:0]       dcache_req_ready,

    input wire  [DCACHE_NUM_REQS-1:0]       dcache_rsp_valid,
    input wire  [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] dcache_rsp_data,
    input wire  [DCACHE_NUM_REQS-1:0][DCACHE_NOSM_TAG_WIDTH-1:0] dcache_rsp_tag,
    output wire [DCACHE_NUM_REQS-1:0]       dcache_rsp_ready,

    output wire                             icache_req_valid,
    output wire                             icache_req_rw,
    output wire [ICACHE_WORD_SIZE-1:0]      icache_req_byteen,
    output wire [ICACHE_ADDR_WIDTH-1:0]     icache_req_addr,
    output wire [ICACHE_WORD_SIZE*8-1:0]    icache_req_data,
    output wire [ICACHE_TAG_WIDTH-1:0]      icache_req_tag,
    input  wire                             icache_req_ready,

    input wire                              icache_rsp_valid,
    input wire  [ICACHE_WORD_SIZE*8-1:0]    icache_rsp_data,
    input wire  [ICACHE_TAG_WIDTH-1:0]      icache_rsp_tag,
    output wire                             icache_rsp_ready,

`ifdef GBAR_ENABLE
    output wire                             gbar_req_valid,
    output wire [`NB_WIDTH-1:0]             gbar_req_id,
    output wire [`NC_WIDTH-1:0]             gbar_req_size_m1,    
    output wire [`NC_WIDTH-1:0]             gbar_req_core_id,
    input wire                              gbar_req_ready,
    input wire                              gbar_rsp_valid,
    input wire [`NB_WIDTH-1:0]              gbar_rsp_id,
`endif

    // simulation helper signals
    output wire                             sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0]  sim_wb_value,

    // Status
    output wire                             busy
);
    
`ifdef GBAR_ENABLE
    VX_gbar_bus_if gbar_bus_if();

    assign gbar_req_valid = gbar_bus_if.req_valid;
    assign gbar_req_id = gbar_bus_if.req_id;
    assign gbar_req_size_m1 = gbar_bus_if.req_size_m1;   
    assign gbar_req_core_id =  gbar_bus_if.req_core_id;
    assign gbar_bus_if.req_ready = gbar_req_ready;
    assign gbar_bus_if.rsp_valid = gbar_rsp_valid;
    assign gbar_bus_if.rsp_id = gbar_rsp_id;
`endif

    VX_dcr_bus_if dcr_bus_if(); 

    assign dcr_bus_if.write_valid = dcr_write_valid;
    assign dcr_bus_if.write_addr = dcr_write_addr;
    assign dcr_bus_if.write_data = dcr_write_data;

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) dcache_bus_if[DCACHE_NUM_REQS]();

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign dcache_req_valid[i] = dcache_bus_if[i].req_valid;
        assign dcache_req_rw[i] = dcache_bus_if[i].req_data.rw;
        assign dcache_req_byteen[i] = dcache_bus_if[i].req_data.byteen;
        assign dcache_req_addr[i] = dcache_bus_if[i].req_data.addr;
        assign dcache_req_data[i] = dcache_bus_if[i].req_data.data;
        assign dcache_req_tag[i] = dcache_bus_if[i].req_data.tag;
        assign dcache_bus_if[i].req_ready = dcache_req_ready[i];

        assign dcache_bus_if[i].rsp_valid = dcache_rsp_valid[i];
        assign dcache_bus_if[i].rsp_data.tag = dcache_rsp_tag[i];
        assign dcache_bus_if[i].rsp_data.data = dcache_rsp_data[i];
        assign dcache_rsp_ready[i] = dcache_bus_if[i].rsp_ready;
    end

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) icache_bus_if();

    assign icache_req_valid = icache_bus_if.req_valid;
    assign icache_req_rw = icache_bus_if.req_data.rw;
    assign icache_req_byteen = icache_bus_if.req_data.byteen;
    assign icache_req_addr = icache_bus_if.req_data.addr;
    assign icache_req_data = icache_bus_if.req_data.data;
    assign icache_req_tag = icache_bus_if.req_data.tag;
    assign icache_bus_if.req_ready = icache_req_ready;

    assign icache_bus_if.rsp_valid = icache_rsp_valid;
    assign icache_bus_if.rsp_data.tag = icache_rsp_tag;
    assign icache_bus_if.rsp_data.data = icache_rsp_data;
    assign icache_rsp_ready = icache_bus_if.rsp_ready;

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_if();
`endif

`ifdef SCOPE
    wire [0:0] scope_reset_w = 1'b0; 
    wire [0:0] scope_bus_in_w = 1'b0; 
    wire [0:0] scope_bus_out_w;
    `UNUSED_VAR (scope_bus_out_w)
`endif

    VX_core #(
        .CORE_ID (0)
    ) core (
        `SCOPE_IO_BIND (0)
        .clk            (clk),
        .reset          (reset),
        
    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_if),
    `endif
        
        .dcr_bus_if     (dcr_bus_if),

        .dcache_bus_if  (dcache_bus_if),

        .icache_bus_if  (icache_bus_if),

    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif

        .sim_ebreak     (sim_ebreak),
        .sim_wb_value   (sim_wb_value),
        .busy           (busy)
    );

endmodule
