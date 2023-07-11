`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_mem_unit # (
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    VX_mem_perf_if.master mem_perf_if,
`endif    

    VX_cache_bus_if.slave   icache_bus_if [`NUM_SOCKETS],

    VX_cache_bus_if.slave   dcache_bus_if [`NUM_SOCKETS],

`ifdef EXT_TEX_ENABLE
`ifdef PERF_ENABLE
    VX_cache_perf_if.master perf_tcache_if,
`endif
    VX_cache_bus_if.slave   tcache_bus_if [`NUM_TEX_UNITS],
`endif

`ifdef EXT_RASTER_ENABLE
`ifdef PERF_ENABLE
    VX_cache_perf_if.master perf_rcache_if,
`endif
    VX_cache_bus_if.slave   rcache_bus_if [`NUM_RASTER_UNITS],
`endif 

`ifdef EXT_ROP_ENABLE
`ifdef PERF_ENABLE
    VX_cache_perf_if.master perf_ocache_if,
`endif
    VX_cache_bus_if.slave   ocache_bus_if [`NUM_ROP_UNITS],
`endif

    VX_mem_bus_if.master    mem_bus_if
);

`ifdef PERF_ENABLE
    VX_cache_perf_if perf_icache_if();
    VX_cache_perf_if perf_dcache_if();
    VX_cache_perf_if perf_smem_if();
    VX_cache_perf_if perf_l2cache_if();
`endif   

/////////////////////////////// I-Cache ///////////////////////////////////

    VX_mem_bus_if #(
        .DATA_WIDTH (ICACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_bus_if();
    
    `RESET_RELAY (icache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-icache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_ICACHES),
        .NUM_INPUTS     (`NUM_SOCKETS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`ICACHE_SIZE),
        .LINE_SIZE      (ICACHE_LINE_SIZE),
        .NUM_BANKS      (`ICACHE_NUM_BANKS),
        .NUM_WAYS       (`ICACHE_NUM_WAYS),
        .WORD_SIZE      (ICACHE_WORD_SIZE),
        .NUM_REQS       (ICACHE_NUM_REQS),
        .CREQ_SIZE      (`ICACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`ICACHE_MREQ_SIZE),
        .TAG_WIDTH      (ICACHE_ARB_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_BITS),
        .WRITE_ENABLE   (0),
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3)
    ) icache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_icache_if),
    `endif
        .clk            (clk),
        .reset          (icache_reset),
        .core_bus_if    (icache_bus_if),
        .mem_bus_if     (icache_mem_bus_if)
    );

/////////////////////////////// D-Cache ///////////////////////////////////

    VX_mem_bus_if #(
        .DATA_WIDTH (DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if();

    VX_cache_bus_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) dcache_switch_bus_if [`NUM_SOCKETS]();

    `RESET_RELAY (dcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-dcache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_DCACHES),
        .NUM_INPUTS     (`NUM_SOCKETS),
        .TAG_SEL_IDX    (1),
        .CACHE_SIZE     (`DCACHE_SIZE),
        .LINE_SIZE      (DCACHE_LINE_SIZE),
        .NUM_BANKS      (`DCACHE_NUM_BANKS),
        .NUM_WAYS       (`DCACHE_NUM_WAYS),
        .NUM_PORTS      (`DCACHE_NUM_PORTS),
        .WORD_SIZE      (DCACHE_WORD_SIZE),
        .NUM_REQS       (DCACHE_NUM_REQS),
        .CREQ_SIZE      (`DCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`DCACHE_MREQ_SIZE),
        .TAG_WIDTH      (DCACHE_NOSM_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_BITS),
        .WRITE_ENABLE   (1),        
        .NC_ENABLE      (1),
        .CORE_OUT_REG   (`SM_ENABLED ? 2 : 3),
        .MEM_OUT_REG    (3)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_dcache_if),
    `endif
        
        .clk            (clk),
        .reset          (dcache_reset),        
        .core_bus_if    (dcache_switch_bus_if),
        .mem_bus_if     (dcache_mem_bus_if)
    );

////////////////////////////// Shared Memory //////////////////////////////

`ifdef SM_ENABLE

    for (genvar i = 0; i < `NUM_SOCKETS; ++i) begin
        VX_cache_bus_if #(
            .NUM_REQS  (DCACHE_NUM_REQS), 
            .WORD_SIZE (DCACHE_WORD_SIZE), 
            .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
        ) smem_switch_out_bus_if[2]();

        `RESET_RELAY (dcache_smem_switch_reset, reset);

        VX_smem_switch #(
            .NUM_REQS     (2),
            .NUM_LANES    (DCACHE_NUM_REQS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_ARB_TAG_WIDTH),
            .TAG_SEL_IDX  (0),
            .ARBITER      ("P"),
            .BUFFERED_REQ (2),
            .BUFFERED_RSP (2)
        ) smem_switch (
            .clk        (clk),
            .reset      (dcache_smem_switch_reset),
            .bus_in_if  (dcache_bus_if[i]),
            .bus_out_if (smem_switch_out_bus_if)
        );

        `ASSIGN_VX_CACHE_BUS_IF (dcache_switch_bus_if[i], smem_switch_out_bus_if[0]);

        // shared memory address mapping:
        // [core_idx][warp_idx][word_idx][thread_idx] <= [core_idx][warp_idx][thread_idx][bank_offset..word_idx]
        localparam SOCKET_BITS      = `CLOG2(`SOCKET_SIZE);
        localparam SMEM_SIZE        = `SOCKET_SIZE * `NUM_WARPS * `NUM_THREADS * `SMEM_LOCAL_SIZE;
        localparam BANK_ADDR_OFFSET = `CLOG2(`STACK_SIZE / DCACHE_WORD_SIZE);
        localparam WORD_SEL_BITS    = `CLOG2(`SMEM_LOCAL_SIZE / DCACHE_WORD_SIZE);
        localparam SMEM_ADDR_WIDTH  = SOCKET_BITS + `NW_BITS + `NT_BITS + WORD_SEL_BITS;

        wire [DCACHE_NUM_REQS-1:0][SMEM_ADDR_WIDTH-1:0] smem_req_addr;

        for (genvar j = 0; j < DCACHE_NUM_REQS; ++j) begin
            if (`NT_BITS != 0) begin
                assign smem_req_addr[j][0 +: `NT_BITS] = smem_switch_out_bus_if[1].req_addr[j][BANK_ADDR_OFFSET +: `NT_BITS];
            end    
            assign smem_req_addr[j][`NT_BITS +: WORD_SEL_BITS] = smem_switch_out_bus_if[1].req_addr[j][0 +: WORD_SEL_BITS];
            if (`NW_BITS != 0) begin
                assign smem_req_addr[j][(`NT_BITS + WORD_SEL_BITS) +: `NW_BITS] = smem_switch_out_bus_if[1].req_addr[j][(BANK_ADDR_OFFSET + `NT_BITS) +: `NW_BITS];
            end
            if (SOCKET_BITS != 0) begin
                assign smem_req_addr[j][(`NT_BITS + WORD_SEL_BITS + `NW_BITS) +: SOCKET_BITS] = smem_switch_out_bus_if[1].req_addr[j][(BANK_ADDR_OFFSET + `NT_BITS + `NW_BITS) +: SOCKET_BITS];
            end
        end

        `RESET_RELAY (smem_reset, reset);
        
        VX_shared_mem #(
            .INSTANCE_ID($sformatf("cluster%0d-smem%0d", CLUSTER_ID, i)),
            .SIZE       (SMEM_SIZE),
            .NUM_REQS   (DCACHE_NUM_REQS),
            .NUM_BANKS  (`SMEM_NUM_BANKS),
            .WORD_SIZE  (DCACHE_WORD_SIZE),
            .ADDR_WIDTH (SMEM_ADDR_WIDTH),
            .UUID_WIDTH (`UUID_BITS), 
            .TAG_WIDTH  (DCACHE_NOSM_TAG_WIDTH),
            .OUT_REG    (2)
        ) shared_mem (        
            .clk        (clk),
            .reset      (smem_reset),

        `ifdef PERF_ENABLE
            .cache_perf_if(perf_smem_if),
        `endif

            // Core request
            .req_valid  (smem_switch_out_bus_if[1].req_valid),
            .req_rw     (smem_switch_out_bus_if[1].req_rw),
            .req_byteen (smem_switch_out_bus_if[1].req_byteen),
            .req_addr   (smem_req_addr),
            .req_data   (smem_switch_out_bus_if[1].req_data),        
            .req_tag    (smem_switch_out_bus_if[1].req_tag),
            .req_ready  (smem_switch_out_bus_if[1].req_ready),

            // Core response
            .rsp_valid  (smem_switch_out_bus_if[1].rsp_valid),
            .rsp_data   (smem_switch_out_bus_if[1].rsp_data),
            .rsp_tag    (smem_switch_out_bus_if[1].rsp_tag),
            .rsp_ready  (smem_switch_out_bus_if[1].rsp_ready)
        ); 
    end   

`else

    for (genvar i = 0; i < `NUM_SOCKETS; ++i) begin
        `ASSIGN_VX_CACHE_BUS_IF (dcache_switch_bus_if[i], dcache_bus_if[i]);
    end

`endif

/////////////////////////////// T-Cache ///////////////////////////////////

`ifdef EXT_TEX_ENABLE    

    VX_mem_bus_if #(
        .DATA_WIDTH (TCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_if();

    `RESET_RELAY (tcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-tcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_TCACHES),
        .NUM_INPUTS     (`NUM_TEX_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .NUM_PORTS      (`TCACHE_NUM_PORTS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .CREQ_SIZE      (`TCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (`UUID_BITS),
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3)
    ) tcache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_tcache_if),
    `endif        
        .clk            (clk),
        .reset          (tcache_reset),
        .core_bus_if    (tcache_bus_if),
        .mem_bus_if     (tcache_mem_bus_if)
    );

`endif

/////////////////////////////// O-Cache ///////////////////////////////////

`ifdef EXT_ROP_ENABLE    

    VX_mem_bus_if #(
        .DATA_WIDTH (OCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_bus_if();

    `RESET_RELAY (ocache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-ocache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_OCACHES),
        .NUM_INPUTS     (`NUM_ROP_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`OCACHE_SIZE),
        .LINE_SIZE      (OCACHE_LINE_SIZE),
        .NUM_BANKS      (`OCACHE_NUM_BANKS),
        .NUM_WAYS       (`OCACHE_NUM_WAYS),
        .NUM_PORTS      (`OCACHE_NUM_PORTS),
        .WORD_SIZE      (OCACHE_WORD_SIZE),
        .NUM_REQS       (OCACHE_NUM_REQS),
        .CREQ_SIZE      (`OCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`OCACHE_MREQ_SIZE),
        .TAG_WIDTH      (OCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_BITS),
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3)
    ) ocache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_ocache_if),
    `endif        
        .clk            (clk),
        .reset          (ocache_reset),

        .core_bus_if    (ocache_bus_if),
        .mem_bus_if     (ocache_mem_bus_if)
    );

`endif

/////////////////////////////// R-Cache ///////////////////////////////////

`ifdef EXT_RASTER_ENABLE

    VX_mem_bus_if #(
        .DATA_WIDTH (RCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_bus_if();

    `RESET_RELAY (rcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-rcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_RCACHES),
        .NUM_INPUTS     (`NUM_RASTER_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`RCACHE_SIZE),
        .LINE_SIZE      (RCACHE_LINE_SIZE),
        .NUM_BANKS      (`RCACHE_NUM_BANKS),
        .NUM_WAYS       (`RCACHE_NUM_WAYS),
        .NUM_PORTS      (`RCACHE_NUM_PORTS),
        .WORD_SIZE      (RCACHE_WORD_SIZE),
        .NUM_REQS       (RCACHE_NUM_REQS),
        .CREQ_SIZE      (`RCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`RCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (0),        
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3)
    ) rcache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_rcache_if),
    `endif        
        .clk            (clk),
        .reset          (rcache_reset),
        .core_bus_if    (rcache_bus_if),
        .mem_bus_if     (rcache_mem_bus_if)
    );

`endif

/////////////////////////////// L2-Cache //////////////////////////////////

    VX_mem_bus_if #(
        .DATA_WIDTH (L2_WORD_SIZE * 8),
        .TAG_WIDTH  (L2_TAG_WIDTH)
    ) l2_mem_bus_if[L2_NUM_REQS]();

    localparam I_MEM_ARB_IDX = 0;
    localparam D_MEM_ARB_IDX = I_MEM_ARB_IDX + 1;
    localparam T_MEM_ARB_IDX = D_MEM_ARB_IDX + 1;
    localparam R_MEM_ARB_IDX = T_MEM_ARB_IDX + `EXT_TEX_ENABLED;
    localparam O_MEM_ARB_IDX = R_MEM_ARB_IDX + `EXT_RASTER_ENABLED;
    `UNUSED_PARAM (T_MEM_ARB_IDX)
    `UNUSED_PARAM (R_MEM_ARB_IDX)
    `UNUSED_PARAM (O_MEM_ARB_IDX)

    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[I_MEM_ARB_IDX], icache_mem_bus_if, L1_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH);
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[D_MEM_ARB_IDX], dcache_mem_bus_if, L1_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH);

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[T_MEM_ARB_IDX], tcache_mem_bus_if, L1_MEM_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH);
`endif

`ifdef EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[R_MEM_ARB_IDX], rcache_mem_bus_if, L1_MEM_TAG_WIDTH, RCACHE_MEM_TAG_WIDTH);
`endif

`ifdef EXT_ROP_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[O_MEM_ARB_IDX], ocache_mem_bus_if, L1_MEM_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH);
`endif

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-l2cache", CLUSTER_ID)),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .NUM_PORTS      (`L2_NUM_PORTS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .CREQ_SIZE      (`L2_CREQ_SIZE),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_MREQ_SIZE),
        .TAG_WIDTH      (L1_MEM_TAG_WIDTH),
        .WRITE_ENABLE   (1),       
        .UUID_WIDTH     (`UUID_BITS),
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (            
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_l2cache_if),
    `endif
        .core_bus_if    (l2_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

`ifdef PERF_ENABLE
    
    `UNUSED_VAR (perf_dcache_if.mem_stalls)
    `UNUSED_VAR (perf_dcache_if.crsp_stalls)

    assign mem_perf_if.icache_reads       = perf_icache_if.reads;
    assign mem_perf_if.icache_read_misses = perf_icache_if.read_misses;
    
    assign mem_perf_if.dcache_reads       = perf_dcache_if.reads;
    assign mem_perf_if.dcache_writes      = perf_dcache_if.writes;
    assign mem_perf_if.dcache_read_misses = perf_dcache_if.read_misses;
    assign mem_perf_if.dcache_write_misses= perf_dcache_if.write_misses;
    assign mem_perf_if.dcache_bank_stalls = perf_dcache_if.bank_stalls;
    assign mem_perf_if.dcache_mshr_stalls = perf_dcache_if.mshr_stalls;

`ifdef SM_ENABLE
    assign mem_perf_if.smem_reads         = perf_smem_if.reads;
    assign mem_perf_if.smem_writes        = perf_smem_if.writes;
    assign mem_perf_if.smem_bank_stalls   = perf_smem_if.bank_stalls;    
`else
    assign mem_perf_if.smem_reads         = '0;
    assign mem_perf_if.smem_writes        = '0;
    assign mem_perf_if.smem_bank_stalls   = '0;
`endif

`ifdef L2_ENABLE
    assign mem_perf_if.l2cache_reads       = perf_l2cache_if.reads;
    assign mem_perf_if.l2cache_writes      = perf_l2cache_if.writes;
    assign mem_perf_if.l2cache_read_misses = perf_l2cache_if.read_misses;
    assign mem_perf_if.l2cache_write_misses= perf_l2cache_if.write_misses;
    assign mem_perf_if.l2cache_bank_stalls = perf_l2cache_if.bank_stalls;
    assign mem_perf_if.l2cache_mshr_stalls = perf_l2cache_if.mshr_stalls;
`else
    assign mem_perf_if.l2cache_reads       = '0;
    assign mem_perf_if.l2cache_writes      = '0;
    assign mem_perf_if.l2cache_read_misses = '0;
    assign mem_perf_if.l2cache_write_misses= '0;
    assign mem_perf_if.l2cache_bank_stalls = '0;
    assign mem_perf_if.l2cache_mshr_stalls = '0;
`endif
    
    assign mem_perf_if.l3cache_reads       = '0;
    assign mem_perf_if.l3cache_writes      = '0;
    assign mem_perf_if.l3cache_read_misses = '0;
    assign mem_perf_if.l3cache_write_misses= '0;
    assign mem_perf_if.l3cache_bank_stalls = '0;
    assign mem_perf_if.l3cache_mshr_stalls = '0;

    assign mem_perf_if.mem_reads   = '0;       
    assign mem_perf_if.mem_writes  = '0;
    assign mem_perf_if.mem_latency = '0;
    
`endif
    
endmodule
