`include "VX_define.vh"
`include "VX_cache_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_cache_types::*;
`IGNORE_WARNINGS_END

module VX_mem_unit # (
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    VX_perf_memsys_if.master perf_memsys_if,
`endif    

    VX_cache_req_if.slave   icache_req_if [`NUM_CORES],  
    VX_cache_rsp_if.master  icache_rsp_if [`NUM_CORES],

    VX_cache_req_if.slave   dcache_req_if [`NUM_CORES],
    VX_cache_rsp_if.master  dcache_rsp_if [`NUM_CORES],

`ifdef EXT_TEX_ENABLE
    VX_cache_req_if.slave   tcache_req_if [`NUM_TEX_UNITS],
    VX_cache_rsp_if.master  tcache_rsp_if [`NUM_TEX_UNITS],
`endif

`ifdef EXT_RASTER_ENABLE
    VX_cache_req_if.slave   rcache_req_if [`NUM_RASTER_UNITS],
    VX_cache_rsp_if.master  rcache_rsp_if [`NUM_RASTER_UNITS],
`endif 

`ifdef EXT_ROP_ENABLE
    VX_cache_req_if.slave   ocache_req_if [`NUM_ROP_UNITS],
    VX_cache_rsp_if.master  ocache_rsp_if [`NUM_ROP_UNITS],
`endif

    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if
);
    
`ifdef PERF_ENABLE
    VX_perf_cache_if perf_icache_if[`NUM_ICACHE]();
    VX_perf_cache_if perf_dcache_if[`NUM_DCACHE]();
    VX_perf_cache_if perf_smem_if();
`endif   

    /////////////////////////////// I-Cache ///////////////////////////////////

    VX_mem_req_if #(
        .DATA_WIDTH (ICACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (ICACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_rsp_if();
    
    `RESET_RELAY (icache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster-icache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_ICACHES),
        .NUM_INPUTS     (`NUM_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`ICACHE_SIZE),
        .LINE_SIZE      (ICACHE_LINE_SIZE),
        .NUM_BANKS      (ICACHE_NUM_BANKS),
        .NUM_WAYS       (`ICACHE_NUM_WAYS),
        .WORD_SIZE      (ICACHE_WORD_SIZE),
        .NUM_REQS       (ICACHE_NUM_REQS),
        .CREQ_SIZE      (`ICACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`ICACHE_MREQ_SIZE),
        .TAG_WIDTH      (ICACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (`UUID_BITS)        
    ) icache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_icache_if),
    `endif
        .clk            (clk),
        .reset          (icache_reset),
        .core_req_if    (icache_req_if),
        .core_rsp_if    (icache_rsp_if),
        .mem_req_if     (icache_mem_req_if),
        .mem_rsp_if     (icache_mem_rsp_if)
    );

    /////////////////////////////// D-Cache ///////////////////////////////////

    VX_mem_req_if #(
        .DATA_WIDTH (DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_rsp_if();

    VX_cache_req_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_req_if [`NUM_CORES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_rsp_if [`NUM_CORES]();

    `RESET_RELAY (dcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster-dcache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_DCACHES),
        .NUM_INPUTS     (`NUM_CORES),
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
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_BITS),        
        .NC_ENABLE      (1),
        .NC_TAG_BIT     (0)
    ) dcache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_dcache_if),
    `endif
        
        .clk            (clk),
        .reset          (dcache_reset),        
        .core_req_if    (dcache_nosm_req_if),
        .core_rsp_if    (dcache_nosm_rsp_if),
        .mem_req_if     (dcache_mem_req_if),
        .mem_rsp_if     (dcache_mem_rsp_if)
    );

    ////////////////////////////// Shared Memory //////////////////////////////

`ifdef SM_ENABLE

    VX_cache_req_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) per_core_smem_req_if [`NUM_CORES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) per_core_smem_rsp_if [`NUM_CORES]();

    for (genvar i = 0; i < `NUM_CORES; ++i) begin
        VX_cache_req_if #(
            .NUM_REQS  (DCACHE_NUM_REQS), 
            .WORD_SIZE (DCACHE_WORD_SIZE), 
            .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
        ) dcache_nosm_switch_req_if[2]();

        VX_cache_rsp_if #(
            .NUM_REQS  (DCACHE_NUM_REQS), 
            .WORD_SIZE (DCACHE_WORD_SIZE), 
            .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
        ) dcache_nosm_switch_rsp_if[2]();

        VX_smem_switch #(
            .NUM_REQS     (2),
            .NUM_LANES    (DCACHE_NUM_REQS),
            .DATA_SIZE    (4),            
            .TAG_WIDTH    (DCACHE_TAG_WIDTH),
            .TAG_SEL_IDX  (0),
            .ARBITER      ("P"),
            .BUFFERED_REQ (2),
            .BUFFERED_RSP (1)
        ) dcache_nosm_switch (
            .clk        (clk),
            .reset      (reset),
            .req_in_if  (dcache_req_if[i]),
            .rsp_in_if  (dcache_rsp_if[i]),
            .req_out_if (dcache_nosm_switch_req_if),
            .rsp_out_if (dcache_nosm_switch_rsp_if)
        );

        `ASSIGN_VX_CACHE_REQ_IF (dcache_nosm_req_if[i], dcache_nosm_switch_req_if[0]);
        `ASSIGN_VX_CACHE_RSP_IF (dcache_nosm_switch_rsp_if[0], dcache_nosm_rsp_if[i]);
        `ASSIGN_VX_CACHE_REQ_IF (per_core_smem_req_if[i], dcache_nosm_switch_req_if[1]);
        `ASSIGN_VX_CACHE_RSP_IF (dcache_nosm_switch_rsp_if[1], per_core_smem_rsp_if[i]);
    end

    localparam DCACHE_SM_TAG_WIDTH = DCACHE_NOSM_TAG_WIDTH + `NC_BITS;

    VX_cache_req_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_SM_TAG_WIDTH)
    ) smem_req_if[1]();

    VX_cache_rsp_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_SM_TAG_WIDTH)
    ) smem_rsp_if[1](); 

    VX_cache_arb #(
        .NUM_INPUTS   (`NUM_CORES),
        .NUM_LANES    (DCACHE_NUM_REQS),
        .DATA_SIZE    (DCACHE_WORD_SIZE),
        .TAG_WIDTH    (DCACHE_NOSM_TAG_WIDTH),
        .TAG_SEL_IDX  (0),
        .ARBITER      ((`NUM_CORES > 8) ? "C" : "R"),
        .BUFFERED_REQ ((`NUM_CORES != 1) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_CORES != 1) ? 1 : 0)        
    ) smem_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_smem_req_if),
        .rsp_in_if  (per_core_smem_rsp_if),
        .req_out_if (smem_req_if),
        .rsp_out_if (smem_rsp_if)
    );

    // shared memory address mapping:  
    // [core_idx][warp_idx][word_idx][thread_idx] <= [core_idx][warp_idx][thread_idx][bank_offset..word_idx]
    localparam BANK_ADDR_OFFSET = `CLOG2(`STACK_SIZE / DCACHE_WORD_SIZE);
    localparam WORD_SEL_BITS    = `CLOG2(`SMEM_LOCAL_SIZE / DCACHE_WORD_SIZE);
    localparam SMEM_ADDR_WIDTH  = (`NC_BITS + `NW_BITS + `NT_BITS + WORD_SEL_BITS);

    wire [DCACHE_NUM_REQS-1:0][SMEM_ADDR_WIDTH-1:0] smem_req_addr;        
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        if (`NT_BITS != 0) begin
            assign smem_req_addr[i][0 +: `NT_BITS] = smem_req_if[0].addr[i][BANK_ADDR_OFFSET +: `NT_BITS];
        end        
        assign smem_req_addr[i][`NT_BITS +: WORD_SEL_BITS] = smem_req_if[0].addr[i][0 +: WORD_SEL_BITS];
        if (`NW_BITS != 0) begin
            assign smem_req_addr[i][(`NT_BITS + WORD_SEL_BITS) +: `NW_BITS] = smem_req_if[0].addr[i][(BANK_ADDR_OFFSET + `NT_BITS) +: `NW_BITS];
        end
        if (`NC_BITS != 0) begin
            assign smem_req_addr[i][(`NT_BITS + WORD_SEL_BITS + `NW_BITS) +: `NC_BITS] = smem_req_if[0].addr[i][(BANK_ADDR_OFFSET + `NT_BITS + `NW_BITS) +: `NC_BITS];
        end
    end

    `RESET_RELAY (smem_reset, reset);
    
    VX_shared_mem #(
        .IDNAME     ($sformatf("cluster%0d-smem", CLUSTER_ID)),
        .SIZE       (`SMEM_SIZE),
        .NUM_REQS   (DCACHE_NUM_REQS),
        .NUM_BANKS  (`SMEM_NUM_BANKS),
        .WORD_SIZE  (DCACHE_WORD_SIZE),
        .ADDR_WIDTH (SMEM_ADDR_WIDTH),
        .REQ_SIZE   (`SMEM_CREQ_SIZE),
        .OUT_REG    (2),
        .UUID_WIDTH (`UUID_BITS), 
        .TAG_WIDTH  (DCACHE_SM_TAG_WIDTH)
    ) smem (            
        .clk        (clk),
        .reset      (smem_reset),

    `ifdef PERF_ENABLE
        .perf_cache_if(perf_smem_if),
    `endif

        // Core request
        .req_valid  (smem_req_if[0].valid),
        .req_rw     (smem_req_if[0].rw),
        .req_byteen (smem_req_if[0].byteen),
        .req_addr   (smem_req_addr),
        .req_data   (smem_req_if[0].data),        
        .req_tag    (smem_req_if[0].tag),
        .req_ready  (smem_req_if[0].ready),

        // Core response
        .rsp_valid  (smem_rsp_if[0].valid),
        .rsp_data   (smem_rsp_if[0].data),
        .rsp_tag    (smem_rsp_if[0].tag),
        .rsp_ready  (smem_rsp_if[0].ready)
    );    

`else

    for (genvar i = 0; i < `NUM_CORES; ++i) begin
        `ASSIGN_VX_CACHE_REQ_IF (dcache_nosm_req_if[i], dcache_req_if[i]);
        `ASSIGN_VX_CACHE_RSP_IF (dcache_rsp_if[i], dcache_nosm_rsp_if[i]);
    end

`endif

    /////////////////////////////// T-Cache ///////////////////////////////////

`ifdef EXT_TEX_ENABLE    

    VX_mem_req_if #(
        .DATA_WIDTH (TCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (TCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_rsp_if();

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
        .UUID_WIDTH     (0),        
        .NC_ENABLE      (0)
    ) tcache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_tcache_if),
    `endif        
        .clk            (clk),
        .reset          (tcache_reset),
        .core_req_if    (tcache_req_if),
        .core_rsp_if    (tcache_rsp_if),
        .mem_req_if     (tcache_mem_req_if),
        .mem_rsp_if     (tcache_mem_rsp_if)
    );

`endif

    /////////////////////////////// O-Cache ///////////////////////////////////

`ifdef EXT_ROP_ENABLE    

    VX_mem_req_if #(
        .DATA_WIDTH (OCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (OCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_rsp_if();

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
        .UUID_WIDTH     (0),        
        .NC_ENABLE      (0)
    ) ocache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_ocache_if),
    `endif        
        .clk            (clk),
        .reset          (ocache_reset),

        .core_req_if    (ocache_req_if),
        .core_rsp_if    (ocache_rsp_if),
        .mem_req_if     (ocache_mem_req_if),
        .mem_rsp_if     (ocache_mem_rsp_if)
    );

`endif

    /////////////////////////////// R-Cache ///////////////////////////////////

`ifdef EXT_RASTER_ENABLE

    VX_mem_req_if #(
        .DATA_WIDTH (RCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (RCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_rsp_if();

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
        .NC_ENABLE      (0)
    ) rcache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_rcache_if),
    `endif        
        .clk            (clk),
        .reset          (rcache_reset),
        .core_req_if    (rcache_req_if),
        .core_rsp_if    (rcache_rsp_if),
        .mem_req_if     (rcache_mem_req_if),
        .mem_rsp_if     (rcache_mem_rsp_if)
    );

`endif

    /////////////////////////////// L2-Cache //////////////////////////////////

    VX_mem_req_if #(
        .DATA_WIDTH (L2_WORD_SIZE * 8),
        .TAG_WIDTH  (L2_TAG_WIDTH)
    ) l2_mem_req_if[L2_NUM_REQS]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (L2_WORD_SIZE * 8),
        .TAG_WIDTH  (L2_TAG_WIDTH)
    ) l2_mem_rsp_if[L2_NUM_REQS]();

    localparam I_MEM_ARB_IDX = 0;
    localparam D_MEM_ARB_IDX = I_MEM_ARB_IDX + 1;
    localparam T_MEM_ARB_IDX = D_MEM_ARB_IDX + 1;
    localparam R_MEM_ARB_IDX = T_MEM_ARB_IDX + `EXT_TEX_ENABLED;
    localparam O_MEM_ARB_IDX = R_MEM_ARB_IDX + `EXT_RASTER_ENABLED;
    `UNUSED_PARAM (T_MEM_ARB_IDX)
    `UNUSED_PARAM (R_MEM_ARB_IDX)
    `UNUSED_PARAM (O_MEM_ARB_IDX)

    `ASSIGN_VX_MEM_REQ_IF_XTAG (l2_mem_req_if[I_MEM_ARB_IDX], icache_mem_req_if);
    assign l2_mem_req_if[I_MEM_ARB_IDX].tag = L1_MEM_TAG_WIDTH'(icache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (icache_mem_rsp_if, l2_mem_rsp_if[I_MEM_ARB_IDX]);
    assign icache_mem_rsp_if.tag = ICACHE_MEM_TAG_WIDTH'(l2_mem_rsp_if[I_MEM_ARB_IDX].tag);

    `ASSIGN_VX_MEM_REQ_IF_XTAG (l2_mem_req_if[D_MEM_ARB_IDX], dcache_mem_req_if);
    assign l2_mem_req_if[D_MEM_ARB_IDX].tag = L1_MEM_TAG_WIDTH'(dcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (dcache_mem_rsp_if, l2_mem_rsp_if[D_MEM_ARB_IDX]);
    assign dcache_mem_rsp_if.tag = DCACHE_MEM_TAG_WIDTH'(l2_mem_rsp_if[D_MEM_ARB_IDX].tag);

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (l2_mem_req_if[T_MEM_ARB_IDX], tcache_mem_req_if);
    assign l2_mem_req_if[T_MEM_ARB_IDX].tag = L1_MEM_TAG_WIDTH'(tcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (tcache_mem_rsp_if, l2_mem_rsp_if[T_MEM_ARB_IDX]);
    assign tcache_mem_rsp_if.tag = TCACHE_MEM_TAG_WIDTH'(l2_mem_rsp_if[T_MEM_ARB_IDX].tag);
`endif

`ifdef EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (l2_mem_req_if[R_MEM_ARB_IDX], rcache_mem_req_if);
    assign l2_mem_req_if[R_MEM_ARB_IDX].tag = L1_MEM_TAG_WIDTH'(rcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (rcache_mem_rsp_if, l2_mem_rsp_if[R_MEM_ARB_IDX]);
    assign rcache_mem_rsp_if.tag = RCACHE_MEM_TAG_WIDTH'(l2_mem_rsp_if[R_MEM_ARB_IDX].tag);
`endif

`ifdef EXT_ROP_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (l2_mem_req_if[O_MEM_ARB_IDX], ocache_mem_req_if);
    assign l2_mem_req_if[O_MEM_ARB_IDX].tag = L1_MEM_TAG_WIDTH'(ocache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (ocache_mem_rsp_if, l2_mem_rsp_if[O_MEM_ARB_IDX]);
    assign ocache_mem_rsp_if.tag = OCACHE_MEM_TAG_WIDTH'(l2_mem_rsp_if[O_MEM_ARB_IDX].tag);
`endif     

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_l2cache_if();
`endif

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-l2cache", CLUSTER_ID)),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (L2_LINE_SIZE),
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
        .MEM_OUT_REG    (2),   
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache_wrap (            
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_l2cache_if),
    `endif
        .core_req_if    (l2_mem_req_if),
        .core_rsp_if    (l2_mem_rsp_if),
        .mem_req_if     (mem_req_if),
        .mem_rsp_if     (mem_rsp_if)
    );

`ifdef PERF_ENABLE
    
    `UNUSED_VAR (perf_dcache_if.mem_stalls)
    `UNUSED_VAR (perf_dcache_if.crsp_stalls)

    assign perf_memsys_if.icache_reads       = perf_icache_if.reads;
    assign perf_memsys_if.icache_read_misses = perf_icache_if.read_misses;
    assign perf_memsys_if.dcache_reads       = perf_dcache_if.reads;
    assign perf_memsys_if.dcache_writes      = perf_dcache_if.writes;
    assign perf_memsys_if.dcache_read_misses = perf_dcache_if.read_misses;
    assign perf_memsys_if.dcache_write_misses= perf_dcache_if.write_misses;
    assign perf_memsys_if.dcache_bank_stalls = perf_dcache_if.bank_stalls;
    assign perf_memsys_if.dcache_mshr_stalls = perf_dcache_if.mshr_stalls;    

`ifdef SM_ENABLE
    assign perf_memsys_if.smem_reads         = perf_smem_if.reads;
    assign perf_memsys_if.smem_writes        = perf_smem_if.writes;
    assign perf_memsys_if.smem_bank_stalls   = perf_smem_if.bank_stalls;    
`else
    assign perf_memsys_if.smem_reads         = 0;
    assign perf_memsys_if.smem_writes        = 0;
    assign perf_memsys_if.smem_bank_stalls   = 0;
`endif

    reg [`PERF_CTR_BITS-1:0] perf_mem_pending_reads;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= 0;
        end else begin
            perf_mem_pending_reads <= perf_mem_pending_reads + 
                `PERF_CTR_BITS'($signed(2'((mem_req_if.valid && mem_req_if.ready && !mem_req_if.rw) && !(mem_rsp_if.valid && mem_rsp_if.ready)) - 
                    2'((mem_rsp_if.valid && mem_rsp_if.ready) && !(mem_req_if.valid && mem_req_if.ready && !mem_req_if.rw))));
        end
    end
    
    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_lat;

    always @(posedge clk) begin
        if (reset) begin       
            perf_mem_reads  <= 0;     
            perf_mem_writes <= 0;            
            perf_mem_lat    <= 0;
        end else begin  
            if (mem_req_if.valid && mem_req_if.ready && !mem_req_if.rw) begin
                perf_mem_reads <= perf_mem_reads + `PERF_CTR_BITS'd1;
            end
            if (mem_req_if.valid && mem_req_if.ready && mem_req_if.rw) begin
                perf_mem_writes <= perf_mem_writes + `PERF_CTR_BITS'd1;
            end      
            perf_mem_lat <= perf_mem_lat + perf_mem_pending_reads;
        end
    end

    assign perf_memsys_if.mem_reads   = perf_mem_reads;       
    assign perf_memsys_if.mem_writes  = perf_mem_writes;
    assign perf_memsys_if.mem_latency = perf_mem_lat;
    
`endif
    
endmodule
