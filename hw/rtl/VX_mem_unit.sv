`include "VX_define.vh"

module VX_mem_unit # (
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_mem_unit

    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    VX_perf_memsys_if.master perf_memsys_if,
`endif    

    // icache interface
    VX_cache_req_if.slave   icache_req_if,  
    VX_cache_rsp_if.master  icache_rsp_if,


    // dcache interface
    VX_cache_req_if.slave   dcache_req_if,
    VX_cache_rsp_if.master  dcache_rsp_if,    

`ifdef EXT_TEX_ENABLE
    // tcache interface
    VX_cache_req_if.slave   tcache_req_if,
    VX_cache_rsp_if.master  tcache_rsp_if,
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_tcache_if,
`endif
`endif

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if
);
    
`ifdef PERF_ENABLE
    VX_perf_cache_if perf_icache_if(), perf_dcache_if(), perf_smem_if();
`endif

    ///////////////////////////////////////////////////////////////////////////
    
    VX_mem_req_if #(
        .DATA_WIDTH (`ICACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`ICACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`ICACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_rsp_if();

    VX_mem_req_if #(
        .DATA_WIDTH (`ICACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`ICACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`ICACHE_TAG_WIDTH)
    ) icache_req_qual_if[`ICACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`ICACHE_WORD_SIZE*8), 
        .TAG_WIDTH  (`ICACHE_TAG_WIDTH)
    ) icache_rsp_qual_if[`ICACHE_NUM_REQS]();

    for (genvar i = 0; i < `ICACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM (icache_req_qual_if, icache_req_if, i);
    end

    `RESET_RELAY (icache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`ICACHE_ID),
        .CACHE_SIZE     (`ICACHE_SIZE),
        .CACHE_LINE_SIZE(`ICACHE_LINE_SIZE),
        .NUM_BANKS      (1),
        .NUM_WAYS       (`ICACHE_NUM_WAYS),
        .WORD_SIZE      (`ICACHE_WORD_SIZE),
        .NUM_REQS       (1),
        .CREQ_SIZE      (`ICACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`ICACHE_MREQ_SIZE),
        .WRITE_ENABLE   (0),
        .REQ_UUID_BITS  (`UUID_BITS),
        .CORE_TAG_WIDTH (`ICACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`ICACHE_MEM_TAG_WIDTH),
        .PASSTHRU       (!`ICACHE_ENABLED)
    ) icache (
        `SCOPE_BIND_VX_mem_unit_icache

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_icache_if),
    `endif

        .clk            (clk),
        .reset          (icache_reset),
        .core_req_if    (icache_req_qual_if),
        .core_rsp_if    (icache_rsp_qual_if),
        .mem_req_if     (icache_mem_req_if),
        .mem_rsp_if     (icache_mem_rsp_if)
    );

    for (genvar i = 0; i < `ICACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM (icache_rsp_if, icache_rsp_qual_if, i);
    end

    ///////////////////////////////////////////////////////////////////////////

    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_rsp_if();

    VX_cache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_rsp_if();
    
    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`DCACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_req_qual_if[`DCACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_WORD_SIZE*8), 
        .TAG_WIDTH  (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_rsp_qual_if[`DCACHE_NUM_REQS]();

    for (genvar i = 0; i < `DCACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM (dcache_nosm_req_qual_if, dcache_nosm_req_if, i);
    end

    `RESET_RELAY (dcache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`DCACHE_ID),
        .CACHE_SIZE     (`DCACHE_SIZE),
        .CACHE_LINE_SIZE(`DCACHE_LINE_SIZE),
        .NUM_BANKS      (`DCACHE_NUM_BANKS),
        .NUM_WAYS       (`DCACHE_NUM_WAYS),
        .NUM_PORTS      (`DCACHE_NUM_PORTS),
        .WORD_SIZE      (`DCACHE_WORD_SIZE),
        .NUM_REQS       (`DCACHE_NUM_REQS),
        .CREQ_SIZE      (`DCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`DCACHE_MREQ_SIZE),
        .WRITE_ENABLE   (1),
        .REQ_UUID_BITS  (`UUID_BITS),
        .CORE_TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`DCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE      (1),
        .NC_TAG_BIT     (0),
        .PASSTHRU       (!`DCACHE_ENABLED)
    ) dcache (
        `SCOPE_BIND_VX_mem_unit_dcache

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_dcache_if),
    `endif
        
        .clk            (clk),
        .reset          (dcache_reset),        
        .core_req_if    (dcache_nosm_req_qual_if),
        .core_rsp_if    (dcache_nosm_rsp_qual_if),
        .mem_req_if     (dcache_mem_req_if),
        .mem_rsp_if     (dcache_mem_rsp_if)
    ); 

    for (genvar i = 0; i < `DCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM (dcache_nosm_rsp_if, dcache_nosm_rsp_qual_if, i);
    end

    ///////////////////////////////////////////////////////////////////////////

`ifdef SM_ENABLE

    VX_cache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) smem_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) smem_rsp_if();

    `RESET_RELAY (smem_reset, reset);

    VX_cache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_demux_req_if[2]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_NOSM_TAG_WIDTH)
    ) dcache_nosm_demux_rsp_if[2]();

    VX_cache_demux #(
        .NUM_REQS      (2),
        .NUM_LANES     (`NUM_THREADS),
        .DATA_SIZE     (4),            
        .TAG_IN_WIDTH  (`DCACHE_TAG_WIDTH),
        .TAG_SEL_IDX   (0),
        .ARBITER       ("P"),
        .BUFFERED_REQ  (1),
        .BUFFERED_RSP  (1)
    ) dcache_nosm_demux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (dcache_req_if),
        .rsp_in_if  (dcache_rsp_if),
        .req_out_if (dcache_nosm_demux_req_if),
        .rsp_out_if (dcache_nosm_demux_rsp_if)
    );

    `ASSIGN_VX_CACHE_REQ_IF (dcache_nosm_req_if, dcache_nosm_demux_req_if[0]);
    `ASSIGN_VX_CACHE_RSP_IF (dcache_nosm_demux_rsp_if[0], dcache_nosm_rsp_if);
    `ASSIGN_VX_CACHE_REQ_IF (smem_req_if, dcache_nosm_demux_req_if[1]);
    `ASSIGN_VX_CACHE_RSP_IF (dcache_nosm_demux_rsp_if[1], smem_rsp_if);

    // shared memory address mapping:  
    // addr = [warp_idx][per_thread_word_idx][bank_idx]
    `define __BANK_ADDR_OFFSET  `CLOG2(`STACK_SIZE / `DCACHE_WORD_SIZE)
    `define __WORD_SEL_BITS     `CLOG2(`SMEM_LOCAL_SIZE / `DCACHE_WORD_SIZE)
    `define __WARP_SEL_BITS     `CLOG2(`NUM_WARPS)
    `define __BANK_SEL_BITS     `CLOG2(`SMEM_NUM_BANKS)
    `define SMEM_ADDR_WIDTH     (`__WARP_SEL_BITS + `__WORD_SEL_BITS + `__BANK_SEL_BITS)

    wire [`DCACHE_NUM_REQS-1:0][`SMEM_ADDR_WIDTH-1:0] smem_req_addr;    
    for (genvar i = 0; i < `DCACHE_NUM_REQS; ++i) begin
        if (`__BANK_SEL_BITS != 0) begin
            assign smem_req_addr[i][0 +: `__BANK_SEL_BITS] = smem_req_if.addr[i][`__BANK_ADDR_OFFSET +: `__BANK_SEL_BITS];
        end
        assign smem_req_addr[i][`__BANK_SEL_BITS +: `__WORD_SEL_BITS] = smem_req_if.addr[i][0 +: `__WORD_SEL_BITS];        
        if (`__WARP_SEL_BITS != 0) begin
            assign smem_req_addr[i][`__BANK_SEL_BITS + `__WORD_SEL_BITS +: `__WARP_SEL_BITS] = smem_req_if.addr[i][(`__BANK_ADDR_OFFSET + `__BANK_SEL_BITS) +: `__WARP_SEL_BITS];
        end
    end

    VX_shared_mem #(
        .IDNAME     (`SMEM_ID),
        .SIZE       (`SMEM_SIZE),
        .NUM_REQS   (`DCACHE_NUM_REQS),
        .NUM_BANKS  (`SMEM_NUM_BANKS),
        .WORD_SIZE  (`DCACHE_WORD_SIZE),
        .ADDR_WIDTH (`SMEM_ADDR_WIDTH),
        .CREQ_SIZE  (`SMEM_CREQ_SIZE),
        .CRSQ_SIZE  (`SMEM_CRSQ_SIZE),
        .REQ_UUID_BITS(`UUID_BITS), 
        .TAG_WIDTH  (`DCACHE_NOSM_TAG_WIDTH)
    ) smem (            
        .clk        (clk),
        .reset      (smem_reset),

    `ifdef PERF_ENABLE
        .perf_cache_if(perf_smem_if),
    `endif

        // Core request
        .req_valid  (smem_req_if.valid),
        .req_rw     (smem_req_if.rw),
        .req_byteen (smem_req_if.byteen),
        .req_addr   (smem_req_addr),
        .req_data   (smem_req_if.data),        
        .req_tag    (smem_req_if.tag),
        .req_ready  (smem_req_if.ready),

        // Core response
        .rsp_valid  (smem_rsp_if.valid),
        .rsp_data   (smem_rsp_if.data),
        .rsp_tag    (smem_rsp_if.tag),
        .rsp_ready  (smem_rsp_if.ready)
    );    

`else

    // core to D-cache request
    for (genvar i = 0; i < `DCACHE_NUM_REQS; ++i) begin
        VX_skid_buffer #(
            .DATAW ((32-`CLOG2(`DCACHE_WORD_SIZE)) + 1 + `DCACHE_WORD_SIZE + (8*`DCACHE_WORD_SIZE) + `DCACHE_TAG_WIDTH)
        ) req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (dcache_req_if.valid[i]),        
            .data_in   ({dcache_req_if.addr[i], dcache_req_if.rw[i], dcache_req_if.byteen[i], dcache_req_if.data[i], dcache_req_if.tag[i]}),
            .ready_in  (dcache_req_if.ready[i]),
            .valid_out (dcache_nosm_req_if.valid[i]),
            .data_out  ({dcache_nosm_req_if.addr[i], dcache_nosm_req_if.rw[i], dcache_nosm_req_if.byteen[i], dcache_nosm_req_if.data[i], dcache_nosm_req_if.tag[i]}),
            .ready_out (dcache_nosm_req_if.ready[i])
        );
    end
    
    // D-cache to core reponse
    `ASSIGN_VX_CACHE_RSP_IF (dcache_rsp_if, dcache_nosm_rsp_if);

`endif    

///////////////////////////////////////////////////////////////////////////////

`ifdef EXT_TEX_ENABLE

    VX_mem_req_if #(
        .DATA_WIDTH (`TCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`TCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`TCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_rsp_if();

    VX_mem_req_if #(
        .DATA_WIDTH (`TCACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`TCACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`TCACHE_TAG_WIDTH)
    ) tcache_req_qual_if[`TCACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`TCACHE_WORD_SIZE*8), 
        .TAG_WIDTH  (`TCACHE_TAG_WIDTH)
    ) tcache_rsp_qual_if[`TCACHE_NUM_REQS]();

    for (genvar i = 0; i < `TCACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM (tcache_req_qual_if, tcache_req_if, i);
    end

    `RESET_RELAY (tcache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`TCACHE_ID),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .CACHE_LINE_SIZE(`TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .WORD_SIZE      (`TCACHE_WORD_SIZE),
        .NUM_REQS       (`TCACHE_NUM_REQS),
        .CREQ_SIZE      (`TCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .WRITE_ENABLE   (0),
        .CORE_TAG_WIDTH (`TCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH),
        .PASSTHRU       (!`TCACHE_ENABLED)
    ) tcache (
        `SCOPE_BIND_VX_mem_unit_tcache

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_tcache_if),
    `endif

        .clk            (clk),
        .reset          (tcache_reset),
        .core_req_if    (tcache_req_qual_if),
        .core_rsp_if    (tcache_rsp_qual_if),
        .mem_req_if     (tcache_mem_req_if),
        .mem_rsp_if     (tcache_mem_rsp_if)
    );

    for (genvar i = 0; i < `TCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM (tcache_rsp_if, tcache_rsp_qual_if, i);
    end    

`endif

    ///////////////////////////////////////////////////////////////////////////    

    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_IN_WIDTH)
    ) l1_mem_req_if[2+`EXT_TEX_ENABLED]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_IN_WIDTH)
    ) l1_mem_rsp_if[2+`EXT_TEX_ENABLED]();

    `ASSIGN_VX_MEM_REQ_IF_XTAG (l1_mem_req_if[0], icache_mem_req_if);
    assign l1_mem_req_if[0].tag = `L1_MEM_TAG_IN_WIDTH'(icache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (icache_mem_rsp_if, l1_mem_rsp_if[0]);
    assign icache_mem_rsp_if.tag = `ICACHE_MEM_TAG_WIDTH'(l1_mem_rsp_if[0].tag);

    `ASSIGN_VX_MEM_REQ_IF_XTAG (l1_mem_req_if[1], dcache_mem_req_if);
    assign l1_mem_req_if[1].tag = `L1_MEM_TAG_IN_WIDTH'(dcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (dcache_mem_rsp_if, l1_mem_rsp_if[1]);
    assign dcache_mem_rsp_if.tag = `DCACHE_MEM_TAG_WIDTH'(l1_mem_rsp_if[1].tag);

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (l1_mem_req_if[2], tcache_mem_req_if);
    assign l1_mem_req_if[2].tag = `L1_MEM_TAG_IN_WIDTH'(tcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (tcache_mem_rsp_if, l1_mem_rsp_if[2]);
    assign tcache_mem_rsp_if.tag = `TCACHE_MEM_TAG_WIDTH'(l1_mem_rsp_if[2].tag);
`endif

    VX_mem_mux #(
        .NUM_REQS     (2 + `EXT_TEX_ENABLED),
        .DATA_WIDTH   (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH   (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_IN_WIDTH (`L1_MEM_TAG_IN_WIDTH),        
        .ARBITER      (`EXT_TEX_ENABLED ? "R" : "P"),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .BUFFERED_REQ (1),
        .BUFFERED_RSP (2)
    ) mem_mux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (l1_mem_req_if),
        .rsp_in_if  (l1_mem_rsp_if),
        .req_out_if (mem_req_if),        
        .rsp_out_if (mem_rsp_if)
    );

    ///////////////////////////////////////////////////////////////////////////

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
