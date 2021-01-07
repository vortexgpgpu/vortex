`include "VX_define.vh"

module VX_mem_unit # (
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_mem_unit

    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    VX_perf_memsys_if       perf_memsys_if,
`endif

    // Core <-> Dcache    
    VX_cache_core_req_if    core_dcache_req_if,
    VX_cache_core_rsp_if    core_dcache_rsp_if,
    
    // Core <-> Icache    
    VX_cache_core_req_if    core_icache_req_if,  
    VX_cache_core_rsp_if    core_icache_rsp_if,

    // DRAM
    VX_cache_dram_req_if    dram_req_if,
    VX_cache_dram_rsp_if    dram_rsp_if
);
    
`ifdef PERF_ENABLE
    VX_perf_cache_if perf_icache_if(), perf_dcache_if(), perf_smem_if();
`endif

    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH (`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH (`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH  (`DDRAM_TAG_WIDTH)
    ) dcache_dram_req_if(), icache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH (`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH  (`DDRAM_TAG_WIDTH)
    ) dcache_dram_rsp_if(), icache_dram_rsp_if();

    VX_cache_core_req_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) dcache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) dcache_rsp_if();

    VX_cache_core_req_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) smem_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) smem_rsp_if();

    VX_databus_arb databus_arb (   
        .clk          (clk),
        .reset        (reset),

        .core_req_if  (core_dcache_req_if),
        .cache_req_if (dcache_req_if),
        .smem_req_if  (smem_req_if),

        .cache_rsp_if (dcache_rsp_if),
        .smem_rsp_if  (smem_rsp_if),
        .core_rsp_if  (core_dcache_rsp_if)
    ); 

    wire icache_reset, dcache_reset;   

    VX_reset_relay #(
        .NUM_NODES (2)
    ) reset_relay (
        .clk       (clk),
        .reset     (reset),
        .reset_out ({dcache_reset, icache_reset})
    );

    VX_cache #(
        .CACHE_ID           (`ICACHE_ID),
        .CACHE_SIZE         (`ICACHE_SIZE),
        .CACHE_LINE_SIZE    (`ICACHE_LINE_SIZE),
        .NUM_BANKS          (`INUM_BANKS),
        .WORD_SIZE          (`IWORD_SIZE),
        .NUM_REQS           (`INUM_REQUESTS),
        .CREQ_SIZE          (`ICREQ_SIZE),
        .MSHR_SIZE          (`IMSHR_SIZE),
        .DRSQ_SIZE          (`IDRSQ_SIZE),
        .CRSQ_SIZE          (`ICRSQ_SIZE),
        .DREQ_SIZE          (`IDREQ_SIZE),
        .DRAM_ENABLE        (1),
        .WRITE_ENABLE       (0),
        .CORE_TAG_WIDTH     (`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`ICORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH     (`DDRAM_TAG_WIDTH)
    ) icache (
        `SCOPE_BIND_VX_mem_unit_icache

        .clk                (clk),
        .reset              (icache_reset),

        // Core request
        .core_req_valid     (core_icache_req_if.valid),
        .core_req_rw        (core_icache_req_if.rw),
        .core_req_byteen    (core_icache_req_if.byteen),
        .core_req_addr      (core_icache_req_if.addr),
        .core_req_data      (core_icache_req_if.data),        
        .core_req_tag       (core_icache_req_if.tag),
        .core_req_ready     (core_icache_req_if.ready),

        // Core response
        .core_rsp_valid     (core_icache_rsp_if.valid),
        .core_rsp_data      (core_icache_rsp_if.data),
        .core_rsp_tag       (core_icache_rsp_if.tag),
        .core_rsp_ready     (core_icache_rsp_if.ready),

    `ifdef PERF_ENABLE
        .perf_cache_if      (perf_icache_if),
    `endif

        // DRAM Req
        .dram_req_valid     (icache_dram_req_if.valid),
        .dram_req_rw        (icache_dram_req_if.rw),        
        .dram_req_byteen    (icache_dram_req_if.byteen),        
        .dram_req_addr      (icache_dram_req_if.addr),
        .dram_req_data      (icache_dram_req_if.data),
        .dram_req_tag       (icache_dram_req_if.tag),
        .dram_req_ready     (icache_dram_req_if.ready),        

        // DRAM response
        .dram_rsp_valid     (icache_dram_rsp_if.valid),        
        .dram_rsp_data      (icache_dram_rsp_if.data),
        .dram_rsp_tag       (icache_dram_rsp_if.tag),
        .dram_rsp_ready     (icache_dram_rsp_if.ready)
    );

    VX_cache #(
        .CACHE_ID           (`DCACHE_ID),
        .CACHE_SIZE         (`DCACHE_SIZE),
        .CACHE_LINE_SIZE    (`DCACHE_LINE_SIZE),
        .NUM_BANKS          (`DNUM_BANKS),
        .WORD_SIZE          (`DWORD_SIZE),
        .NUM_REQS           (`DNUM_REQUESTS),
        .CREQ_SIZE          (`DCREQ_SIZE),
        .MSHR_SIZE          (`DMSHR_SIZE),
        .DRSQ_SIZE          (`DDRSQ_SIZE),
        .CRSQ_SIZE          (`DCRSQ_SIZE),
        .DREQ_SIZE          (`DDREQ_SIZE),    
        .DRAM_ENABLE        (1),
        .WRITE_ENABLE       (1),
        .CORE_TAG_WIDTH     (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`DCORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH     (`DDRAM_TAG_WIDTH)
    ) dcache (
        `SCOPE_BIND_VX_mem_unit_dcache
        
        .clk                (clk),
        .reset              (dcache_reset),

        // Core req
        .core_req_valid     (dcache_req_if.valid),
        .core_req_rw        (dcache_req_if.rw),
        .core_req_byteen    (dcache_req_if.byteen),
        .core_req_addr      (dcache_req_if.addr),
        .core_req_data      (dcache_req_if.data),        
        .core_req_tag       (dcache_req_if.tag),
        .core_req_ready     (dcache_req_if.ready),

        // Core response
        .core_rsp_valid     (dcache_rsp_if.valid),
        .core_rsp_data      (dcache_rsp_if.data),
        .core_rsp_tag       (dcache_rsp_if.tag),
        .core_rsp_ready     (dcache_rsp_if.ready),

    `ifdef PERF_ENABLE
        .perf_cache_if      (perf_dcache_if),
    `endif

        // DRAM request
        .dram_req_valid     (dcache_dram_req_if.valid),
        .dram_req_rw        (dcache_dram_req_if.rw),        
        .dram_req_byteen    (dcache_dram_req_if.byteen),        
        .dram_req_addr      (dcache_dram_req_if.addr),
        .dram_req_data      (dcache_dram_req_if.data),
        .dram_req_tag       (dcache_dram_req_if.tag),
        .dram_req_ready     (dcache_dram_req_if.ready),

        // DRAM response
        .dram_rsp_valid     (dcache_dram_rsp_if.valid),        
        .dram_rsp_data      (dcache_dram_rsp_if.data),
        .dram_rsp_tag       (dcache_dram_rsp_if.tag),
        .dram_rsp_ready     (dcache_dram_rsp_if.ready)
    ); 

    if (`SM_ENABLE) begin

        wire scache_reset;   

        VX_reset_relay reset_relay (
            .clk       (clk),
            .reset     (reset),
            .reset_out (scache_reset)
        );

        VX_cache #(
            .CACHE_ID           (`SCACHE_ID),
            .CACHE_SIZE         (`SMEM_SIZE),
            .CACHE_LINE_SIZE    (`SCACHE_LINE_SIZE),
            .NUM_BANKS          (`SNUM_BANKS),
            .WORD_SIZE          (`SWORD_SIZE),
            .NUM_REQS           (`SNUM_REQUESTS),
            .CREQ_SIZE          (`SCREQ_SIZE),
            .MSHR_SIZE          (8),
            .DRSQ_SIZE          (1),
            .CRSQ_SIZE          (`SCRSQ_SIZE),
            .DREQ_SIZE          (1),
            .DRAM_ENABLE        (0),
            .WRITE_ENABLE       (1),
            .CORE_TAG_WIDTH     (`DCORE_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (`DCORE_TAG_ID_BITS),
            .BANK_ADDR_OFFSET   (`SBANK_ADDR_OFFSET)
        ) smem (
            `SCOPE_BIND_VX_mem_unit_smem
            
            .clk                (clk),
            .reset              (scache_reset),

            // Core request
            .core_req_valid     (smem_req_if.valid),
            .core_req_rw        (smem_req_if.rw),
            .core_req_byteen    (smem_req_if.byteen),
            .core_req_addr      (smem_req_if.addr),
            .core_req_data      (smem_req_if.data),        
            .core_req_tag       (smem_req_if.tag),
            .core_req_ready     (smem_req_if.ready),

            // Core response
            .core_rsp_valid     (smem_rsp_if.valid),
            .core_rsp_data      (smem_rsp_if.data),
            .core_rsp_tag       (smem_rsp_if.tag),
            .core_rsp_ready     (smem_rsp_if.ready),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_smem_if),
        `endif

            // DRAM request
            `UNUSED_PIN (dram_req_valid),
            `UNUSED_PIN (dram_req_rw),        
            `UNUSED_PIN (dram_req_byteen),        
            `UNUSED_PIN (dram_req_addr),
            `UNUSED_PIN (dram_req_data),
            `UNUSED_PIN (dram_req_tag),
            .dram_req_ready     (1'b0),       

            // DRAM response
            .dram_rsp_valid     (1'b0),
            .dram_rsp_data      ((`SCACHE_LINE_SIZE*8)'(0)),
            .dram_rsp_tag       (`LOG2UP(`SNUM_BANKS)'(0)),
            `UNUSED_PIN (dram_rsp_ready)
        );
    
    end

    VX_mem_arb #(
        .NUM_REQS      (2),
        .DATA_WIDTH    (`DDRAM_LINE_WIDTH),
        .ADDR_WIDTH    (`DDRAM_ADDR_WIDTH),
        .TAG_IN_WIDTH  (`DDRAM_TAG_WIDTH),
        .TAG_OUT_WIDTH (`XDRAM_TAG_WIDTH),
        .BUFFERED_REQ  (1),
        .BUFFERED_RSP  (0)
    ) dram_arb (
        .clk            (clk),
        .reset          (reset),

        // Source request
        .req_valid_in   ({dcache_dram_req_if.valid,   icache_dram_req_if.valid}),
        .req_rw_in      ({dcache_dram_req_if.rw,      icache_dram_req_if.rw}),
        .req_byteen_in  ({dcache_dram_req_if.byteen,  icache_dram_req_if.byteen}),
        .req_addr_in    ({dcache_dram_req_if.addr,    icache_dram_req_if.addr}),
        .req_data_in    ({dcache_dram_req_if.data,    icache_dram_req_if.data}),  
        .req_tag_in     ({dcache_dram_req_if.tag,     icache_dram_req_if.tag}),  
        .req_ready_in   ({dcache_dram_req_if.ready,   icache_dram_req_if.ready}),

        // DRAM request
        .req_valid_out  (dram_req_if.valid),
        .req_rw_out     (dram_req_if.rw),        
        .req_byteen_out (dram_req_if.byteen),        
        .req_addr_out   (dram_req_if.addr),
        .req_data_out   (dram_req_if.data),
        .req_tag_out    (dram_req_if.tag),
        .req_ready_out  (dram_req_if.ready),

        // Source response
        .rsp_valid_out  ({dcache_dram_rsp_if.valid,   icache_dram_rsp_if.valid}),
        .rsp_data_out   ({dcache_dram_rsp_if.data,    icache_dram_rsp_if.data}),
        .rsp_tag_out    ({dcache_dram_rsp_if.tag,     icache_dram_rsp_if.tag}),
        .rsp_ready_out  ({dcache_dram_rsp_if.ready,   icache_dram_rsp_if.ready}),
        
        // DRAM response
        .rsp_valid_in   (dram_rsp_if.valid),
        .rsp_tag_in     (dram_rsp_if.tag),
        .rsp_data_in    (dram_rsp_if.data),
        .rsp_ready_in   (dram_rsp_if.ready)
    );

`ifdef PERF_ENABLE
    
    assign perf_memsys_if.icache_reads       = perf_icache_if.reads;
    assign perf_memsys_if.icache_read_misses = perf_icache_if.read_misses;
    assign perf_memsys_if.icache_pipe_stalls = perf_icache_if.pipe_stalls;
    assign perf_memsys_if.icache_crsp_stalls = perf_icache_if.crsp_stalls;

    assign perf_memsys_if.dcache_reads       = perf_dcache_if.reads;
    assign perf_memsys_if.dcache_writes      = perf_dcache_if.writes;
    assign perf_memsys_if.dcache_read_misses = perf_dcache_if.read_misses;
    assign perf_memsys_if.dcache_write_misses= perf_dcache_if.write_misses;
    assign perf_memsys_if.dcache_bank_stalls = perf_dcache_if.bank_stalls;
    assign perf_memsys_if.dcache_mshr_stalls = perf_dcache_if.mshr_stalls;    
    assign perf_memsys_if.dcache_pipe_stalls = perf_dcache_if.pipe_stalls;
    assign perf_memsys_if.dcache_crsp_stalls = perf_dcache_if.crsp_stalls;

if (`SM_ENABLE) begin
    assign perf_memsys_if.smem_reads         = perf_smem_if.reads;
    assign perf_memsys_if.smem_writes        = perf_smem_if.writes;
    assign perf_memsys_if.smem_bank_stalls   = perf_smem_if.bank_stalls;    
end else begin
    assign perf_memsys_if.smem_reads         = 0;
    assign perf_memsys_if.smem_writes        = 0;
    assign perf_memsys_if.smem_bank_stalls   = 0;
end
    
    reg [63:0] perf_dram_lat_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_dram_lat_per_cycle <= 0;
        end else begin 
            if (dram_req_if.valid && !dram_req_if.rw && dram_req_if.ready && dram_rsp_if.valid && dram_rsp_if.ready) begin
                perf_dram_lat_per_cycle <= perf_dram_lat_per_cycle;
            end else if (dram_req_if.valid && !dram_req_if.rw && dram_req_if.ready) begin
                perf_dram_lat_per_cycle <= perf_dram_lat_per_cycle + 64'd1;
            end else if (dram_rsp_if.valid && dram_rsp_if.ready) begin
                perf_dram_lat_per_cycle <= perf_dram_lat_per_cycle - 64'd1;
            end
        end
    end
    
    reg [63:0] perf_dram_reads, perf_dram_writes, perf_dram_lat, perf_dram_stalls;

    always @(posedge clk) begin
        if (reset) begin       
            perf_dram_reads  <= 0;     
            perf_dram_writes <= 0;            
            perf_dram_lat    <= 0;
            perf_dram_stalls <= 0;
        end else begin  
            if (dram_req_if.valid && dram_req_if.ready && !dram_req_if.rw) begin
                perf_dram_reads <= perf_dram_reads + 64'd1;
            end
            if (dram_req_if.valid && dram_req_if.ready && dram_req_if.rw) begin
                perf_dram_writes <= perf_dram_writes + 64'd1;
            end            
            if (dram_req_if.valid && !dram_req_if.ready) begin
                perf_dram_stalls <= perf_dram_stalls + 64'd1;
            end       
            perf_dram_lat <= perf_dram_lat + perf_dram_lat_per_cycle;
        end
    end

    assign perf_memsys_if.dram_reads   = perf_dram_reads;       
    assign perf_memsys_if.dram_writes  = perf_dram_writes;
    assign perf_memsys_if.dram_latency = perf_dram_lat; 
    assign perf_memsys_if.dram_stalls  = perf_dram_stalls;
`endif
    
endmodule
