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

    // Core <-> Dcache    
    VX_dcache_req_if.slave  dcache_req_if,
    VX_dcache_rsp_if.master dcache_rsp_if,
    
    // Core <-> Icache    
    VX_icache_req_if.slave  icache_req_if,  
    VX_icache_rsp_if.master icache_rsp_if,

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if
);
    
`ifdef PERF_ENABLE
    VX_perf_cache_if perf_icache_if(), perf_dcache_if(), perf_smem_if();
`endif

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
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_rsp_if();

    VX_dcache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE)
    ) dcache_req_tmp_if();

    VX_dcache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE)
    ) dcache_rsp_tmp_if();

    `RESET_RELAY (icache_reset);
    `RESET_RELAY (dcache_reset);
    `RESET_RELAY (mem_arb_reset);

    VX_cache #(
        .CACHE_ID           (`ICACHE_ID),
        .CACHE_SIZE         (`ICACHE_SIZE),
        .CACHE_LINE_SIZE    (`ICACHE_LINE_SIZE),
        .NUM_BANKS          (1),
        .WORD_SIZE          (`ICACHE_WORD_SIZE),
        .NUM_REQS           (1),
        .CREQ_SIZE          (`ICACHE_CREQ_SIZE),
        .CRSQ_SIZE          (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE          (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE          (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE          (`ICACHE_MREQ_SIZE),
        .WRITE_ENABLE       (0),
        .CORE_TAG_WIDTH     (`ICACHE_CORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`ICACHE_CORE_TAG_ID_BITS),
        .MEM_TAG_WIDTH      (`ICACHE_MEM_TAG_WIDTH)
    ) icache (
        `SCOPE_BIND_VX_mem_unit_icache

        .clk                (clk),
        .reset              (icache_reset),

        // Core request
        .core_req_valid     (icache_req_if.valid),
        .core_req_rw        (1'b0),
        .core_req_byteen    ('b0),
        .core_req_addr      (icache_req_if.addr),
        .core_req_data      ('x),        
        .core_req_tag       (icache_req_if.tag),
        .core_req_ready     (icache_req_if.ready),

        // Core response
        .core_rsp_valid     (icache_rsp_if.valid),
        .core_rsp_data      (icache_rsp_if.data),
        .core_rsp_tag       (icache_rsp_if.tag),
        .core_rsp_ready     (icache_rsp_if.ready),
        `UNUSED_PIN (core_rsp_tmask),

    `ifdef PERF_ENABLE
        .perf_cache_if      (perf_icache_if),
    `endif

        // Memory Request
        .mem_req_valid     (icache_mem_req_if.valid),
        .mem_req_rw        (icache_mem_req_if.rw),        
        .mem_req_byteen    (icache_mem_req_if.byteen),        
        .mem_req_addr      (icache_mem_req_if.addr),
        .mem_req_data      (icache_mem_req_if.data),
        .mem_req_tag       (icache_mem_req_if.tag),
        .mem_req_ready     (icache_mem_req_if.ready),        

        // Memory response
        .mem_rsp_valid     (icache_mem_rsp_if.valid),        
        .mem_rsp_data      (icache_mem_rsp_if.data),
        .mem_rsp_tag       (icache_mem_rsp_if.tag),
        .mem_rsp_ready     (icache_mem_rsp_if.ready)
    );

    VX_cache #(
        .CACHE_ID           (`DCACHE_ID),
        .CACHE_SIZE         (`DCACHE_SIZE),
        .CACHE_LINE_SIZE    (`DCACHE_LINE_SIZE),
        .NUM_BANKS          (`DCACHE_NUM_BANKS),
        .NUM_PORTS          (`DCACHE_NUM_PORTS),
        .WORD_SIZE          (`DCACHE_WORD_SIZE),
        .NUM_REQS           (`DCACHE_NUM_REQS),
        .CREQ_SIZE          (`DCACHE_CREQ_SIZE),
        .CRSQ_SIZE          (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE          (`DCACHE_MSHR_SIZE),
        .MRSQ_SIZE          (`DCACHE_MRSQ_SIZE),
        .MREQ_SIZE          (`DCACHE_MREQ_SIZE),
        .WRITE_ENABLE       (1),
        .CORE_TAG_WIDTH     (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE),
        .CORE_TAG_ID_BITS   (`DCACHE_CORE_TAG_ID_BITS-`SM_ENABLE),
        .MEM_TAG_WIDTH      (`DCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE          (1)
    ) dcache (
        `SCOPE_BIND_VX_mem_unit_dcache
        
        .clk                (clk),
        .reset              (dcache_reset),

        // Core req
        .core_req_valid     (dcache_req_tmp_if.valid),
        .core_req_rw        (dcache_req_tmp_if.rw),
        .core_req_byteen    (dcache_req_tmp_if.byteen),
        .core_req_addr      (dcache_req_tmp_if.addr),
        .core_req_data      (dcache_req_tmp_if.data),        
        .core_req_tag       (dcache_req_tmp_if.tag),
        .core_req_ready     (dcache_req_tmp_if.ready),

        // Core response
        .core_rsp_valid     (dcache_rsp_tmp_if.valid),
        .core_rsp_tmask     (dcache_rsp_tmp_if.tmask),
        .core_rsp_data      (dcache_rsp_tmp_if.data),
        .core_rsp_tag       (dcache_rsp_tmp_if.tag),
        .core_rsp_ready     (dcache_rsp_tmp_if.ready),

    `ifdef PERF_ENABLE
        .perf_cache_if      (perf_dcache_if),
    `endif

        // Memory request
        .mem_req_valid      (dcache_mem_req_if.valid),
        .mem_req_rw         (dcache_mem_req_if.rw),        
        .mem_req_byteen     (dcache_mem_req_if.byteen),        
        .mem_req_addr       (dcache_mem_req_if.addr),
        .mem_req_data       (dcache_mem_req_if.data),
        .mem_req_tag        (dcache_mem_req_if.tag),
        .mem_req_ready      (dcache_mem_req_if.ready),

        // Memory response
        .mem_rsp_valid      (dcache_mem_rsp_if.valid),        
        .mem_rsp_data       (dcache_mem_rsp_if.data),
        .mem_rsp_tag        (dcache_mem_rsp_if.tag),
        .mem_rsp_ready      (dcache_mem_rsp_if.ready)
    ); 

    if (`SM_ENABLE) begin                
        VX_dcache_req_if #(
            .NUM_REQS  (`DCACHE_NUM_REQS), 
            .WORD_SIZE (`DCACHE_WORD_SIZE), 
            .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE)
        ) smem_req_if();

        VX_dcache_rsp_if #(
            .NUM_REQS  (`DCACHE_NUM_REQS), 
            .WORD_SIZE (`DCACHE_WORD_SIZE), 
            .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE)
        ) smem_rsp_if();

        `RESET_RELAY (smem_arb_reset);
        `RESET_RELAY (smem_reset);

        VX_smem_arb #(
            .NUM_REQS      (2),
            .LANES         (`NUM_THREADS),
            .DATA_SIZE     (4),            
            .TAG_IN_WIDTH  (`DCACHE_CORE_TAG_WIDTH),
            .TAG_SEL_IDX   (0), // SM flag
            .TYPE          ("P"),
            .BUFFERED_REQ  (2),
            .BUFFERED_RSP  (1)
        ) smem_arb (
            .clk          (clk),
            .reset        (smem_arb_reset),

            // input request
            .req_valid_in   (dcache_req_if.valid),
            .req_rw_in      (dcache_req_if.rw),        
            .req_byteen_in  (dcache_req_if.byteen),        
            .req_addr_in    (dcache_req_if.addr),
            .req_data_in    (dcache_req_if.data),
            .req_tag_in     (dcache_req_if.tag),
            .req_ready_in   (dcache_req_if.ready),
            
            // output requests
            .req_valid_out  ({smem_req_if.valid,  dcache_req_tmp_if.valid}),
            .req_rw_out     ({smem_req_if.rw,     dcache_req_tmp_if.rw}),
            .req_byteen_out ({smem_req_if.byteen, dcache_req_tmp_if.byteen}),
            .req_addr_out   ({smem_req_if.addr,   dcache_req_tmp_if.addr}),
            .req_data_out   ({smem_req_if.data,   dcache_req_tmp_if.data}),  
            .req_tag_out    ({smem_req_if.tag,    dcache_req_tmp_if.tag}),  
            .req_ready_out  ({smem_req_if.ready,  dcache_req_tmp_if.ready}),            
            
            // input responses
            .rsp_valid_in   ({smem_rsp_if.valid, dcache_rsp_tmp_if.valid}),
            .rsp_tmask_in   ({smem_rsp_if.tmask, dcache_rsp_tmp_if.tmask}),
            .rsp_data_in    ({smem_rsp_if.data,  dcache_rsp_tmp_if.data}),
            .rsp_tag_in     ({smem_rsp_if.tag,   dcache_rsp_tmp_if.tag}),
            .rsp_ready_in   ({smem_rsp_if.ready, dcache_rsp_tmp_if.ready}),

            // output response
            .rsp_valid_out  (dcache_rsp_if.valid),
            .rsp_tmask_out  (dcache_rsp_if.tmask),
            .rsp_tag_out    (dcache_rsp_if.tag),
            .rsp_data_out   (dcache_rsp_if.data),
            .rsp_ready_out  (dcache_rsp_if.ready)
        );

        VX_shared_mem #(
            .CACHE_ID           (`SMEM_ID),
            .CACHE_SIZE         (`SMEM_SIZE),
            .NUM_BANKS          (`SMEM_NUM_BANKS),
            .WORD_SIZE          (`SMEM_WORD_SIZE),
            .NUM_REQS           (`SMEM_NUM_REQS),
            .CREQ_SIZE          (`SMEM_CREQ_SIZE),
            .CRSQ_SIZE          (`SMEM_CRSQ_SIZE),
            .CORE_TAG_WIDTH     (`DCACHE_CORE_TAG_WIDTH-`SM_ENABLE),
            .CORE_TAG_ID_BITS   (`DCACHE_CORE_TAG_ID_BITS-`SM_ENABLE),
            .BANK_ADDR_OFFSET   (`SMEM_BANK_ADDR_OFFSET)
        ) smem (            
            .clk                (clk),
            .reset              (smem_reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_smem_if),
        `endif

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
            .core_rsp_tmask     (smem_rsp_if.tmask),
            .core_rsp_data      (smem_rsp_if.data),
            .core_rsp_tag       (smem_rsp_if.tag),
            .core_rsp_ready     (smem_rsp_if.ready)
        );    
    end else begin
        // core to D-cache request
        for (genvar i = 0; i < `DCACHE_NUM_REQS; ++i) begin
            VX_skid_buffer #(
                .DATAW ((32-`CLOG2(`DCACHE_WORD_SIZE)) + 1 + `DCACHE_WORD_SIZE + (8*`DCACHE_WORD_SIZE) + `DCACHE_CORE_TAG_WIDTH)
            ) req_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (dcache_req_if.valid[i]),        
                .data_in   ({dcache_req_if.addr[i], dcache_req_if.rw[i], dcache_req_if.byteen[i], dcache_req_if.data[i], dcache_req_if.tag[i]}),
                .ready_in  (dcache_req_if.ready[i]),
                .valid_out (dcache_req_tmp_if.valid[i]),
                .data_out  ({dcache_req_tmp_if.addr[i], dcache_req_tmp_if.rw[i], dcache_req_tmp_if.byteen[i], dcache_req_tmp_if.data[i], dcache_req_tmp_if.tag[i]}),
                .ready_out (dcache_req_tmp_if.ready[i])
            );
        end
        
        // D-cache to core reponse
        assign dcache_rsp_if.valid  = dcache_rsp_tmp_if.valid;
        assign dcache_rsp_if.tmask  = dcache_rsp_tmp_if.tmask;
        assign dcache_rsp_if.tag    = dcache_rsp_tmp_if.tag;
        assign dcache_rsp_if.data   = dcache_rsp_tmp_if.data;
        assign dcache_rsp_tmp_if.ready = dcache_rsp_if.ready;
    end

    wire [`DCACHE_MEM_TAG_WIDTH-1:0] icache_mem_req_tag = `DCACHE_MEM_TAG_WIDTH'(icache_mem_req_if.tag);
    wire [`DCACHE_MEM_TAG_WIDTH-1:0] icache_mem_rsp_tag;
    assign icache_mem_rsp_if.tag = icache_mem_rsp_tag[`ICACHE_MEM_TAG_WIDTH-1:0];
    `UNUSED_VAR (icache_mem_rsp_tag)

    VX_mem_arb #(
        .NUM_REQS      (2),
        .DATA_WIDTH    (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH    (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_IN_WIDTH  (`DCACHE_MEM_TAG_WIDTH),
        .TYPE          ("R"),
        .TAG_SEL_IDX   (1), // Skip 0 for NC flag
        .BUFFERED_REQ  (1),
        .BUFFERED_RSP  (2)
    ) mem_arb (
        .clk            (clk),
        .reset          (mem_arb_reset),

        // Source request
        .req_valid_in   ({dcache_mem_req_if.valid,  icache_mem_req_if.valid}),
        .req_rw_in      ({dcache_mem_req_if.rw,     icache_mem_req_if.rw}),
        .req_byteen_in  ({dcache_mem_req_if.byteen, icache_mem_req_if.byteen}),
        .req_addr_in    ({dcache_mem_req_if.addr,   icache_mem_req_if.addr}),
        .req_data_in    ({dcache_mem_req_if.data,   icache_mem_req_if.data}),  
        .req_tag_in     ({dcache_mem_req_if.tag,    icache_mem_req_tag}),  
        .req_ready_in   ({dcache_mem_req_if.ready,  icache_mem_req_if.ready}),

        // Memory request
        .req_valid_out  (mem_req_if.valid),
        .req_rw_out     (mem_req_if.rw),        
        .req_byteen_out (mem_req_if.byteen),        
        .req_addr_out   (mem_req_if.addr),
        .req_data_out   (mem_req_if.data),
        .req_tag_out    (mem_req_if.tag),
        .req_ready_out  (mem_req_if.ready),

        // Source response
        .rsp_valid_out  ({dcache_mem_rsp_if.valid, icache_mem_rsp_if.valid}),
        .rsp_data_out   ({dcache_mem_rsp_if.data,  icache_mem_rsp_if.data}),
        .rsp_tag_out    ({dcache_mem_rsp_if.tag,   icache_mem_rsp_tag}),
        .rsp_ready_out  ({dcache_mem_rsp_if.ready, icache_mem_rsp_if.ready}),
        
        // Memory response
        .rsp_valid_in   (mem_rsp_if.valid),
        .rsp_tag_in     (mem_rsp_if.tag),
        .rsp_data_in    (mem_rsp_if.data),
        .rsp_ready_in   (mem_rsp_if.ready)
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
    
    reg [`PERF_CTR_BITS-1:0] perf_mem_lat_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_lat_per_cycle <= 0;
        end else begin
            perf_mem_lat_per_cycle <= perf_mem_lat_per_cycle + 
                `PERF_CTR_BITS'($signed(2'((mem_req_if.valid && !mem_req_if.rw && mem_req_if.ready) && !(mem_rsp_if.valid && mem_rsp_if.ready)) - 
                    2'((mem_rsp_if.valid && mem_rsp_if.ready) && !(mem_req_if.valid && !mem_req_if.rw && mem_req_if.ready))));
        end
    end
    
    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_lat;
    reg [`PERF_CTR_BITS-1:0] perf_mem_stalls;

    always @(posedge clk) begin
        if (reset) begin       
            perf_mem_reads  <= 0;     
            perf_mem_writes <= 0;            
            perf_mem_lat    <= 0;
            perf_mem_stalls <= 0;
        end else begin  
            if (mem_req_if.valid && mem_req_if.ready && !mem_req_if.rw) begin
                perf_mem_reads <= perf_mem_reads + `PERF_CTR_BITS'd1;
            end
            if (mem_req_if.valid && mem_req_if.ready && mem_req_if.rw) begin
                perf_mem_writes <= perf_mem_writes + `PERF_CTR_BITS'd1;
            end            
            if (mem_req_if.valid && !mem_req_if.ready) begin
                perf_mem_stalls <= perf_mem_stalls + `PERF_CTR_BITS'd1;
            end       
            perf_mem_lat <= perf_mem_lat + perf_mem_lat_per_cycle;
        end
    end

    assign perf_memsys_if.mem_reads   = perf_mem_reads;       
    assign perf_memsys_if.mem_writes  = perf_mem_writes;
    assign perf_memsys_if.mem_latency = perf_mem_lat; 
    assign perf_memsys_if.mem_stalls  = perf_mem_stalls;
`endif
    
endmodule
