`include "VX_cache_define.vh"

module VX_cache_wrap #(
    parameter string INSTANCE_ID    = "",

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 4096, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64, 
    // Number of banks
    parameter NUM_BANKS             = 1,
    // Number of ports per banks
    parameter NUM_PORTS             = 1,
    // Number of associative ways
    parameter NUM_WAYS              = 1,
    // Size of a word in bytes
    parameter WORD_SIZE             = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE             = 0,
    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 8, 
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // Core response output register
    parameter CORE_OUT_REG          = 0,

    // Memory request output register
    parameter MEM_OUT_REG           = 0,

    // enable bypass for non-cacheable addresses
    parameter NC_TAG_BIT            = 0,
    parameter NC_ENABLE             = 0,

    // Force bypass for all requests
    parameter PASSTHRU              = 0
 ) (
    
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_cache_if,
`endif

    // Core request
    VX_mem_req_if.slave     core_req_if [NUM_REQS],

    // Core response
    VX_mem_rsp_if.master    core_rsp_if [NUM_REQS],

    // Memory request
    VX_mem_req_if.master    mem_req_if,
    
    // Memory response
    VX_mem_rsp_if.slave     mem_rsp_if
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid parameter"))    
    `STATIC_ASSERT(NUM_BANKS == (1 << $clog2(NUM_BANKS)), ("invalid parameter"))
    `STATIC_ASSERT(NUM_PORTS <= NUM_REQS, ("invalid parameter"))
    `STATIC_ASSERT(NUM_PORTS <= `WORDS_PER_LINE, ("invalid parameter"))

    localparam WORD_SEL_BITS    = `UP(`WORD_SEL_BITS);
    localparam MSHR_ADDR_WIDTH  = `LOG2UP(MSHR_SIZE);    
    localparam CORE_TAG_X_WIDTH = TAG_WIDTH - NC_ENABLE;
    localparam MEM_TAG_X_WIDTH  = MSHR_ADDR_WIDTH + `BANK_SEL_BITS;
    localparam MEM_TAG_WIDTH    = PASSTHRU ? (NC_ENABLE ? `CACHE_NC_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, TAG_WIDTH) : 
                                                          `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, TAG_WIDTH)) : 
                                             (NC_ENABLE ? `CACHE_NC_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, NUM_REQS, LINE_SIZE, WORD_SIZE, TAG_WIDTH) :
                                                          `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS));

    localparam NC_BYPASS = (NC_ENABLE || PASSTHRU);
    localparam DIRECT_PASSTHRU = PASSTHRU && (`WORD_SEL_BITS == 0) && (NUM_REQS == 1);

    localparam CORE_REQ_BUF_ENABLE = (1 != NUM_BANKS) || (1 != NUM_REQS);
    localparam MEM_REQ_BUF_ENABLE  = (1 != NUM_BANKS);

    wire [NUM_REQS-1:0]                     core_req_valid;
    wire [NUM_REQS-1:0]                     core_req_rw;
    wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]      core_req_byteen;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]    core_req_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]      core_req_tag;
    wire [NUM_REQS-1:0]                     core_req_ready;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_valid[i]    = core_req_if[i].valid;
        assign core_req_rw[i]       = core_req_if[i].rw;
        assign core_req_addr[i]     = core_req_if[i].addr;
        assign core_req_byteen[i]   = core_req_if[i].byteen;
        assign core_req_data[i]     = core_req_if[i].data;
        assign core_req_tag[i]      = core_req_if[i].tag;
        assign core_req_if[i].ready = core_req_ready[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    // Core response buffering
    wire [NUM_REQS-1:0]                  core_rsp_valid_s;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_s;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   core_rsp_tag_s;
    wire [NUM_REQS-1:0]                  core_rsp_ready_s;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_generic_buffer #(
            .DATAW   (`WORD_WIDTH + TAG_WIDTH),
            .SKID    ((NC_BYPASS && !DIRECT_PASSTHRU) ? (CORE_OUT_REG >> 1) : 0),
            .OUT_REG ((NC_BYPASS && !DIRECT_PASSTHRU) ? (CORE_OUT_REG & 1) : 0)
        ) core_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_rsp_valid_s[i]),
            .ready_in  (core_rsp_ready_s[i]),
            .data_in   ({core_rsp_data_s[i], core_rsp_tag_s[i]}),
            .data_out  ({core_rsp_if[i].data, core_rsp_if[i].tag}), 
            .valid_out (core_rsp_if[i].valid),
            .ready_out (core_rsp_if[i].ready)
        );
    end

    ///////////////////////////////////////////////////////////////////////////

    // Memory request buffering
    wire                             mem_req_valid_s;
    wire                             mem_req_rw_s;
    wire [LINE_SIZE-1:0]             mem_req_byteen_s;   
    wire [`MEM_ADDR_WIDTH-1:0]       mem_req_addr_s;
    wire [`LINE_WIDTH-1:0]           mem_req_data_s;
    wire [MEM_TAG_WIDTH-1:0]         mem_req_tag_s;
    wire                             mem_req_ready_s;

    VX_generic_buffer #(
        .DATAW   (1 + LINE_SIZE + `MEM_ADDR_WIDTH + `LINE_WIDTH + MEM_TAG_WIDTH),
        .SKID    ((NC_BYPASS && !DIRECT_PASSTHRU) ? (MEM_OUT_REG >> 1) : 0),
        .OUT_REG ((NC_BYPASS && !DIRECT_PASSTHRU) ? (MEM_OUT_REG & 1) : 0)
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid_s),        
        .ready_in  (mem_req_ready_s),      
        .data_in   ({mem_req_rw_s,  mem_req_byteen_s,  mem_req_addr_s,  mem_req_data_s,  mem_req_tag_s}),
        .data_out  ({mem_req_if.rw, mem_req_if.byteen, mem_req_if.addr, mem_req_if.data, mem_req_if.tag}),        
        .valid_out (mem_req_if.valid),        
        .ready_out (mem_req_if.ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Core request    
    wire [NUM_REQS-1:0]                     core_req_valid_b;
    wire [NUM_REQS-1:0]                     core_req_rw_b;
    wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr_b;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]      core_req_byteen_b;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]    core_req_data_b;
    wire [NUM_REQS-1:0][CORE_TAG_X_WIDTH-1:0] core_req_tag_b;
    wire [NUM_REQS-1:0]                     core_req_ready_b;

    // Core response
    wire [NUM_REQS-1:0]                     core_rsp_valid_b;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]    core_rsp_data_b;
    wire [NUM_REQS-1:0][CORE_TAG_X_WIDTH-1:0] core_rsp_tag_b;
    wire [NUM_REQS-1:0]                     core_rsp_ready_b;

    // Memory request
    wire                            mem_req_valid_b;
    wire                            mem_req_rw_b;
    wire [`MEM_ADDR_WIDTH-1:0]      mem_req_addr_b;
    wire [LINE_SIZE-1:0]            mem_req_byteen_b;
    wire [`LINE_WIDTH-1:0]          mem_req_data_b;
    wire [MEM_TAG_X_WIDTH-1:0]      mem_req_tag_b;
    wire                            mem_req_ready_b;
    
    // Memory response
    wire                            mem_rsp_valid_b;
    wire [`LINE_WIDTH-1:0]          mem_rsp_data_b;
    wire [MEM_TAG_X_WIDTH-1:0]      mem_rsp_tag_b;
    wire                            mem_rsp_ready_b;

    if (NC_BYPASS) begin
       
        `RESET_RELAY (nc_bypass_reset, reset);

        VX_nc_bypass #(
            .NUM_REQS          (NUM_REQS),
            .NC_TAG_BIT        (NC_TAG_BIT),

            .NC_ENABLE         (NC_ENABLE),
            .PASSTHRU          (PASSTHRU),

            .CORE_ADDR_WIDTH   (`WORD_ADDR_WIDTH),
            .CORE_DATA_SIZE    (WORD_SIZE),    
            .CORE_TAG_IN_WIDTH (TAG_WIDTH),
                
            .MEM_ADDR_WIDTH    (`MEM_ADDR_WIDTH),
            .MEM_DATA_SIZE     (LINE_SIZE),
            .MEM_TAG_IN_WIDTH  (MEM_TAG_X_WIDTH),
            .MEM_TAG_OUT_WIDTH (MEM_TAG_WIDTH),

            .UUID_WIDTH        (UUID_WIDTH)
        ) nc_bypass (
            .clk                (clk),
            .reset              (nc_bypass_reset),

            // Core request in
            .core_req_valid_in  (core_req_valid),
            .core_req_rw_in     (core_req_rw),
            .core_req_byteen_in (core_req_byteen),
            .core_req_addr_in   (core_req_addr),
            .core_req_data_in   (core_req_data),        
            .core_req_tag_in    (core_req_tag),
            .core_req_ready_in  (core_req_ready),

            // Core request out
            .core_req_valid_out (core_req_valid_b),
            .core_req_rw_out    (core_req_rw_b),
            .core_req_byteen_out(core_req_byteen_b),
            .core_req_addr_out  (core_req_addr_b),
            .core_req_data_out  (core_req_data_b),        
            .core_req_tag_out   (core_req_tag_b),
            .core_req_ready_out (core_req_ready_b),

            // Core response in
            .core_rsp_valid_in  (core_rsp_valid_b),
            .core_rsp_data_in   (core_rsp_data_b),
            .core_rsp_tag_in    (core_rsp_tag_b),
            .core_rsp_ready_in  (core_rsp_ready_b),

            // Core response out
            .core_rsp_valid_out (core_rsp_valid_s),
            .core_rsp_data_out  (core_rsp_data_s),
            .core_rsp_tag_out   (core_rsp_tag_s),
            .core_rsp_ready_out (core_rsp_ready_s),

            // Memory request in
            .mem_req_valid_in   (mem_req_valid_b),
            .mem_req_rw_in      (mem_req_rw_b),  
            .mem_req_addr_in    (mem_req_addr_b),
            .mem_req_byteen_in  (mem_req_byteen_b),
            .mem_req_data_in    (mem_req_data_b),
            .mem_req_tag_in     (mem_req_tag_b),
            .mem_req_ready_in   (mem_req_ready_b),

            // Memory request out
            .mem_req_valid_out  (mem_req_valid_s),
            .mem_req_addr_out   (mem_req_addr_s),
            .mem_req_rw_out     (mem_req_rw_s),
            .mem_req_byteen_out (mem_req_byteen_s),
            .mem_req_data_out   (mem_req_data_s),
            .mem_req_tag_out    (mem_req_tag_s),
            .mem_req_ready_out  (mem_req_ready_s),

            // Memory response in
            .mem_rsp_valid_in   (mem_rsp_if.valid),        
            .mem_rsp_data_in    (mem_rsp_if.data),
            .mem_rsp_tag_in     (mem_rsp_if.tag),
            .mem_rsp_ready_in   (mem_rsp_if.ready),

            // Memory response out
            .mem_rsp_valid_out  (mem_rsp_valid_b),        
            .mem_rsp_data_out   (mem_rsp_data_b),
            .mem_rsp_tag_out    (mem_rsp_tag_b),
            .mem_rsp_ready_out  (mem_rsp_ready_b)
        );
    end else begin        
        assign core_req_valid_b = core_req_valid;
        assign core_req_rw_b    = core_req_rw;
        assign core_req_addr_b  = core_req_addr;
        assign core_req_byteen_b= core_req_byteen;
        assign core_req_data_b  = core_req_data;
        assign core_req_tag_b   = core_req_tag;
        assign core_req_ready   = core_req_ready_b;

        assign core_rsp_valid_s = core_rsp_valid_b;
        assign core_rsp_data_s  = core_rsp_data_b;
        assign core_rsp_tag_s   = core_rsp_tag_b;
        assign core_rsp_ready_b = core_rsp_ready_s;

        assign mem_req_valid_s  = mem_req_valid_b;
        assign mem_req_addr_s   = mem_req_addr_b;
        assign mem_req_rw_s     = mem_req_rw_b;
        assign mem_req_byteen_s = mem_req_byteen_b;
        assign mem_req_data_s   = mem_req_data_b;
        assign mem_req_ready_b  = mem_req_ready_s;

        // Add explicit NC=0 flag to the memory request tag

        VX_bits_insert #( 
            .N   (MEM_TAG_WIDTH-1),
            .POS (NC_TAG_BIT)
        ) mem_req_tag_insert (
            .data_in  (mem_req_tag_b),
            .sel_in   (1'b0),
            .data_out (mem_req_tag_s)
        );

        assign mem_rsp_valid_b  = mem_rsp_if.valid;
        assign mem_rsp_data_b   = mem_rsp_if.data;
        assign mem_rsp_if.ready = mem_rsp_ready_b;

        // Remove NC flag from the memory response tag

        VX_bits_remove #( 
            .N   (MEM_TAG_WIDTH),
            .POS (NC_TAG_BIT)
        ) mem_rsp_tag_remove (
            .data_in  (mem_rsp_if.tag),
            .data_out (mem_rsp_tag_b)
        );
    end 

    if (PASSTHRU != 0) begin

        `UNUSED_VAR (core_req_valid_b)
        `UNUSED_VAR (core_req_rw_b)
        `UNUSED_VAR (core_req_addr_b)
        `UNUSED_VAR (core_req_byteen_b)
        `UNUSED_VAR (core_req_data_b)
        `UNUSED_VAR (core_req_tag_b)
        assign core_req_ready_b = 0;

        assign core_rsp_valid_b = '0;
        assign core_rsp_data_b  = 'x;
        assign core_rsp_tag_b   = 'x;
        `UNUSED_VAR (core_rsp_ready_b)

        assign mem_req_valid_b  = 0;
        assign mem_req_addr_b   = 'x;
        assign mem_req_rw_b     = 'x;
        assign mem_req_byteen_b = 'x;
        assign mem_req_data_b   = 'x;
        assign mem_req_tag_b    = 'x;
        `UNUSED_VAR (mem_req_ready_b)

        `UNUSED_VAR (mem_rsp_valid_b)
        `UNUSED_VAR (mem_rsp_data_b)
        `UNUSED_VAR (mem_rsp_tag_b)
        assign mem_rsp_ready_b = 0;

    `ifdef PERF_ENABLE
        assign perf_cache_if.reads        = 'x;
        assign perf_cache_if.writes       = 'x;
        assign perf_cache_if.read_misses  = 'x;
        assign perf_cache_if.write_misses = 'x;
        assign perf_cache_if.bank_stalls  = 'x;
        assign perf_cache_if.mshr_stalls  = 'x;
        assign perf_cache_if.mem_stalls   = 'x;
        assign perf_cache_if.crsp_stalls  = 'x;
    `endif

    end else begin

        VX_mem_req_if #(
            .DATA_WIDTH (`WORD_WIDTH),
            .TAG_WIDTH  (CORE_TAG_X_WIDTH)
        ) core_req_wrap_if[NUM_REQS]();
        
        VX_mem_rsp_if #(
            .DATA_WIDTH (`WORD_WIDTH),
            .TAG_WIDTH  (CORE_TAG_X_WIDTH)
        ) core_rsp_wrap_if[NUM_REQS]();

        VX_mem_req_if #(
            .DATA_WIDTH (`LINE_WIDTH), 
            .TAG_WIDTH  (MEM_TAG_X_WIDTH)
        ) mem_req_wrap_if();

        VX_mem_rsp_if #(
            .DATA_WIDTH (`LINE_WIDTH), 
            .TAG_WIDTH  (MEM_TAG_X_WIDTH)
        ) mem_rsp_wrap_if();

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_wrap_if[i].valid  = core_req_valid_b[i];
            assign core_req_wrap_if[i].rw     = core_req_rw_b[i];
            assign core_req_wrap_if[i].addr   = core_req_addr_b[i];
            assign core_req_wrap_if[i].byteen = core_req_byteen_b[i];
            assign core_req_wrap_if[i].data   = core_req_data_b[i];
            assign core_req_wrap_if[i].tag    = core_req_tag_b[i];
            assign core_req_ready_b[i] = core_req_wrap_if[i].ready;
        end

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_valid_b[i] = core_rsp_wrap_if[i].valid;
            assign core_rsp_data_b[i]  = core_rsp_wrap_if[i].data;
            assign core_rsp_tag_b[i]   = core_rsp_wrap_if[i].tag;
            assign core_rsp_wrap_if[i].ready = core_rsp_ready_b[i];
        end

        assign mem_req_valid_b  = mem_req_wrap_if.valid;
        assign mem_req_addr_b   = mem_req_wrap_if.addr;
        assign mem_req_rw_b     = mem_req_wrap_if.rw;
        assign mem_req_byteen_b = mem_req_wrap_if.byteen;
        assign mem_req_data_b   = mem_req_wrap_if.data;
        assign mem_req_tag_b    = mem_req_wrap_if.tag;
        assign mem_req_wrap_if.ready = mem_req_ready_b;

        assign mem_rsp_wrap_if.valid = mem_rsp_valid_b;
        assign mem_rsp_wrap_if.data  = mem_rsp_data_b;
        assign mem_rsp_wrap_if.tag   = mem_rsp_tag_b;
        assign mem_rsp_ready_b = mem_rsp_wrap_if.ready;

        `RESET_RELAY (cache_reset, reset);

        VX_cache #(
            .INSTANCE_ID  (INSTANCE_ID),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .NUM_PORTS    (NUM_PORTS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .CREQ_SIZE    (CREQ_SIZE),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MRSQ_SIZE    (MRSQ_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .WRITE_ENABLE (WRITE_ENABLE),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (CORE_TAG_X_WIDTH),
            .CORE_OUT_REG (NC_BYPASS ? 1 : CORE_OUT_REG),
            .MEM_OUT_REG  (NC_BYPASS ? 1 : MEM_OUT_REG)
        ) cache (
            .clk            (clk),
            .reset          (cache_reset),

        `ifdef PERF_ENABLE
            .perf_cache_if  (perf_cache_if),
        `endif

            .core_req_if    (core_req_wrap_if),
            .core_rsp_if    (core_rsp_wrap_if),
            .mem_req_if     (mem_req_wrap_if),
            .mem_rsp_if     (mem_rsp_wrap_if)
        );
        
    end

`ifdef DBG_TRACE_CACHE_BANK

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        wire [`UP(UUID_WIDTH)-1:0] core_req_uuid;
        wire [`UP(UUID_WIDTH)-1:0] core_rsp_uuid;

        `ASSIGN_REQ_UUID (core_req_uuid, core_req_if[i].tag)
        `ASSIGN_REQ_UUID (core_rsp_uuid, core_rsp_if[i].tag)

        wire core_req_fire = core_req_if[i].valid && core_req_if[i].ready;
        wire core_rsp_fire = core_rsp_if[i].valid && core_rsp_if[i].ready;

        always @(posedge clk) begin
            if (core_req_fire) begin
                if (core_req_if[i].rw)
                    `TRACE(1, ("%d: %s core-wr-req: tid=%0d, addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, i, `TO_FULL_ADDR(core_req_if[i].addr), core_req_if[i].tag, core_req_if[i].byteen, core_req_if[i].data, core_req_uuid));
                else
                    `TRACE(1, ("%d: %s core-rd-req: tid=%0d, addr=0x%0h, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID, i, `TO_FULL_ADDR(core_req_if[i].addr), core_req_if[i].tag, core_req_uuid));
            end
            if (core_rsp_fire) begin
                `TRACE(1, ("%d: %s core-rd-rsp: tid=%0d, tag=0x%0h, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, i, core_rsp_if[i].tag, core_rsp_if[i].data, core_rsp_uuid));
            end        
        end
    end   

    wire [`UP(UUID_WIDTH)-1:0] mem_req_uuid;
    wire [`UP(UUID_WIDTH)-1:0] mem_rsp_uuid;

    if ((UUID_WIDTH != 0) && (NC_ENABLE || PASSTHRU)) begin
        assign mem_req_uuid = mem_req_if.tag[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_rsp_uuid = mem_rsp_if.tag[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign mem_req_uuid = 0;
        assign mem_rsp_uuid = 0;
    end

    wire mem_req_fire = mem_req_if.valid && mem_req_if.ready;
    wire mem_rsp_fire = mem_rsp_if.valid && mem_rsp_if.ready;

    always @(posedge clk) begin
        if (mem_req_fire) begin
            if (mem_req_if.rw)
                `TRACE(1, ("%d: %s mem-wr-req: addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, `TO_FULL_ADDR(mem_req_if.addr), mem_req_if.tag, mem_req_if.byteen, mem_req_if.data, mem_req_uuid));
            else
                `TRACE(1, ("%d: %s mem-rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID, `TO_FULL_ADDR(mem_req_if.addr), mem_req_if.tag, mem_req_uuid));
        end
        if (mem_rsp_fire) begin
            `TRACE(1, ("%d: %s mem-rd-rsp: tag=0x%0h, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, mem_rsp_if.tag, mem_rsp_if.data, mem_rsp_uuid));
        end
    end    
`endif

endmodule
