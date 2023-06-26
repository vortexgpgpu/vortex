`include "VX_raster_define.vh"

module VX_raster_unit #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1, 
    parameter NUM_SLICES      = 1,  // number of slices
    parameter TILE_LOGSIZE    = 5,  // tile log size
    parameter BLOCK_LOGSIZE   = 2,  // block log size
    parameter MEM_FIFO_DEPTH  = 4,  // memory queue size
    parameter QUAD_FIFO_DEPTH = 4,  // quad queue size
    parameter OUTPUT_QUADS    = 4   // number of output quads    
) (
    `SCOPE_IO_DECL

    // Clock
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_raster_perf_if.master perf_raster_if,
`endif

    // Memory interface
    VX_cache_bus_if.master  cache_bus_if,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Outputs
    VX_raster_bus_if.master raster_bus_if
);
    localparam EDGE_FUNC_LATENCY = `LATENCY_IMUL;
    localparam SLICES_BITS = $clog2(NUM_SLICES+1);

    // A primitive data contains (xloc, yloc, pid, edges, extents)
    localparam PRIM_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS;

    `STATIC_ASSERT(TILE_LOGSIZE > BLOCK_LOGSIZE, ("invalid parameter"))

    // DCRs

    raster_dcrs_t raster_dcrs;

    VX_raster_dcr #(
        .INSTANCE_ID (INSTANCE_ID)
    ) raster_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .raster_dcrs(raster_dcrs)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Output from the request
    wire [`VX_RASTER_DIM_BITS-1:0] mem_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0] mem_yloc;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] mem_edges;
    wire [`VX_RASTER_PID_BITS-1:0] mem_pid;
    
    // Memory unit status
    reg running;
    wire mem_unit_busy;
    wire mem_unit_valid;    
    wire mem_unit_ready;

    `RESET_RELAY (mem_reset, reset);

    // Generate start pulse
    always @(posedge clk) begin
        running <= ~mem_reset;
    end
    wire mem_unit_start = ~mem_reset && ~running;

    // Memory unit
    VX_raster_mem #(
        .INSTANCE_ID   (INSTANCE_ID),
        .INSTANCE_IDX  (INSTANCE_IDX),
        .NUM_INSTANCES (NUM_INSTANCES),
        .TILE_LOGSIZE  (TILE_LOGSIZE),
        .QUEUE_SIZE    (MEM_FIFO_DEPTH)
    ) raster_mem (
        .clk          (clk),
        .reset        (mem_reset),

        .start        (mem_unit_start),        
        .busy         (mem_unit_busy),

        .dcrs         (raster_dcrs),

        .cache_bus_if (cache_bus_if),

        .valid_out    (mem_unit_valid),
        .xloc_out     (mem_xloc),
        .yloc_out     (mem_yloc),
        .edges_out    (mem_edges),
        .pid_out      (mem_pid),
        .ready_out    (mem_unit_ready)
    );

    // Edge function and extents calculation

    wire [2:0][`RASTER_DATA_BITS-1:0] edge_eval;
    wire [2:0][`RASTER_DATA_BITS-1:0] mem_extents;
    wire edge_func_stall;

    VX_raster_extents #(
        .TILE_LOGSIZE (TILE_LOGSIZE)
    ) raster_extents (
        .edges   (mem_edges),
        .extents (mem_extents)
    );

    VX_raster_edge #(
        .LATENCY (EDGE_FUNC_LATENCY)
    ) raster_edge (
        .clk    (clk),
        .reset  (reset),
        .enable (~edge_func_stall),
        .xloc   (mem_xloc),
        .yloc   (mem_yloc),
        .edges  (mem_edges),
        .result (edge_eval)
    );

    wire                            slice_arb_valid_in;  
    wire [`VX_RASTER_DIM_BITS-1:0]  slice_arb_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0]  slice_arb_yloc;
    wire [`VX_RASTER_PID_BITS-1:0]  slice_arb_pid;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_arb_edges, slice_arb_edges_e;
    wire [2:0][`RASTER_DATA_BITS-1:0] slice_arb_extents;
    wire                            slice_arb_ready_in;

    VX_shift_register #(
        .DATAW  (1 + 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS),
        .DEPTH  (EDGE_FUNC_LATENCY),
        .RESETW (1)
    ) edge_func_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~edge_func_stall),
        .data_in  ({mem_unit_valid,     mem_xloc,       mem_yloc,       mem_pid,       mem_edges,       mem_extents}),
        .data_out ({slice_arb_valid_in, slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges, slice_arb_extents})
    );

    `EDGE_UPDATE (slice_arb_edges_e, slice_arb_edges, edge_eval);

    assign edge_func_stall = slice_arb_valid_in && ~slice_arb_ready_in;

    assign mem_unit_ready = ~edge_func_stall;

    wire [NUM_SLICES-1:0] slice_arb_valid_out;    
    wire [NUM_SLICES-1:0][PRIM_DATA_WIDTH-1:0] slice_arb_data_out;
    wire [NUM_SLICES-1:0] slice_arb_ready_out;

    VX_stream_arb #(
        .NUM_OUTPUTS (NUM_SLICES),
        .DATAW       (PRIM_DATA_WIDTH),
        .ARBITER     ("R"),
        .BUFFERED    (1)       
    ) slice_req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (slice_arb_valid_in),
        .ready_in   (slice_arb_ready_in),
        .data_in    ({slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges_e, slice_arb_extents}),
        .data_out   (slice_arb_data_out),        
        .valid_out  (slice_arb_valid_out),        
        .ready_out  (slice_arb_ready_out)
    );

    // track pending tile data 
    // this is needed to determine when rasterization has completed

    wire no_pending_tiledata;
    wire mem_unit_fire = mem_unit_valid && mem_unit_ready;
    wire [NUM_SLICES-1:0] slice_arb_fire_out = slice_arb_valid_out & slice_arb_ready_out;
    wire [SLICES_BITS-1:0] slice_arb_fire_out_cnt;
    
    `POP_COUNT(slice_arb_fire_out_cnt, slice_arb_fire_out);

    VX_pending_size #( 
        .SIZE  (EDGE_FUNC_LATENCY + 2 * NUM_SLICES),
        .DECRW (SLICES_BITS)
    ) pending_slice_inputs (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_unit_fire),
        .decr  (slice_arb_fire_out_cnt),
        .empty (no_pending_tiledata),
        `UNUSED_PIN (size),
        `UNUSED_PIN (full)
    );

    wire has_pending_inputs = mem_unit_start
                           || mem_unit_busy 
                           || mem_unit_valid 
                           || ~no_pending_tiledata;

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) slice_raster_bus_if[NUM_SLICES]();

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) raster_bus_tmp_if[1]();

    wire [NUM_SLICES-1:0] slice_valid_in;
    wire [NUM_SLICES-1:0] slice_busy_out;
    wire [NUM_SLICES-1:0] slice_valid_out;

    // Generate all slices
    for (genvar i = 0; i < NUM_SLICES; ++i) begin
        wire [`VX_RASTER_DIM_BITS-1:0] slice_xloc_in;
        wire [`VX_RASTER_DIM_BITS-1:0] slice_yloc_in;
        wire [`VX_RASTER_PID_BITS-1:0] slice_pid_in;
        wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_edges_in;
        wire [2:0][`RASTER_DATA_BITS-1:0] slice_extents_in;
        wire slice_ready_in;

        assign slice_valid_in[i] = slice_arb_valid_out[i];
        assign {slice_xloc_in, slice_yloc_in, slice_pid_in, slice_edges_in, slice_extents_in} = slice_arb_data_out[i];
        assign slice_arb_ready_out[i] = slice_ready_in;

        `RESET_RELAY (slice_reset, reset);

        VX_raster_slice #(
            .INSTANCE_ID     (INSTANCE_ID),            
            .TILE_LOGSIZE    (TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
            .OUTPUT_QUADS    (OUTPUT_QUADS),
            .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH)
        ) raster_slice (
            .clk        (clk),
            .reset      (slice_reset),

            .dcrs       (raster_dcrs),

            .valid_in   (slice_valid_in[i]),
            .xloc_in    (slice_xloc_in),
            .yloc_in    (slice_yloc_in),
            .xmin_in    (raster_dcrs.dst_xmin),
            .xmax_in    (raster_dcrs.dst_xmax),
            .ymin_in    (raster_dcrs.dst_ymin),
            .ymax_in    (raster_dcrs.dst_ymax),
            .edges_in   (slice_edges_in),
            .pid_in     (slice_pid_in),
            .extents_in (slice_extents_in),
            .ready_in   (slice_ready_in),

            .valid_out  (slice_valid_out[i]),
            .stamps_out (slice_raster_bus_if[i].req_stamps),
            .busy_out   (slice_busy_out[i]),
            .ready_out  (slice_raster_bus_if[i].req_ready)
        );

        assign slice_raster_bus_if[i].req_done = running
                                              && ~has_pending_inputs
                                              && ~(| slice_valid_in)
                                              && ~(| slice_busy_out)
                                              && ~(| slice_valid_out);

        assign slice_raster_bus_if[i].req_valid = slice_valid_out[i] 
                                               || slice_raster_bus_if[i].req_done;
    end

    `RESET_RELAY (raster_arb_reset, reset);

    VX_raster_arb #(
        .NUM_INPUTS (NUM_SLICES),
        .NUM_LANES  (OUTPUT_QUADS),
        .ARBITER    ("R"),
        .BUFFERED   (2)
    ) raster_arb (
        .clk        (clk),
        .reset      (raster_arb_reset),
        .bus_in_if  (slice_raster_bus_if),
        .bus_out_if (raster_bus_tmp_if)
    );

    `ASSIGN_VX_RASTER_BUS_IF (raster_bus_if, raster_bus_tmp_if[0]);

`ifdef DBG_SCOPE_RASTER
    if (INSTANCE_ID == "cluster0-raster0") begin
    `ifdef SCOPE
        wire cache_req_fire = cache_bus_if.req_valid && cache_bus_if.req_ready;
        wire cache_rsp_fire = cache_bus_if.rsp_valid && cache_bus_if.rsp_ready;
        wire raster_req_fire = raster_bus_if.req_valid && raster_bus_if.req_ready;
        VX_scope_tap #(
            .SCOPE_ID (4),
            .TRIGGERW (9),
            .PROBEW   (76)
        ) scope_tap (
            .clk(clk),
            .reset(scope_reset),
            .start(1'b0),
            .stop(1'b0),
            .triggers({
                reset,
                cache_req_fire,
                cache_rsp_fire,
                raster_req_fire,
                mem_unit_busy,
                mem_unit_ready,
                mem_unit_start,
                mem_unit_valid,
                raster_bus_if.req_done
            }),
            .probes({
                cache_bus_if.rsp_data,
                cache_bus_if.rsp_tag,
                cache_bus_if.req_tag,
                cache_bus_if.req_addr,
                cache_bus_if.req_rw,
                no_pending_tiledata
            }),
            .bus_in(scope_bus_in),
            .bus_out(scope_bus_out)
        );
    `endif
    `ifdef CHIPSCOPE
        ila_raster ila_raster_inst (
            .clk    (clk),
            .probe0 ({cache_bus_if.rsp_data, cache_bus_if.rsp_tag, cache_bus_if.rsp_ready, cache_bus_if.rsp_valid, cache_bus_if.req_tag, cache_bus_if.req_addr, cache_bus_if.req_rw, cache_bus_if.req_valid, cache_bus_if.req_ready}),
            .probe1 ({no_pending_tiledata, mem_unit_busy, mem_unit_ready, mem_unit_start, mem_unit_valid, raster_bus_if.req_done, raster_bus_if.req_valid, raster_bus_if.req_ready})
        );
    `endif
    end
`endif

`ifdef PERF_ENABLE
    wire [$clog2(RCACHE_NUM_REQS+1)-1:0] perf_mem_req_per_cycle;
    wire [$clog2(RCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [$clog2(RCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [RCACHE_NUM_REQS-1:0] perf_mem_req_fire = cache_bus_if.req_valid & cache_bus_if.req_ready;
    wire [RCACHE_NUM_REQS-1:0] perf_mem_rsp_fire = cache_bus_if.rsp_valid & cache_bus_if.rsp_ready;

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_fire);
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_fire);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = raster_bus_if.req_valid && ~raster_bus_if.req_ready && ~raster_bus_if.req_done;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + `PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_raster_if.mem_reads    = perf_mem_reads;
    assign perf_raster_if.mem_latency  = perf_mem_latency;
    assign perf_raster_if.stall_cycles = perf_stall_cycles;
`endif

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_bus_if.req_valid && raster_bus_if.req_ready) begin
            for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
                `TRACE(1, ("%d: %s-out[%0d]: done=%b, x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                    $time, INSTANCE_ID, i, raster_bus_if.req_done,
                    raster_bus_if.req_stamps[i].pos_x, raster_bus_if.req_stamps[i].pos_y, raster_bus_if.req_stamps[i].mask, raster_bus_if.req_stamps[i].pid,
                    raster_bus_if.req_stamps[i].bcoords[0][0], raster_bus_if.req_stamps[i].bcoords[1][0], raster_bus_if.req_stamps[i].bcoords[2][0], 
                    raster_bus_if.req_stamps[i].bcoords[0][1], raster_bus_if.req_stamps[i].bcoords[1][1], raster_bus_if.req_stamps[i].bcoords[2][1], 
                    raster_bus_if.req_stamps[i].bcoords[0][2], raster_bus_if.req_stamps[i].bcoords[1][2], raster_bus_if.req_stamps[i].bcoords[2][2], 
                    raster_bus_if.req_stamps[i].bcoords[0][3], raster_bus_if.req_stamps[i].bcoords[1][3], raster_bus_if.req_stamps[i].bcoords[2][3]));
            end
        end
    end
`endif

endmodule

///////////////////////////////////////////////////////////////////////////////

module VX_raster_unit_top #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1, 
    parameter NUM_SLICES      = 1,  // number of slices
    parameter TILE_LOGSIZE    = 5,  // tile log size
    parameter BLOCK_LOGSIZE   = 2,  // block log size
    parameter MEM_FIFO_DEPTH  = 8,  // memory queue size
    parameter QUAD_FIFO_DEPTH = 8,  // quad queue size
    parameter OUTPUT_QUADS    = 4   // number of output quads    
) (
    input wire                              clk,
    input wire                              reset,
    
    input wire                              dcr_write_valid,
    input wire [`VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [`VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    output wire                             raster_req_valid,  
    output raster_stamp_t [OUTPUT_QUADS-1:0] raster_req_stamps,
    output wire                             raster_req_done,    
    input wire                              raster_req_ready,

    output wire [RCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [RCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [RCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [RCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [RCACHE_NUM_REQS-1:0][RCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [RCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_raster_perf_if perf_raster_if();

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.write_valid = dcr_write_valid;
    assign dcr_bus_if.write_addr = dcr_write_addr;
    assign dcr_bus_if.write_data = dcr_write_data;

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) raster_bus_if();

    assign raster_req_valid = raster_bus_if.req_valid;
    assign raster_req_stamps = raster_bus_if.req_stamps;
    assign raster_bus_if.req_done = raster_req_done;
    assign raster_bus_if.req_ready = raster_req_ready;

    VX_cache_bus_if #(
        .NUM_REQS  (RCACHE_NUM_REQS), 
        .WORD_SIZE (RCACHE_WORD_SIZE), 
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) cache_bus_if();

    assign cache_req_valid = cache_bus_if.req_valid;
    assign cache_req_rw = cache_bus_if.req_rw;
    assign cache_req_byteen = cache_bus_if.req_byteen;
    assign cache_req_addr = cache_bus_if.req_addr;
    assign cache_req_data = cache_bus_if.req_data;
    assign cache_req_tag = cache_bus_if.req_tag;
    assign cache_bus_if.req_ready = cache_req_ready;

    assign cache_bus_if.rsp_valid = cache_rsp_valid;
    assign cache_bus_if.rsp_tag = cache_rsp_tag;
    assign cache_bus_if.rsp_data = cache_rsp_data;
    assign cache_rsp_ready = cache_bus_if.rsp_ready;

`ifdef SCOPE
    wire [0:0] scope_reset_w = 1'b0; 
    wire [0:0] scope_bus_in_w = 1'b0; 
    wire [0:0] scope_bus_out_w;
    `UNUSED_VAR (scope_bus_out_w)
`endif

    VX_raster_unit #( 
        .INSTANCE_ID     (INSTANCE_ID),
        .INSTANCE_IDX    (INSTANCE_IDX),
        .NUM_INSTANCES   (NUM_INSTANCES),
        .NUM_SLICES      (NUM_SLICES),
        .TILE_LOGSIZE    (TILE_LOGSIZE),
        .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
        .MEM_FIFO_DEPTH  (MEM_FIFO_DEPTH),
        .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH),
        .OUTPUT_QUADS    (OUTPUT_QUADS)
    ) raster_unit (
        `SCOPE_IO_BIND (0)
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_raster_if(perf_raster_if),
    `endif 
        .dcr_bus_if    (dcr_bus_if),
        .raster_bus_if (raster_bus_if),
        .cache_bus_if  (cache_bus_if)
    );

endmodule
