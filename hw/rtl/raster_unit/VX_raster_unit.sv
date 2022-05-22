`include "VX_raster_define.vh"

module VX_raster_unit #(
    parameter CLUSTER_ID      = 0,
    parameter NUM_SLICES      = 1,  // number of raster slices
    parameter TILE_LOGSIZE    = 5,  // tile log size
    parameter BLOCK_LOGSIZE   = 2,  // block log size
    parameter MEM_FIFO_DEPTH  = 4,  // memory queue size
    parameter QUAD_FIFO_DEPTH = 4,  // quad queue size
    parameter OUTPUT_QUADS    = 4   // number of output quads
    
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_raster_perf_if.master raster_perf_if,
`endif

    // Memory interface
    VX_cache_req_if.master  cache_req_if,
    VX_cache_rsp_if.slave   cache_rsp_if,

    // Inputs
    VX_raster_dcr_if.slave  raster_dcr_if,

    // Outputs
    VX_raster_req_if.master raster_req_if
);
    localparam EDGE_FUNC_LATENCY = `LATENCY_IMUL;

    // A primitive data contains (x_loc, y_loc, pid, edges, extents)
    localparam PRIM_DATA_WIDTH = 2 * `RASTER_DIM_BITS + `RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS;

    `STATIC_ASSERT(TILE_LOGSIZE > BLOCK_LOGSIZE, ("invalid parameter"))
    
    raster_dcrs_t raster_dcrs;
    assign raster_dcrs = raster_dcr_if.data;
    `UNUSED_VAR (raster_dcrs)

    // Output from the request
    wire [`RASTER_DIM_BITS-1:0] mem_x_loc;
    wire [`RASTER_DIM_BITS-1:0] mem_y_loc;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] mem_edges;
    wire [`RASTER_PID_BITS-1:0] mem_pid;
    
    // Memory unit status
    reg mem_unit_start;
    wire mem_unit_busy;
    wire mem_unit_valid;    
    wire mem_unit_ready;

    // Start execution    
    always @(posedge clk) begin
        mem_unit_start <= reset;
    end

    // Memory unit
    VX_raster_mem #(
        .TILE_LOGSIZE (TILE_LOGSIZE),
        .QUEUE_SIZE   (MEM_FIFO_DEPTH)
    ) raster_mem (
        .clk          (clk),
        .reset        (reset),

        .start        (mem_unit_start),        
        .busy         (mem_unit_busy),

        .dcrs         (raster_dcrs),

        .cache_req_if (cache_req_if),
        .cache_rsp_if (cache_rsp_if), 

        .valid_out    (mem_unit_valid),
        .x_loc_out    (mem_x_loc),
        .y_loc_out    (mem_y_loc),
        .edges_out    (mem_edges),
        .pid_out      (mem_pid),
        .ready_out    (mem_unit_ready)
    );

    // Edge function and extents calculation

    wire [2:0][`RASTER_DATA_BITS-1:0] edge_eval;
    wire [2:0][`RASTER_DATA_BITS-1:0] mem_extents;
    wire edge_func_enable;

    VX_raster_extents #(
        .TILE_LOGSIZE (TILE_LOGSIZE)
    ) raster_extents (
        .edges   (mem_edges),
        .extents (mem_extents)
    );

    VX_raster_edge_function #(
        .LATENCY (EDGE_FUNC_LATENCY)
    ) raster_edge_function (
        .clk    (clk),
        .reset  (reset),
        .enable (edge_func_enable),
        .x_loc  (mem_x_loc),
        .y_loc  (mem_y_loc),
        .edges  (mem_edges),
        .result (edge_eval)
    );

    wire                         slice_valid;    
    wire [`RASTER_DIM_BITS-1:0]  slice_x_loc;
    wire [`RASTER_DIM_BITS-1:0]  slice_y_loc;
    wire [`RASTER_PID_BITS-1:0]  slice_pid;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_edges, slice_edges_e;
    wire [2:0][`RASTER_DATA_BITS-1:0] slice_extents;
    wire                         slice_ready;

    VX_shift_register #(
        .DATAW  (1 + 2 * `RASTER_DIM_BITS + `RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS),
        .DEPTH  (EDGE_FUNC_LATENCY),
        .RESETW (1)
    ) edge_func_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (edge_func_enable),
        .data_in  ({mem_unit_valid, mem_x_loc,   mem_y_loc,   mem_pid,   mem_edges,   mem_extents}),
        .data_out ({slice_valid,    slice_x_loc, slice_y_loc, slice_pid, slice_edges, slice_extents})
    );

    `EDGE_UPDATE (slice_edges_e, slice_edges, edge_eval);

    assign edge_func_enable = slice_ready || ~slice_valid;

    assign mem_unit_ready = edge_func_enable;

    wire [NUM_SLICES-1:0] slices_valid_in;    
    wire [NUM_SLICES-1:0][PRIM_DATA_WIDTH-1:0] slices_data_in;
    wire [NUM_SLICES-1:0] slices_ready_in;

    VX_stream_demux #(
        .NUM_REQS (NUM_SLICES),
        .DATAW    (PRIM_DATA_WIDTH),
        .BUFFERED (1)
    ) slice_req_demux (
        .clk        (clk),
        .reset      (reset),
        `UNUSED_PIN (sel_in),
        .valid_in   (slice_valid),
        .ready_in   (slice_ready),
        .data_in    ({slice_x_loc, slice_y_loc, slice_pid, slice_edges_e, slice_extents}),
        .data_out   (slices_data_in),        
        .valid_out  (slices_valid_in),        
        .ready_out  (slices_ready_in)
    );

    // track pending slice inputs 

    wire no_pending_slice_input;    

    wire mem_unit_fire = mem_unit_valid && mem_unit_ready;

    wire slices_input_fire = | (slices_valid_in & slices_ready_in);

    VX_pending_size #( 
        .SIZE (MEM_FIFO_DEPTH)
    ) pending_slice_inputs (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_unit_fire),
        .decr  (slices_input_fire),
        .empty (no_pending_slice_input),
        `UNUSED_PIN (size),
        `UNUSED_PIN (full)
    );

    VX_raster_req_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) per_slice_raster_req_if[NUM_SLICES]();

    // Generate all slices
    for (genvar i = 0; i < NUM_SLICES; ++i) begin
        wire [`RASTER_DIM_BITS-1:0]         slice_x_loc_in;
        wire [`RASTER_DIM_BITS-1:0]         slice_y_loc_in;
        wire [`RASTER_PID_BITS-1:0]         slice_pid_in;
        wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_edges_in;
        wire [2:0][`RASTER_DATA_BITS-1:0]   slice_extents_in;

        assign {slice_x_loc_in, slice_y_loc_in, slice_pid_in, slice_edges_in, slice_extents_in} = slices_data_in[i];

        VX_raster_slice #(
            .SLICE_ID        (i),            
            .TILE_LOGSIZE    (TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
            .OUTPUT_QUADS    (OUTPUT_QUADS),
            .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH)
        ) raster_slice (
            .clk        (clk),
            .reset      (reset),

            .dcrs       (raster_dcrs),

            .valid_in   (slices_valid_in[i]),
            .x_loc_in   (slice_x_loc_in),
            .y_loc_in   (slice_y_loc_in),
            .edges_in   (slice_edges_in),
            .pid_in     (slice_pid_in),
            .extents_in (slice_extents_in),
            .ready_in   (slices_ready_in[i]),

            .valid_out  (per_slice_raster_req_if[i].valid),
            .mask_out   (per_slice_raster_req_if[i].tmask),
            .stamps_out (per_slice_raster_req_if[i].stamps),
            .empty_out  (per_slice_raster_req_if[i].empty),
            .ready_out  (per_slice_raster_req_if[i].ready)
        );
    end
                     
    VX_raster_req_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) raster_req_tmp_if();

    VX_raster_req_mux #(
        .NUM_REQS  (NUM_SLICES),
        .NUM_LANES (OUTPUT_QUADS),
        .BUFFERED  ((NUM_SLICES > 1) ? 1 : 0)
    ) raster_req_mux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_slice_raster_req_if),
        .req_out_if (raster_req_tmp_if)
    );   

    wire no_slice_input = ~mem_unit_busy 
                       && ~mem_unit_valid 
                       && no_pending_slice_input;

    assign raster_req_if.valid  = raster_req_tmp_if.valid || raster_req_if.empty;
    assign raster_req_if.tmask  = raster_req_tmp_if.tmask;
    assign raster_req_if.stamps = raster_req_tmp_if.stamps;
    assign raster_req_if.empty  = raster_req_tmp_if.empty && no_slice_input;
    assign raster_req_tmp_if.ready = raster_req_if.ready;

`ifdef PERF_ENABLE
    wire [$clog2(`RCACHE_NUM_REQS+1)-1:0] perf_mem_req_per_cycle;
    wire [$clog2(`RCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [$clog2(`RCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [`RCACHE_NUM_REQS-1:0] perf_mem_req_per_req = cache_req_if.valid & cache_req_if.ready;
    wire [`RCACHE_NUM_REQS-1:0] perf_mem_rsp_per_req = cache_rsp_if.valid & cache_rsp_if.ready;

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_per_req);
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_per_req);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= 0;
        end else begin
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'(perf_pending_reads_cycle);
        end
    end

    wire perf_stall_cycle = raster_req_if.valid && ~raster_req_if.ready && ~raster_req_if.empty;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= 0;
            perf_mem_latency  <= 0;
            perf_stall_cycles <= 0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + `PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign raster_perf_if.mem_reads    = perf_mem_reads;
    assign raster_perf_if.mem_latency  = perf_mem_latency;
    assign raster_perf_if.stall_cycles = perf_stall_cycles;
`endif

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_req_if.ready && raster_req_if.valid) begin
            for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
                `TRACE(1, ("%d: raster-out[%0d]: empty=%b, x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                    $time, i, raster_req_if.empty,
                    raster_req_if.stamps[i].pos_x,  raster_req_if.stamps[i].pos_y, raster_req_if.stamps[i].mask, raster_req_if.stamps[i].pid,
                    raster_req_if.stamps[i].bcoords[0][0], raster_req_if.stamps[i].bcoords[0][1], raster_req_if.stamps[i].bcoords[0][2], 
                    raster_req_if.stamps[i].bcoords[1][0], raster_req_if.stamps[i].bcoords[1][1], raster_req_if.stamps[i].bcoords[1][2], 
                    raster_req_if.stamps[i].bcoords[2][0], raster_req_if.stamps[i].bcoords[2][1], raster_req_if.stamps[i].bcoords[2][2], 
                    raster_req_if.stamps[i].bcoords[3][0], raster_req_if.stamps[i].bcoords[3][1], raster_req_if.stamps[i].bcoords[3][2]));
            end
        end
    end
`endif

endmodule
