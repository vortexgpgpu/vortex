`include "VX_raster_define.vh"

module VX_raster_unit #(
    parameter CLUSTER_ID      = 0,
    parameter OUTPUT_QUADS    = 4,  // number of output quads
    parameter NUM_SLICES      = 1,  // number of raster slices
    parameter TILE_SIZE       = 64, // tile size
    parameter BLOCK_SIZE      = 4,  // block size
    parameter RS_SIZE         = 8,  // Reservation station size
    parameter TILE_FIFO_DEPTH = (TILE_SIZE * TILE_SIZE) / (BLOCK_SIZE * BLOCK_SIZE), // Tile fifo depth
    parameter QUAD_FIFO_DEPTH = 16  // Quad fifo depth
    
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_raster_perf_if.master raster_perf_if,
`endif

    // Memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_raster_dcr_if.slave  raster_dcr_if,
    VX_raster_req_if.master raster_req_if
);
    localparam MUL_LATENCY    = 3;
    localparam SLICE_BITS     = `LOG2UP(NUM_SLICES);
    localparam MEM_DELAY_BITS = `CLOG2(MUL_LATENCY+1);

    `STATIC_ASSERT(TILE_FIFO_DEPTH >= (TILE_SIZE * TILE_SIZE) / (BLOCK_SIZE * BLOCK_SIZE), ("invalid parameter"))
    
    raster_dcrs_t raster_dcrs;
    assign raster_dcrs = raster_dcr_if.data;
    `UNUSED_VAR (raster_dcrs)

    // Output from the request
    wire        [`RASTER_DIM_BITS-1:0]            x_loc;
    wire        [`RASTER_DIM_BITS-1:0]            y_loc;
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edges[2:0][2:0];
    wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0] pid;

    // Slice selected for tile
    wire [SLICE_BITS-1:0] slice_idx;

    // Top raster unit ready signal
    wire raster_unit_ready, raster_mem_ready;
    wire mem_valid;

    // FSM to control the valid signals for the rest of the system
    reg input_valid;
    always @(posedge clk) begin
        input_valid <= 0;
        if (reset) begin
            input_valid <= 1;
        end
    end

    // flag to denote that a valid raster mem data is being generated for the slice
    // use this flag to stop the memory from generating another data and sending    
    reg processing_mem_data;
    reg [MEM_DELAY_BITS-1:0] delay_ctr;
    
    // FSM to stop multiple memory responses to the slices while one data set is being processed
    always @(posedge clk) begin
        if (reset) begin
            processing_mem_data <= 0;
            delay_ctr <= '0;
        end else begin
            if (delay_ctr == MUL_LATENCY) begin
                processing_mem_data <= 0;
                delay_ctr <= '0;
            end else if (processing_mem_data) begin
                delay_ctr <= delay_ctr + MEM_DELAY_BITS'(1);
            end else if (mem_valid) begin
                processing_mem_data <= 1;
                delay_ctr <= delay_ctr + MEM_DELAY_BITS'(1);
            end
        end
    end

    // Mem to raster slice control signals
    wire [NUM_SLICES-1:0] raster_slice_ready;

    VX_raster_mem #(
        .NUM_SLICES (NUM_SLICES),
        .TILE_SIZE  (TILE_SIZE),
        .RS_SIZE    (RS_SIZE)
    ) raster_mem (
        .clk                (clk),
        .reset              (reset),
       
        .cache_req_if       (cache_req_if),
        .cache_rsp_if       (cache_rsp_if),

        .input_valid        (input_valid),
        .num_tiles          (raster_dcrs.tile_count),
        .tbuf_baseaddr      (raster_dcrs.tbuf_addr),
        .pbuf_baseaddr      (raster_dcrs.pbuf_addr),
        .pbuf_stride        (raster_dcrs.pbuf_stride),
        .raster_slice_ready (raster_slice_ready & {NUM_SLICES{!processing_mem_data}}),
        .out_x_loc          (x_loc),
        .out_y_loc          (y_loc),
        .out_edges          (edges),
        .out_pid            (pid),
        .out_slice_index    (slice_idx),
        .ready              (raster_mem_ready),
        .out_valid          (mem_valid)
    );

    // Complete the edge function values and extents
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edge_func_val [2:0];
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] extents [2:0];

    VX_raster_extents #(
        .TILE_SIZE (TILE_SIZE)
    ) raster_extents (
        .edges   (edges),
        .extents (extents)
    );

    VX_raster_edge_functions #(
        .MUL_LATENCY (MUL_LATENCY)
    ) raster_edge_function (
        .clk           (clk),
        .x_loc         (x_loc),
        .y_loc         (y_loc),
        .edges         (edges),
        .edge_func_val (edge_func_val)
    );

    wire                                          slice_valid;
    wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0] slice_pid;
    wire        [`RASTER_DIM_BITS-1:0]            slice_x_loc;
    wire        [`RASTER_DIM_BITS-1:0]            slice_y_loc;
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] slice_edges [2:0][2:0];    
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] slice_extents [2:0];
    wire        [SLICE_BITS-1:0]                  temp_slice_index;

    VX_shift_register #(
        .DATAW  (1 +  2*`RASTER_DIM_BITS + `RASTER_PRIMITIVE_DATA_BITS + SLICE_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS + 9*`RASTER_PRIMITIVE_DATA_BITS),
        .DEPTH  (MUL_LATENCY),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({
            mem_valid, x_loc, y_loc, pid, slice_idx,
            extents[0],  extents[1],  extents[2],
            edges[0][0], edges[0][1], edges[0][2],
            edges[1][0], edges[1][1], edges[1][2],
            edges[2][0], edges[2][1], edges[2][2]}),
        .data_out ({
            slice_valid, slice_x_loc, slice_y_loc, slice_pid, temp_slice_index,
            slice_extents[0],  slice_extents[1],  slice_extents[2],
            slice_edges[0][0], slice_edges[0][1], slice_edges[0][2],
            slice_edges[1][0], slice_edges[1][1], slice_edges[1][2],
            slice_edges[2][0], slice_edges[2][1], slice_edges[2][2]})
    );

    wire        [OUTPUT_QUADS-1:0] quad_valid [NUM_SLICES-1:0];
    wire        [NUM_SLICES-1:0] quad_queue_empty;
    reg         [NUM_SLICES-1:0] quad_pop;

    wire        [`RASTER_DIM_BITS-1:0]            temp_quad_x_loc [NUM_SLICES-1:0][OUTPUT_QUADS-1:0];
    wire        [`RASTER_DIM_BITS-1:0]            temp_quad_y_loc [NUM_SLICES-1:0][OUTPUT_QUADS-1:0];
    wire        [3:0]                             temp_quad_masks [NUM_SLICES-1:0][OUTPUT_QUADS-1:0];
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_quad_bcoords [NUM_SLICES-1:0][OUTPUT_QUADS-1:0][2:0][3:0];
    wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_out_pid [NUM_SLICES-1:0][OUTPUT_QUADS-1:0];    
    
    wire arbiter_valid;

    // Raster slices in generate block here
    for (genvar i = 0; i < NUM_SLICES; ++i) begin
        VX_raster_slice #(
            .SLICE_ID        (i),
            .BLOCK_SIZE      (BLOCK_SIZE),
            .TILE_SIZE       (TILE_SIZE),
            .OUTPUT_QUADS    (OUTPUT_QUADS),
            .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH),
            .TILE_FIFO_DEPTH (TILE_FIFO_DEPTH)
        ) raster_slice (
            .clk                    (clk),
            .reset                  (reset),
            // Input valid logic
            // 1. If memory data is valid
            // 2. If memory arbiter decides to assign data to this slice
            .input_valid            (slice_valid && (i == temp_slice_index)),
            .x_loc                  (slice_x_loc),
            .y_loc                  (slice_y_loc),
            .edges                  (slice_edges),
            .pid                    (slice_pid),
            .edge_func_val          (edge_func_val),
            .extents                (slice_extents),
            // Pop quad only if the quad receiver outside the raster is ready
            .pop_quad               (quad_pop[i] && arbiter_valid && raster_req_if.ready),
            .ready                  (raster_slice_ready[i]),
            .quad_queue_empty       (quad_queue_empty[i]),
            .out_pid                (temp_out_pid[i]),
            .out_quad_x_loc         (temp_quad_x_loc[i]),
            .out_quad_y_loc         (temp_quad_y_loc[i]),
            .out_quad_masks         (temp_quad_masks[i]),
            .out_quad_bcoords       (temp_quad_bcoords[i]),
            .valid                  (quad_valid[i]),
            .dst_width              (raster_dcrs.dst_width),
            .dst_height             (raster_dcrs.dst_height)
        );
    end

    reg        [OUTPUT_QUADS-1:0]                out_valid;
    reg        [`RASTER_DIM_BITS-1:0]            out_quad_x_loc [OUTPUT_QUADS-1:0];
    reg        [`RASTER_DIM_BITS-1:0]            out_quad_y_loc [OUTPUT_QUADS-1:0];
    reg        [3:0]                             out_quad_masks [OUTPUT_QUADS-1:0];
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] out_quad_bcoords [OUTPUT_QUADS-1:0][2:0][3:0];    
    reg        [`RASTER_PRIMITIVE_DATA_BITS-1:0] out_pid [OUTPUT_QUADS-1:0];

    // add arbiter if # raster slice > 1
    if (NUM_SLICES > 1) begin
        wire [SLICE_BITS-1:0] quad_index;

        VX_fair_arbiter #(
            .NUM_REQS       (NUM_SLICES)
        ) tile_fifo_arb (
            .clk            (clk),
            .reset          (reset),
            `UNUSED_PIN     (unlock),
            .requests       (~quad_queue_empty),
            .grant_index    (quad_index),
            .grant_onehot   (quad_pop),
            .grant_valid    (arbiter_valid)
        );
        
        always_comb begin
            if (arbiter_valid) begin                
                out_quad_x_loc   = temp_quad_x_loc[quad_index];
                out_quad_y_loc   = temp_quad_y_loc[quad_index];
                out_quad_masks   = temp_quad_masks[quad_index];
                out_quad_bcoords = temp_quad_bcoords[quad_index];
                out_pid          = temp_out_pid[quad_index];
                out_valid        = quad_valid[quad_index];
            end
        end
    end else begin
        assign arbiter_valid = !quad_queue_empty[0];
        
        always_comb begin
            quad_pop[0] = 0;            
            out_quad_x_loc   = temp_quad_x_loc[0];
            out_quad_y_loc   = temp_quad_y_loc[0];
            out_quad_masks   = temp_quad_masks[0];
            out_quad_bcoords = temp_quad_bcoords[0];
            out_pid          = temp_out_pid[0];
            out_valid        = quad_valid[0];
            if (!quad_queue_empty[0]) begin
                quad_pop[0]  = 1;
            end
        end
    end

    assign raster_unit_ready = (& raster_slice_ready) 
                            && raster_mem_ready 
                            && (& quad_queue_empty);

    VX_raster_rsp_switch #(
        .CLUSTER_ID   (CLUSTER_ID),
        .OUTPUT_QUADS (OUTPUT_QUADS)
    ) raster_rsp_switch (
        .valid          ((arbiter_valid && (| out_valid)) || raster_unit_ready),
        .empty          (raster_unit_ready),
        .x_loc          (out_quad_x_loc),
        .y_loc          (out_quad_y_loc),
        .masks          (out_quad_masks),
        .bcoords        (out_quad_bcoords),
        .pid            (out_pid),
        .raster_req_if  (raster_req_if)
    );

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_req_if.ready && raster_req_if.valid) begin
            for (int i = 0; i < OUTPUT_QUADS; ++i) begin
                dpi_trace(1, "%d: raster-out[%0d]: empty=%b, x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={%0d %0d %0d %0d, %0d %0d %0d %0d, %0d %0d %0d %0d}\n",
                    $time, i, raster_req_if.empty,
                    raster_req_if.stamps[i].pos_x,  raster_req_if.stamps[i].pos_y, raster_req_if.stamps[i].mask, raster_req_if.stamps[i].pid,
                    raster_req_if.stamps[i].bcoord_x[0], raster_req_if.stamps[i].bcoord_x[1], raster_req_if.stamps[i].bcoord_x[2], raster_req_if.stamps[i].bcoord_x[3],
                    raster_req_if.stamps[i].bcoord_y[0], raster_req_if.stamps[i].bcoord_y[1], raster_req_if.stamps[i].bcoord_y[2], raster_req_if.stamps[i].bcoord_y[3],
                    raster_req_if.stamps[i].bcoord_z[0], raster_req_if.stamps[i].bcoord_z[1], raster_req_if.stamps[i].bcoord_z[2], raster_req_if.stamps[i].bcoord_z[3]
                );
            end
        end
    end
`endif

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
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
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

endmodule
