`include "VX_raster_define.vh"

// Top unit for the raster unit
// Instantiates the following modules:
//      1. DCR connections
//      2. Requests switch
//      3. Raster slices
//      4. Response switch

module VX_raster_unit #(
    parameter CLUSTER_ID  = 0,
    parameter NUM_OUTPUTS = 4,               // number of output queues
    parameter NUM_SLICES = 1,                // number of raster slices
    parameter RASTER_TILE_SIZE = 16,         // tile size
    parameter RASTER_BLOCK_SIZE = 4,         // block size
    parameter RASTER_RS_SIZE = 8,            // Reservation station size
    parameter RASTER_QUAD_OUTPUT_RATE = 4,   // Rate output quad generation
    parameter RASTER_QUAD_FIFO_DEPTH  = 64,  // Quad fifo depth
    parameter RASTER_TILE_FIFO_DEPTH  = 32   // tile fifo depth
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

    // NECESSARY TO AVOID DEADLOCKS
    `STATIC_ASSERT(RASTER_TILE_FIFO_DEPTH >= (RASTER_TILE_SIZE*RASTER_TILE_SIZE)/(
        RASTER_BLOCK_SIZE*RASTER_BLOCK_SIZE) + 1, ("must be 0 or power of 2!"))

    localparam RASTER_SLICE_BITS = `LOG2UP(NUM_SLICES);

    raster_dcrs_t raster_dcrs;
    assign raster_dcrs = raster_dcr_if.data;

    // Output from the request
    logic [`RASTER_DIM_BITS-1:0]                   x_loc, y_loc;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]              edges[2:0][2:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]              pid;
    // Slice selected for tile
    logic [RASTER_SLICE_BITS-1:0]                       slice_index;

    // FSM to control the valid signals for the rest of the system
    reg raster_input_valid;
    always @(posedge clk) begin
        raster_input_valid <= 0;
        if (reset) begin
            raster_input_valid <= 1;
        end
    end

    // Mem to raster slice control signals
    logic mem_valid;
    logic [NUM_SLICES-1:0] raster_slice_ready;
    VX_raster_mem #(
        .RASTER_SLICE_NUM(NUM_SLICES),
        .RASTER_SLICE_BITS(RASTER_SLICE_BITS),
        .RASTER_TILE_SIZE(RASTER_TILE_SIZE),
        .RASTER_RS_SIZE(RASTER_RS_SIZE)
    ) raster_mem (
        .clk(clk),
        .reset(reset),
        .input_valid(raster_input_valid),
        .num_tiles(raster_dcrs.tile_count),
        .tbuf_baseaddr(raster_dcrs.tbuf_addr),
        .pbuf_baseaddr(raster_dcrs.pbuf_addr),
        .pbuf_stride(raster_dcrs.pbuf_stride),
        .raster_slice_ready(raster_slice_ready),
        .out_x_loc(x_loc),
        .out_y_loc(y_loc),
        .out_edges(edges),
        .out_pid(pid),
        .out_slice_index(slice_index),
        `UNUSED_PIN(ready),
        .out_valid(mem_valid),
        .cache_req_if(cache_req_if),
        .cache_rsp_if(cache_rsp_if)
    );

    // Complete the edge function values and extents
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] edge_func_val[2:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] extents [2:0];

    VX_raster_extents #(
        .RASTER_TILE_SIZE(RASTER_TILE_SIZE)
    ) raster_extents (
        .edges(edges),
        .extents(extents)
    );

    VX_raster_edge_functions raster_edge_function (
        .x_loc(x_loc),
        .y_loc(y_loc),
        .edges(edges),
        .edge_func_val(edge_func_val)
    );

    logic quad_valid[NUM_SLICES-1:0][RASTER_QUAD_OUTPUT_RATE-1:0];
    
    logic [`RASTER_DIM_BITS-1:0] temp_quad_x_loc[NUM_SLICES-1:0][RASTER_QUAD_OUTPUT_RATE-1:0],
        temp_quad_y_loc[NUM_SLICES-1:0][RASTER_QUAD_OUTPUT_RATE-1:0];
    logic [3:0] temp_quad_masks[NUM_SLICES-1:0][RASTER_QUAD_OUTPUT_RATE-1:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_quad_bcoords[NUM_SLICES-1:0][RASTER_QUAD_OUTPUT_RATE-1:0][2:0][3:0];
    logic quad_queue_empty[NUM_SLICES-1:0];
    logic quad_pop[NUM_SLICES-1:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_out_pid[NUM_SLICES-1:0];
    
    // TODO: Add raster slices in generate block here
    for (genvar i = 0; i < RASTER_SLICE_BITS; ++i) begin
        VX_raster_slice #(
            .RASTER_BLOCK_SIZE(RASTER_BLOCK_SIZE),
            .RASTER_TILE_SIZE(RASTER_TILE_SIZE),
            .RASTER_QUAD_OUTPUT_RATE(RASTER_QUAD_OUTPUT_RATE),
            .RASTER_QUAD_FIFO_DEPTH(RASTER_QUAD_FIFO_DEPTH),
            .RASTER_TILE_FIFO_DEPTH(RASTER_TILE_FIFO_DEPTH)
        ) raster_slice (
            .clk(clk),
            .reset(reset),
            // Input valid logic
            // 1. If memory data is valid
            // 2. If memory arbiter decides to assign data to this slice
            .input_valid(mem_valid && (i == slice_index)),
            .x_loc(x_loc),
            .y_loc(y_loc),
            .edges(edges),
            .pid(pid),
            .edge_func_val(edge_func_val),
            .extents(extents),
            // Pop quad only if the quad receiver outside the raster is ready
            .pop_quad(quad_pop[i] && arbiter_valid && raster_req_if.ready),
            .ready(raster_slice_ready[i]),
            .quad_queue_empty(quad_queue_empty[i]),
            .out_pid(temp_out_pid[i]),
            .out_quad_x_loc(temp_quad_x_loc[i]),
            .out_quad_y_loc(temp_quad_y_loc[i]),
            .out_quad_masks(temp_quad_masks[i]),
            .out_quad_bcoords(temp_quad_bcoords[i]),
            .valid(quad_valid[i])
        );
    end

    logic [`RASTER_DIM_BITS-1:0] out_quad_x_loc[RASTER_QUAD_OUTPUT_RATE-1:0];
    logic [`RASTER_DIM_BITS-1:0] out_quad_y_loc[RASTER_QUAD_OUTPUT_RATE-1:0];
    logic [3:0] out_quad_masks[RASTER_QUAD_OUTPUT_RATE-1:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] out_quad_bcoords[RASTER_QUAD_OUTPUT_RATE-1:0][2:0][3:0];
    logic arbiter_valid;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] out_pid;
    generate
        // add arbiter if # raster slice > 1
        if (NUM_SLICES > 1) begin
            logic quad_index[RASTER_SLICE_BITS-1:0];
            VX_fair_arbiter #(
                .NUM_REQS   (NUM_SLICES),
            ) tile_fifo_arbiter (
                .clk            (clk),
                .reset          (reset),
                .enable         (!(&quad_queue_empty)),
                .requests       (~quad_queue_empty),
                .grant_index    (quad_index),
                .grant_onehot   (quad_pop),
                .grant_valid    (arbiter_valid)
            );
            always_comb begin
                if (arbiter_valid) begin
                    out_quad_x_loc = temp_quad_x_loc[quad_index];
                    out_quad_y_loc = temp_quad_y_loc[quad_index];
                    out_quad_masks = temp_quad_masks[quad_index];
                    out_quad_bcoords = temp_quad_bcoords[quad_index];
                    out_pid          = temp_out_pid[quad_index];
                end
            end
        end
        else begin
            always_comb begin
                arbiter_valid = 1;
                if (!quad_queue_empty[0]) begin
                    quad_pop[0] = 1;
                    if (|quad_valid) begin
                        out_quad_x_loc = temp_quad_x_loc[0];
                        out_quad_y_loc = temp_quad_y_loc[0];
                        out_quad_masks = temp_quad_masks[0];
                        out_quad_bcoords = temp_quad_bcoords[0];
                        out_pid          = temp_out_pid[0];
                    end
                end
            end
        end
    endgenerate

    VX_raster_rsp_switch #(
    ) raster_rsp_switch (
        .valid(arbiter_valid),
        .empty((&raster_slice_ready) & raster_req_if.ready),
        // Quad data
        .x_loc(out_quad_x_loc),
        .y_loc(out_quad_y_loc),
        .masks(out_quad_masks),
        .bcoords(out_quad_bcoords),
        .pid(out_pid),
        .raster_req_if(raster_req_if)
    );

endmodule
