// Block level evaluator
// Functionality: Receives a block of AxA (where A is pow(2))
//     1. Breaks it into quad and runs quad evaluators on it
//     2. Stores the result in quad queues
//     3. Queues direction read as outputs

`include "VX_raster_define.vh"

module VX_raster_be #(
    parameter BLOCK_SIZE       = 4,
    parameter OUTPUT_QUADS = 2,
    parameter QUAD_FIFO_DEPTH  = 16,
    parameter SLICE_ID                = 1
) (
    // Standard inputs
    input wire clk,
    input wire  reset,
    input wire input_valid, // to indicate current input is a valid update
    input wire pop,         // to fetch data from the quad queue
    
    output wire empty,      // to indicate no data left in data queue
    output wire ready,      // to indicate it has sent all previous quad data

    // Block related input data
    input wire         [`RASTER_DIM_BITS-1:0]              x_loc, y_loc,
    // edge equation data for the 3 edges and ax+by+c
    input wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edges[2:0][2:0],
    input wire         [`RASTER_PRIMITIVE_DATA_BITS-1:0]   pid,
    // edge function computation value propagated
    input wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edge_func_val[2:0],
    // Rendering region
    input wire         [`RASTER_DIM_BITS-1:0]              dst_width, dst_height,

    // Quad related output data
    output wire        [`RASTER_DIM_BITS-1:0]              out_quad_x_loc[OUTPUT_QUADS-1:0],
    output wire        [`RASTER_DIM_BITS-1:0]              out_quad_y_loc[OUTPUT_QUADS-1:0],
    output wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_pid[OUTPUT_QUADS-1:0],
    output wire        [3:0]                               out_quad_masks[OUTPUT_QUADS-1:0],
    output wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_quad_bcoords[OUTPUT_QUADS-1:0][2:0][3:0],
    output wire        [OUTPUT_QUADS-1:0]       valid
);

    // Local parameter setup
    localparam RASTER_QUAD_NUM           = BLOCK_SIZE/2;
    localparam RASTER_QUAD_SPACE         = RASTER_QUAD_NUM*RASTER_QUAD_NUM;
    localparam RASTER_QUAD_ARBITER_RANGE = RASTER_QUAD_SPACE/OUTPUT_QUADS + 1;
    localparam ARBITER_BITS              = `LOG2UP(RASTER_QUAD_ARBITER_RANGE) + 1;
    localparam QE_LATENCY                = 2; // Decided based on the number of pipe stages in quad evaluator

    // Temporary (temp_) for combinatorial part, quad_ register for data storage
    wire        [`RASTER_DIM_BITS-1:0]             temp_quad_x_loc     [RASTER_QUAD_SPACE-1:0];
    reg        [`RASTER_DIM_BITS-1:0]             quad_x_loc          [RASTER_QUAD_SPACE-1:0];
    wire        [`RASTER_DIM_BITS-1:0]             temp_quad_y_loc     [RASTER_QUAD_SPACE-1:0];
    reg        [`RASTER_DIM_BITS-1:0]             quad_y_loc          [RASTER_QUAD_SPACE-1:0];
    wire        [3:0]                              temp_quad_masks     [RASTER_QUAD_SPACE-1:0];
    reg        [3:0]                               quad_masks          [RASTER_QUAD_SPACE-1:0];
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  temp_quad_bcoords   [RASTER_QUAD_SPACE-1:0][2:0][3:0];
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  quad_bcoords        [RASTER_QUAD_SPACE-1:0][2:0][3:0];

    // Wire to hold the edge function values for quad evaluation
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  local_edge_func_val [RASTER_QUAD_SPACE-1:0][2:0];

    // Status signal to log if it is working on valid data
    reg valid_data;
    // Fifo and arbiter signals
    wire full, push;
    reg [ARBITER_BITS-1:0] arbiter_index;
    // Per fifo signals
    wire [OUTPUT_QUADS-1:0] full_flag, empty_flag, push_flag, pop_flag;

    // FSM to track BE run start to end
    wire fsm_complete;

    // Generate the RASTER_QUAD_NUM x RASTER_QUAD_NUM quad evaluators

    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            assign temp_quad_x_loc[i*RASTER_QUAD_NUM+j] = x_loc + `RASTER_DIM_BITS'(i*2);
            assign temp_quad_y_loc[i*RASTER_QUAD_NUM+j] = y_loc + `RASTER_DIM_BITS'(j*2);
        end
    end
    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign local_edge_func_val[i*RASTER_QUAD_NUM+j][k] = edge_func_val[k] + i*2*edges[k][0] + j*2*edges[k][1];
            end
        end
    end

    localparam PIPE_REG_1_WIDTH = 2*`RASTER_DIM_BITS + 3*3*`RASTER_PRIMITIVE_DATA_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS;
    wire [RASTER_QUAD_NUM*RASTER_QUAD_NUM*PIPE_REG_1_WIDTH-1:0] pipe_reg_1_in, pipe_reg_1_r;


    wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edges_r[RASTER_QUAD_NUM-1:0][RASTER_QUAD_NUM-1:0][2:0][2:0];
    wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edge_func_val_r[RASTER_QUAD_NUM-1:0][RASTER_QUAD_NUM-1:0][2:0];
    wire         [`RASTER_DIM_BITS-1:0]                  x_loc_r[RASTER_QUAD_NUM-1:0][RASTER_QUAD_NUM-1:0], y_loc_r[RASTER_QUAD_NUM-1:0][RASTER_QUAD_NUM-1:0];

    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            assign pipe_reg_1_in[(i*RASTER_QUAD_NUM+j)*PIPE_REG_1_WIDTH+:PIPE_REG_1_WIDTH] =
                {
                    temp_quad_x_loc[i*RASTER_QUAD_NUM+j], temp_quad_y_loc[i*RASTER_QUAD_NUM+j],
                    local_edge_func_val[i*RASTER_QUAD_NUM+j][0], local_edge_func_val[i*RASTER_QUAD_NUM+j][1], local_edge_func_val[i*RASTER_QUAD_NUM+j][2],
                    edges[0][0], edges[0][1], edges[0][2],
                    edges[1][0], edges[1][1], edges[1][2],
                    edges[2][0], edges[2][1], edges[2][2]
                };
            
            assign {
                x_loc_r[i][j], y_loc_r[i][j],
                edge_func_val_r[i][j][0], edge_func_val_r[i][j][1], edge_func_val_r[i][j][2],
                edges_r[i][j][0][0], edges_r[i][j][0][1], edges_r[i][j][0][2],
                edges_r[i][j][1][0], edges_r[i][j][1][1], edges_r[i][j][1][2],
                edges_r[i][j][2][0], edges_r[i][j][2][1], edges_r[i][j][2][2]
            } = pipe_reg_1_r[(i*RASTER_QUAD_NUM+j)*PIPE_REG_1_WIDTH+:PIPE_REG_1_WIDTH];
        end
    end

    VX_pipe_register #(
        .DATAW  (RASTER_QUAD_NUM*RASTER_QUAD_NUM*PIPE_REG_1_WIDTH),
        .RESETW (1)
    ) be_pipe_reg_1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (fsm_complete),
        .data_in  ({
            pipe_reg_1_in
        }),
        .data_out ({
            pipe_reg_1_r
        })
    );

    reg [`RASTER_PRIMITIVE_DATA_BITS-1:0] quad_pid [RASTER_QUAD_SPACE-1:0];
    wire [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_pid_r;
    wire input_valid_r;
    
    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            wire [`RASTER_DIM_BITS-1:0] qe_x_loc_out, qe_y_loc_out;
            VX_raster_qe #(
                .SLICE_ID       (SLICE_ID),
                .QUAD_ID        (i*RASTER_QUAD_NUM+j)
            ) qe (
                .clk            (clk),
                .reset          (reset),
                .enable         (fsm_complete),
                .edges          (edges_r[i][j]),
                .edge_func_val  (edge_func_val_r[i][j]),
                .dst_width      (dst_width),
                .dst_height     (dst_height),
                .x_loc          (x_loc_r[i][j]),
                .y_loc          (y_loc_r[i][j]),
                .x_loc_o        (qe_x_loc_out),
                .y_loc_o        (qe_y_loc_out),
                .masks_o        (temp_quad_masks[i*RASTER_QUAD_NUM+j]),
                .bcoords_o      (temp_quad_bcoords[i*RASTER_QUAD_NUM+j]),
                .out_enable     (input_valid_r == 1 && fsm_complete == 1)
            );
            // Save the temp data into quad registers to prevent overwrite by redundant data
            always_comb begin
                quad_x_loc[i*RASTER_QUAD_NUM+j]   = qe_x_loc_out;
                quad_y_loc[i*RASTER_QUAD_NUM+j]   = qe_y_loc_out;
                quad_masks[i*RASTER_QUAD_NUM+j]   = temp_quad_masks[i*RASTER_QUAD_NUM+j];
                quad_bcoords[i*RASTER_QUAD_NUM+j] = temp_quad_bcoords[i*RASTER_QUAD_NUM+j];
                quad_pid[i*RASTER_QUAD_NUM+j]     = temp_pid_r;
            end
        end
    end
    
    VX_shift_register #(
        .DATAW  (1 + `RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (1),
        .DEPTH  (QE_LATENCY)
    ) be_pipe_reg_2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (fsm_complete),
        .data_in  ({input_valid, pid}),
        .data_out ({input_valid_r, temp_pid_r})
    );

    // Simple arbiter implementation
    always @(posedge clk) begin
        // Reset condition
        if (reset == 1) begin
            arbiter_index <= RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0] - 1;
            valid_data    <= 0;
        end
        // Initialization condition
        else if (input_valid_r == 1 && fsm_complete == 1) begin
            arbiter_index <= 0;
            valid_data    <= 1;
        end
        // Arbitration condition
        else if (full == 0 && push == 1)
            arbiter_index <= arbiter_index + ARBITER_BITS'(1);
        else if (ready)
            valid_data    <= 0;
    end

    assign push = (arbiter_index < (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0])) && !full && !reset && valid_data;
    assign fsm_complete = (
        arbiter_index > (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0]-1) ||
        arbiter_index == (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0]-1)
    );
    assign ready = fsm_complete && !full;

    localparam FIFO_DATA_WIDTH = 2*`RASTER_DIM_BITS + 4 + `RASTER_PRIMITIVE_DATA_BITS*3*4 + 
        `RASTER_PRIMITIVE_DATA_BITS + 1;
    // Generate the required number of FIFOs
    for (genvar i = 0; i < OUTPUT_QUADS; ++i) begin
        // Quad queue
        wire [FIFO_DATA_WIDTH-1:0] fifo_push_data, fifo_pop_data;
        //assign push_flag[i] = push && quad_masks[arbiter_index*OUTPUT_QUADS + i] != 0 && !full;
        assign push_flag[i] = push && !full;
        assign pop_flag[i]  = pop & !empty_flag[i];
        assign fifo_push_data = (arbiter_index*OUTPUT_QUADS + i) < RASTER_QUAD_SPACE ?
            {
                quad_x_loc[arbiter_index*OUTPUT_QUADS + i] >> 1,
                quad_y_loc[arbiter_index*OUTPUT_QUADS + i] >> 1,
                quad_masks[arbiter_index*OUTPUT_QUADS + i],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][0][0],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][0][1],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][0][2],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][0][3],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][1][0],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][1][1],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][1][2],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][1][3],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][2][0],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][2][1],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][2][2],
                quad_bcoords[arbiter_index*OUTPUT_QUADS + i][2][3],
                quad_pid    [arbiter_index*OUTPUT_QUADS + i],
                (valid_data)
            } : {FIFO_DATA_WIDTH{1'b0}};

        wire fifo_valid;
        assign {out_quad_x_loc[i], out_quad_y_loc[i], out_quad_masks[i],
            out_quad_bcoords[i][0][0], out_quad_bcoords[i][0][1], out_quad_bcoords[i][0][2], out_quad_bcoords[i][0][3],
            out_quad_bcoords[i][1][0], out_quad_bcoords[i][1][1], out_quad_bcoords[i][1][2], out_quad_bcoords[i][1][3],
            out_quad_bcoords[i][2][0], out_quad_bcoords[i][2][1], out_quad_bcoords[i][2][2], out_quad_bcoords[i][2][3],
            out_pid[i], fifo_valid} = fifo_pop_data;
        assign valid[i] = fifo_valid && !empty_flag[i];
        VX_fifo_queue #(
            .DATAW	    (FIFO_DATA_WIDTH),
            .SIZE       (QUAD_FIFO_DEPTH),
            .OUT_REG    (1)
        ) quad_fifo_queue (
            .clk        (clk),
            .reset      (reset),
            .push       (push_flag[i]),
            .pop        (pop_flag[i]),
            .data_in    (fifo_push_data),
            .data_out   (fifo_pop_data),
            .full       (full_flag[i]),
            .empty      (empty_flag[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );
    end

    assign full = |(full_flag);
    assign empty = &(empty_flag);

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (input_valid) begin
            dpi_trace(2, "%d: raster-block-in: x=%0d, y=%0d, pid=%0d, dst_width=%0d, dst_height=%0d, edge1={%0d, %0d, %0d}, edge2={%0d, %0d, %0d}, edge3={%0d, %0d, %0d}, edge_func_val=%0d %0d %0d\n",
                $time, x_loc, y_loc, pid, dst_width, dst_height,
                edges[0][0], edges[0][1], edges[0][2],
                edges[1][0], edges[1][1], edges[1][2],
                edges[2][0], edges[2][1], edges[2][2],
                edge_func_val[0], edge_func_val[1], edge_func_val[2]);
        end
    end
    always @(posedge clk) begin
        if (pop) begin
            for (int i = 0; i < OUTPUT_QUADS; ++i) begin
                if (valid[i]) begin
                    dpi_trace(2, "%d: raster-be-out[%0d]: x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={%d %d %d %d, %d %d %d %d, %d %d %d %d}\n",
                        $time, i, out_quad_x_loc[i], out_quad_y_loc[i], out_quad_masks[i], out_pid[i],
                        out_quad_bcoords[i][0][0], out_quad_bcoords[i][0][1], out_quad_bcoords[i][0][2], out_quad_bcoords[i][0][3],
                        out_quad_bcoords[i][1][0], out_quad_bcoords[i][1][1], out_quad_bcoords[i][1][2], out_quad_bcoords[i][1][3],
                        out_quad_bcoords[i][2][0], out_quad_bcoords[i][2][1], out_quad_bcoords[i][2][2], out_quad_bcoords[i][2][3]);
                end
            end
        end
    end
`endif

endmodule
