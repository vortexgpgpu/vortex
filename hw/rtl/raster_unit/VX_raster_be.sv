// Block level evaluator
// Functionality: Receives a block of AxA (where A is pow(2))
//     1. Breaks it into quad and runs quad evaluators on it
//     2. Stores the result in quad queues
//     3. Queues direction read as outputs

`include "VX_raster_define.vh"

module VX_raster_be #(
    parameter RASTER_BLOCK_SIZE       = 4,
    parameter RASTER_QUAD_OUTPUT_RATE = 2,
    parameter RASTER_QUAD_FIFO_DEPTH  = 16
) (
    // Standard inputs
    input logic clk, reset,
    input logic input_valid, // to indicate current input is a valid update
    input logic pop,         // to fetch data from the quad queue
    
    output logic empty,      // to indicate no data left in data queue
    output logic ready,      // to indicate it has sent all previous quad data

    // Block related input data
    input logic         [`RASTER_DIM_BITS-1:0]              x_loc, y_loc,
    // edge equation data for the 3 edges and ax+by+c
    input logic signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edges[2:0][2:0],
    input logic         [`RASTER_PRIMITIVE_DATA_BITS-1:0]   pid,
    // edge function computation value propagated
    input logic signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edge_func_val[2:0],

    // Quad related output data
    output logic        [`RASTER_DIM_BITS-1:0]              out_quad_x_loc[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic        [`RASTER_DIM_BITS-1:0]              out_quad_y_loc[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic        [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_pid[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic        [3:0]                               out_quad_masks[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_quad_bcoords[RASTER_QUAD_OUTPUT_RATE-1:0][2:0][3:0],
    output logic        [RASTER_QUAD_OUTPUT_RATE-1:0]       valid
);

    // Local parameter setup
    localparam RASTER_QUAD_NUM           = RASTER_BLOCK_SIZE/2;
    localparam RASTER_QUAD_SPACE         = RASTER_QUAD_NUM*RASTER_QUAD_NUM;
    localparam RASTER_QUAD_ARBITER_RANGE = RASTER_QUAD_SPACE/RASTER_QUAD_OUTPUT_RATE + 1;
    localparam ARBITER_BITS              = `LOG2UP(RASTER_QUAD_ARBITER_RANGE) + 1;
    localparam QE_LATENCY                = 2;

    // Temporary (temp_) for combinatorial part, quad_ register for data storage
    logic        [`RASTER_DIM_BITS-1:0]             temp_quad_x_loc     [RASTER_QUAD_SPACE-1:0],
                                                    temp_quad_x_loc_r   [RASTER_QUAD_SPACE-1:0],
                                                    quad_x_loc          [RASTER_QUAD_SPACE-1:0];
    logic        [`RASTER_DIM_BITS-1:0]             temp_quad_y_loc     [RASTER_QUAD_SPACE-1:0],
                                                    temp_quad_y_loc_r   [RASTER_QUAD_SPACE-1:0],
                                                    quad_y_loc          [RASTER_QUAD_SPACE-1:0];
    logic        [3:0]                              temp_quad_masks     [RASTER_QUAD_SPACE-1:0], 
                                                    quad_masks          [RASTER_QUAD_SPACE-1:0];
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  temp_quad_bcoords   [RASTER_QUAD_SPACE-1:0][2:0][3:0],
                                                    quad_bcoords        [RASTER_QUAD_SPACE-1:0][2:0][3:0];

    // Wire to hold the edge function values for quad evaluation
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  local_edge_func_val [RASTER_QUAD_SPACE-1:0][2:0];

    // Status signal to log if it is working on valid data
    logic valid_data;
    // Fifo and arbiter signals
    logic full, push;
    logic [ARBITER_BITS-1:0] arbiter_index;
    // Per fifo signals
    logic [RASTER_QUAD_OUTPUT_RATE-1:0] full_flag, empty_flag, push_flag, pop_flag;

    // FSM to track BE run start to end
    logic fsm_complete;

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
    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            VX_raster_qe qe (
                .clk            (clk),
                .reset          (reset),
                .enable         (fsm_complete),
                .edges          (edges),
                .edge_func_val  (local_edge_func_val[i*RASTER_QUAD_NUM+j]),
                .masks          (temp_quad_masks[i*RASTER_QUAD_NUM+j]),
                .bcoords        (temp_quad_bcoords[i*RASTER_QUAD_NUM+j])
            );
        end
    end

    logic input_valid_r;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_pid_r;
    VX_shift_register #(
        .DATAW  (1 + `RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (1),
        .DEPTH  (QE_LATENCY)
    ) be_pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (fsm_complete),
        .data_in  ({input_valid, pid}),
        .data_out ({input_valid_r, temp_pid_r})
    );

    for (genvar i = 0; i < RASTER_QUAD_NUM; ++i) begin
        for (genvar j = 0; j < RASTER_QUAD_NUM; ++j) begin
            VX_shift_register #(
                .DATAW  (`RASTER_DIM_BITS),
                .RESETW (1),
                .DEPTH  (QE_LATENCY)
            ) be_pipe_reg2 (
                .clk      (clk),
                .reset    (reset),
                .enable   (fsm_complete),
                .data_in  (temp_quad_x_loc[i*RASTER_QUAD_NUM+j]),
                .data_out (temp_quad_x_loc_r[i*RASTER_QUAD_NUM+j])
            );
            VX_shift_register #(
                .DATAW  (`RASTER_DIM_BITS),
                .RESETW (1),
                .DEPTH  (QE_LATENCY)
            ) be_pipe_reg3 (
                .clk      (clk),
                .reset    (reset),
                .enable   (fsm_complete),
                .data_in  (temp_quad_y_loc[i*RASTER_QUAD_NUM+j]),
                .data_out (temp_quad_y_loc_r[i*RASTER_QUAD_NUM+j])
            );
        end
    end

    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] quad_pid [RASTER_QUAD_SPACE-1:0];
    // Store the temp results in registers
    for(genvar i = 0; i < RASTER_QUAD_SPACE; ++i) begin
        // Save the temp data into quad registers to prevent overwrite by redundant data
        always @(posedge clk) begin
            if (input_valid_r == 1 && fsm_complete == 1) begin // overwrite only the first time
                quad_x_loc[i]   <= temp_quad_x_loc_r[i];
                quad_y_loc[i]   <= temp_quad_y_loc_r[i];
                quad_masks[i]   <= temp_quad_masks[i];
                quad_bcoords[i] <= temp_quad_bcoords[i];
                quad_pid[i]     <= temp_pid_r;
            end
        end
    end

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

    /* verilator lint_off CMPCONST */
    /* verilator lint_off UNSIGNED */
    assign push = (arbiter_index < (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0])) && !full && !reset && valid_data;
    assign fsm_complete = (
        arbiter_index > (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0]-1) ||
        arbiter_index == (RASTER_QUAD_ARBITER_RANGE[ARBITER_BITS-1:0]-1)
    );
    assign ready = fsm_complete && !full;
    /* verilator lint_on CMPCONST */
    /* verilator lint_on UNSIGNED */

    localparam FIFO_DATA_WIDTH = 2*`RASTER_DIM_BITS + 4 + `RASTER_PRIMITIVE_DATA_BITS*3*4 + 
        `RASTER_PRIMITIVE_DATA_BITS + 1;
    // Generate the required number of FIFOs
    for (genvar i = 0; i < RASTER_QUAD_OUTPUT_RATE; ++i) begin
        // Quad queue
        logic [FIFO_DATA_WIDTH-1:0] fifo_push_data, fifo_pop_data;
        //assign push_flag[i] = push && quad_masks[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i] != 0 && !full;
        assign push_flag[i] = push && !full;
        assign pop_flag[i]  = pop & !empty_flag[i];
        assign fifo_push_data = (arbiter_index*RASTER_QUAD_OUTPUT_RATE + i) < RASTER_QUAD_SPACE ?
            {
                quad_x_loc[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i] >> 1,
                quad_y_loc[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i] >> 1,
                quad_masks[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][0][0],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][0][1],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][0][2],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][0][3],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][1][0],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][1][1],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][1][2],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][1][3],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][2][0],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][2][1],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][2][2],
                quad_bcoords[arbiter_index*RASTER_QUAD_OUTPUT_RATE + i][2][3],
                quad_pid    [arbiter_index*RASTER_QUAD_OUTPUT_RATE + i],
                (valid_data)
            } : {FIFO_DATA_WIDTH{1'b0}};

        logic fifo_valid;
        assign {out_quad_x_loc[i], out_quad_y_loc[i], out_quad_masks[i],
            out_quad_bcoords[i][0][0], out_quad_bcoords[i][0][1], out_quad_bcoords[i][0][2], out_quad_bcoords[i][0][3],
            out_quad_bcoords[i][1][0], out_quad_bcoords[i][1][1], out_quad_bcoords[i][1][2], out_quad_bcoords[i][1][3],
            out_quad_bcoords[i][2][0], out_quad_bcoords[i][2][1], out_quad_bcoords[i][2][2], out_quad_bcoords[i][2][3],
            out_pid[i], fifo_valid} = fifo_pop_data;
        assign valid[i] = fifo_valid && !empty_flag[i];
        VX_fifo_queue #(
            .DATAW	    (FIFO_DATA_WIDTH),
            .SIZE       (RASTER_QUAD_FIFO_DEPTH),
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
            dpi_trace(2, "%d: raster-block-in: x=%0d, y=%0d, pid=%0d, edge1.x=%0d, edge1.y=%0d, edge1.z=%0d, edge2.x=%0d, edge2.y=%0d, edge2.z=%0d, edge3.x=%0d, edge3.y=%0d, edge3.z=%0d, edge_func_val=%0d %0d %0d\n",
                $time, x_loc, y_loc, pid,
                edges[0][0], edges[0][1], edges[0][2],
                edges[1][0], edges[1][1], edges[1][2],
                edges[2][0], edges[2][1], edges[2][2],
                edge_func_val[0], edge_func_val[1], edge_func_val[2]);
        end
    end
    always @(posedge clk) begin
        if (pop) begin
            for (int i = 0; i < RASTER_QUAD_OUTPUT_RATE; ++i) begin
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
