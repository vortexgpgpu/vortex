// Quad evaluator
// Functionality: Receives a 2x2 quads
//      check whether primitive overlaps each quad

`include "VX_raster_define.vh"

module VX_raster_qe #(
    parameter string INSTANCE_ID = "",
    parameter NUM_QUADS = 4
) (
    input wire clk,
    input wire reset, 

    // Device configurations
    raster_dcrs_t dcrs,

    output wire                                         empty,

    input wire                                          enable,   
    
    // Inputs    
    input wire                                          valid_in,
    input wire [`RASTER_PID_BITS-1:0]                   pid_in,
    input wire [NUM_QUADS-1:0][`RASTER_DIM_BITS-1:0]    x_loc_in,
    input wire [NUM_QUADS-1:0][`RASTER_DIM_BITS-1:0]    y_loc_in,
    input wire [`RASTER_DIM_BITS-1:0]                   x_max_in,
    input wire [`RASTER_DIM_BITS-1:0]                   y_min_in,
    input wire [`RASTER_DIM_BITS-1:0]                   y_max_in,
    input wire [NUM_QUADS-1:0][2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,

    // Outputs
    output wire [NUM_QUADS-1:0]                         valid_out,
    output wire [`RASTER_PID_BITS-1:0]                  pid_out,    
    output wire [NUM_QUADS-1:0][3:0]                    mask_out,    
    output wire [NUM_QUADS-1:0][`RASTER_DIM_BITS-1:0]   x_loc_out,
    output wire [NUM_QUADS-1:0][`RASTER_DIM_BITS-1:0]   y_loc_out,    
    output wire [NUM_QUADS-1:0][2:0][3:0][`RASTER_DATA_BITS-1:0] bcoords_out
);
    `UNUSED_VAR (dcrs)

    wire [NUM_QUADS-1:0] valid;
    wire [NUM_QUADS-1:0][2:0][3:0][`RASTER_DATA_BITS-1:0] edge_eval;
    wire [NUM_QUADS-1:0][3:0] overlap;

     // Check if primitive overlaps current quad
    for (genvar q = 0; q < NUM_QUADS; ++q) begin        
        for (genvar i = 0; i < 2; ++i) begin
            for (genvar j = 0; j < 2; ++j) begin            
                for (genvar k = 0; k < 3; ++k) begin
                    assign edge_eval[q][k][2 * j + i] = i * edges_in[q][k][0] + j * edges_in[q][k][1] + edges_in[q][k][2];
                end    
                wire [`RASTER_DIM_BITS-1:0] quad_x = x_loc_in[q] | i;
                wire [`RASTER_DIM_BITS-1:0] quad_y = y_loc_in[q] | j;
                assign overlap[q][2 * j + i] = ~(edge_eval[q][0][2 * j + i][`RASTER_DATA_BITS-1] 
                                              || edge_eval[q][1][2 * j + i][`RASTER_DATA_BITS-1] 
                                              || edge_eval[q][2][2 * j + i][`RASTER_DATA_BITS-1])
                                           && (quad_x < x_max_in)
                                           && (quad_y >= y_min_in)
                                           && (quad_y < y_max_in);
            end
        end

        assign valid[q] = valid_in && (| overlap[q]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_QUADS + `RASTER_PID_BITS + NUM_QUADS * (4 + 2 * `RASTER_DIM_BITS + 4 * 3 * `RASTER_DATA_BITS)),
        .RESETW (1 + NUM_QUADS)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({~valid_in, valid,     pid_in,  overlap,  x_loc_in,  y_loc_in,  edge_eval}),
        .data_out ({empty,     valid_out, pid_out, mask_out, x_loc_out, y_loc_out, bcoords_out})
    );

endmodule
