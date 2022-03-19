`include "VX_rop_define.vh"

module VX_rop_ds #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4,
    parameter TAG_WIDTH  = 1
) (
    input wire clk,
    input wire reset,   

    // Handshake
    input wire                  valid_in,
    input wire [TAG_WIDTH-1:0]  tag_in,
    output wire                 ready_in,   
     
    output wire                 valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    input wire                  ready_out,    

    // Configuration states
    input wire [`ROP_DEPTH_FUNC_BITS-1:0]               depth_func,
    input wire                                          depth_writemask,
    input wire [`ROP_DEPTH_FUNC_BITS-1:0]               stencil_func,    
    input wire [`ROP_STENCIL_OP_BITS-1:0]               stencil_zpass,
    input wire [`ROP_STENCIL_OP_BITS-1:0]               stencil_zfail,
    input wire [`ROP_STENCIL_OP_BITS-1:0]               stencil_fail,
    input wire [`ROP_STENCIL_BITS-1:0]                  stencil_ref,
    input wire [`ROP_STENCIL_BITS-1:0]                  stencil_mask,
    input wire [`ROP_STENCIL_BITS-1:0]                  stencil_writemask,

    // Input values
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]     depth_ref,
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]     depth_val,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_val,    

    // Output values
    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]    depth_out,        
    output wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]  stencil_out,
    output wire [NUM_LANES-1:0]                         test_out
); 
     wire stall = ~ready_out && valid_out;
    
    assign ready_in = ~stall;
    
    // Depth Test /////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0] dpass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_compare #(
            .DATAW     (`ROP_DEPTH_BITS)
        ) rop_compare_depth (
            .func   (depth_func),
            .a      (depth_ref[i]),
            .b      (depth_val[i]),
            .result (dpass[i])
        );
    end

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] depth_write;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign depth_write[i] = (dpass[i] & depth_writemask) ? depth_ref[i] : depth_val[i];
    end

    // Stencil Test ///////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0] spass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_compare #(
            .DATAW (`ROP_STENCIL_BITS)
        ) rop_compare_stencil (
            .func   (stencil_func),
            .a      (stencil_ref & stencil_mask),
            .b      (stencil_val[i] & stencil_mask),
            .result (spass[i])
        );
    end    
    
    wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_op;
                    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign stencil_op[i] = spass[i] ? (dpass[i] ? stencil_zpass : stencil_zfail) : stencil_fail;
    end    

    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stenil_result;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_stencil_op #(
            .DATAW (`ROP_STENCIL_BITS)
        ) rop_stencil_op (
            .stencil_op     (stencil_op[i]),
            .stencil_ref    (stencil_ref),
            .stencil_val    (stencil_val[i]),
            .stencil_result (stenil_result[i])
        );
    end

    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_write;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < `ROP_STENCIL_BITS; ++j) begin
            assign stencil_write[i][j] = stencil_writemask[j] ? stenil_result[i][j] : stencil_val[i][j];
        end
    end

    wire [NUM_LANES-1:0] test_result = spass & dpass;

    // Output /////////////////////////////////////////////////////////////////

    VX_pipe_register #(
        .DATAW	(1 + TAG_WIDTH + NUM_LANES * (`ROP_DEPTH_BITS + `ROP_STENCIL_BITS + 1)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in,  tag_in,  depth_write,  stencil_write, test_result}),
        .data_out ({valid_out, tag_out, depth_out,    stencil_out,    test_out})
    );

endmodule