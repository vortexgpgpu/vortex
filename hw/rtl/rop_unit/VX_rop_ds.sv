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
    input wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_func,    
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zpass,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zfail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_fail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_ref,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_mask,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_writemask,

    // Input values
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]     depth_ref,
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]     depth_val,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_val,    

    // Output values
    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]    depth_out,        
    output wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]  stencil_out,
    output wire [NUM_LANES-1:0]                         pass_out
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

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] depth_write, depth_write_s;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign depth_write[i] = (dpass[i] & depth_writemask) ? depth_ref[i] : depth_val[i];
    end

    // Stencil Test ///////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0] spass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_compare #(
            .DATAW (`ROP_STENCIL_BITS)
        ) rop_compare_stencil (
            .func   (stencil_func[i]),
            .a      (stencil_ref[i] & stencil_mask[i]),
            .b      (stencil_val[i] & stencil_mask[i]),
            .result (spass[i])
        );
    end    
    
    wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_op, stencil_op_s;
    wire [NUM_LANES-1:0] pass, pass_s;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign stencil_op[i] = spass[i] ? (dpass[i] ? stencil_zpass[i] : stencil_zfail[i]) : stencil_fail[i];
    end    

    assign pass = spass & dpass;

    wire valid_in_s;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_ref_s;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_val_s;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_writemask_s;
    wire [TAG_WIDTH-1:0]  tag_in_s;
    
    VX_pipe_register #(
        .DATAW	(1 + NUM_LANES * (`ROP_STENCIL_OP_BITS + 3 * `ROP_STENCIL_BITS + `ROP_DEPTH_BITS + 1) + TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in,   stencil_op,   stencil_ref,   stencil_val,   stencil_writemask,   depth_write,   pass,   tag_in}),
        .data_out ({valid_in_s, stencil_op_s, stencil_ref_s, stencil_val_s, stencil_writemask_s, depth_write_s, pass_s, tag_in_s})
    );

    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_result;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_stencil_op #(
            .DATAW (`ROP_STENCIL_BITS)
        ) rop_stencil_op (
            .stencil_op     (stencil_op_s[i]),
            .stencil_ref    (stencil_ref_s[i]),
            .stencil_val    (stencil_val_s[i]),
            .stencil_result (stencil_result[i])
        );
    end

    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] stencil_write;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < `ROP_STENCIL_BITS; ++j) begin
            assign stencil_write[i][j] = stencil_writemask_s[i][j] ? stencil_result[i][j] : stencil_val_s[i][j];
        end
    end

    // Output /////////////////////////////////////////////////////////////////

    VX_pipe_register #(
        .DATAW	(1 + TAG_WIDTH + NUM_LANES * (`ROP_DEPTH_BITS + `ROP_STENCIL_BITS + 1)),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s, tag_in_s,  depth_write_s, stencil_write, pass_s}),
        .data_out ({valid_out,  tag_out,   depth_out,     stencil_out,   pass_out})
    );

endmodule
