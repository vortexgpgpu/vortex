`include "VX_rop_define.vh"

module VX_rop_ds #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4
) (
    input wire clk,
    input wire reset,

    // Depth Test
    input wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] depth_func,
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]      depth_ref,
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]      depth_val,
    input wire [NUM_LANES-1:0]                           depth_mask,

    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]     depth_result,

    // Stencil Test
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_val,
    input wire [NUM_LANES-1:0]                           is_backface,

    input wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_front_func,    
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_front_zpass,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_front_zfail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_front_fail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_front_mask,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_front_ref,
    input wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_back_func,    
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_back_zpass,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_back_zfail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_back_fail,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_back_mask,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_back_ref,

    output wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_result,
    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS+`ROP_STENCIL_BITS-1:0] mask_out

);

    reg [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]      dresult;

    reg [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_func;
    reg [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_op;
    reg [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_mask;
    reg [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_ref;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   sresult;

    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_ref_m;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]   stencil_val_m;

    reg  [NUM_LANES-1:0][`ROP_DEPTH_BITS+`ROP_STENCIL_BITS-1:0] stencil_write_mask;

    wire [NUM_LANES-1:0] dpass;
    wire [NUM_LANES-1:0] spass;

    ///////////////////////////////////////////////////////////////

    // Depth Test

    VX_rop_compare #(
        .NUM_LANES (NUM_LANES),
        .DATAW     (`ROP_DEPTH_BITS)
    ) depth_compare (
        .func   (depth_func),
        .a      (depth_ref),
        .b      (depth_val),
        .result (dpass)
    );

    always @(*) begin
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
             if (dpass[i] & depth_mask[i])
                dresult[i] = depth_ref[i];
            else    
                dresult[i] = depth_val[i];
        end
    end

    ///////////////////////////////////////////////////////////////

    // Stencil Test

     always @(*) begin
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
            stencil_func[i] = is_backface[i] ? stencil_back_func[i] : stencil_front_func[i];
            stencil_mask[i] = is_backface[i] ? stencil_back_mask[i] : stencil_front_mask[i];
            stencil_ref[i]  = is_backface[i] ? stencil_back_ref[i]  : stencil_front_ref[i];
        end
    end

    assign stencil_ref_m = stencil_ref & stencil_mask;
    assign stencil_val_m = stencil_val & stencil_mask;

    VX_rop_compare #(
        .NUM_LANES (NUM_LANES),
        .DATAW (`ROP_STENCIL_BITS)
    ) stencil_compare (
        .func   (stencil_func),
        .a      (stencil_ref_m),
        .b      (stencil_val_m),
        .result (spass)
    );

    always @(*) begin
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
            stencil_write_mask[i] = {stencil_mask[i], {`ROP_DEPTH_BITS{1'b0}}};
            if (spass[i]) begin
                if (dpass[i]) begin
                    stencil_write_mask[i] = stencil_write_mask[i] | {{`ROP_STENCIL_BITS{1'b0}}, {`ROP_DEPTH_BITS{depth_mask[i]}}};
                    stencil_op[i] = is_backface[i] ? stencil_back_zpass[i] : stencil_front_zpass[i];
                end else
                    stencil_op[i] = is_backface[i] ? stencil_back_zfail[i] : stencil_front_zfail[i];
            end else
                stencil_op[i] = is_backface[i] ? stencil_back_fail[i] : stencil_front_fail[i];
        end
    end

    VX_rop_stencil_op #(
        .NUM_LANES (NUM_LANES),
        .DATAW (8)
    ) stencil_op_ (
        .stencil_op     (stencil_op),
        .stencil_ref    (stencil_ref),
        .stencil_val    (stencil_val),
        .stencil_result (sresult)
    );

    ///////////////////////////////////////////////////////////////

    VX_pipe_register #(
        .DATAW	((`ROP_DEPTH_BITS + `ROP_STENCIL_BITS  + `ROP_DEPTH_BITS + `ROP_STENCIL_BITS) * NUM_LANES),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({dresult,      sresult,        stencil_write_mask}),
        .data_out ({depth_result, stencil_result, mask_out})
    );

    ///////////////////////////////////////////////////////////////

endmodule