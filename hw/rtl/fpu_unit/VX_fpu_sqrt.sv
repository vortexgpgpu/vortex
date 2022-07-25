`include "VX_fpu_define.vh"

module VX_fpu_sqrt #( 
    parameter NUM_LANES = 1,
    parameter TAGW = 1    
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    output wire [NUM_LANES-1:0][31:0] result,  

    output wire has_fflags,
    output fflags_t [NUM_LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    

`ifdef QUARTUS
    
    VX_acl_fsqrt #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAGW)
    ) fp_sqrt (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .tag_in     (tag_in),
        .frm        (frm),  
        .dataa      (dataa),
        .has_fflags (has_fflags),
        .fflags     (fflags),   
        .result     (result),
        .tag_out    (tag_out),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

`elsif VIVADO

    VX_xil_fsqrt #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAGW)
    ) fp_sqrt (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .tag_in     (tag_in),
        .frm        (frm),  
        .dataa      (dataa),
        .has_fflags (has_fflags),
        .fflags     (fflags),   
        .result     (result),
        .tag_out    (tag_out),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

`else

    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [31:0] r;
        fflags_t f;

        always @(*) begin        
            dpi_fsqrt (enable && valid_in, dataa[i], frm, r, f);
        end
        `UNUSED_VAR (f)

        VX_shift_register #(
            .DATAW  (32),
            .DEPTH  (`LATENCY_FSQRT),
            .RESETW (1)
        ) shift_req_dpi (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (r),
            .data_out (result[i])
        );
    end

    VX_shift_register #(
        .DATAW  (1 + TAGW),
        .DEPTH  (`LATENCY_FSQRT),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in}),
        .data_out ({valid_out, tag_out})
    );

    assign ready_in = enable;

    `UNUSED_VAR (frm)
    assign has_fflags = 0;
    assign fflags = 0;

`endif

endmodule
