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

    `UNUSED_VAR (frm)
    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    VX_shift_register #(
        .DATAW  (1 + TAGW),
        .DEPTH  (`LATENCY_FSQRT),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in}),
        .data_out ({valid_out, tag_out})
    );

    assign ready_in = enable;    

`ifdef QUARTUS
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        acl_fsqrt fsqrt (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (dataa[i]),
            .q      (result[i])
        );
    end

`elsif VIVADO

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire tuser;       

        xil_fsqrt fsqrt (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (dataa[i]),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (result[i]),
            .m_axis_result_tuser (tuser)
        );

        assign fflags[i].NX = 1'b0;
        assign fflags[i].UF = 1'b0;
        assign fflags[i].OF = 1'b0;
        assign fflags[i].DZ = 1'b0;
        assign fflags[i].NV = tuser;
    end

`else

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [31:0] r;

        fflags_t f;
        `UNUSED_VAR (f)

        always @(*) begin        
            dpi_fsqrt (enable && valid_in, dataa[i], frm, r, f);
        end
        
        VX_shift_register #(
            .DATAW  (32),
            .DEPTH  (`LATENCY_FSQRT)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (enable),
            .data_in  (r),
            .data_out (result[i])
        );
    end

    assign has_fflags = 0;
    assign fflags = '0;

`endif

endmodule
