`include "VX_fpu_define.vh"

module VX_fpu_fpga #(
    parameter NUM_LANES = 4, 
    parameter TAGW      = 4
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_MOD_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
    input wire [NUM_LANES-1:0][31:0]  datac,
    output wire [NUM_LANES-1:0][31:0] result, 

    output wire has_fflags,
    output wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam FPU_FMA  = 0;
    localparam FPU_DIV  = 1;
    localparam FPU_SQRT = 2;
    localparam FPU_CVT  = 3;
    localparam FPU_NCP  = 4;
    localparam NUM_FPC  = 5;
    localparam FPC_BITS = `LOG2UP(NUM_FPC);

    localparam RSP_ARB_DATAW = (NUM_LANES * `XLEN) + 1 + (NUM_LANES * $bits(fflags_t)) + TAGW;
    
    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][NUM_LANES-1:0][31:0] per_core_result;
    wire [NUM_FPC-1:0][TAGW-1:0] per_core_tag_out;
    wire [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;
    
    wire [NUM_FPC-1:0] per_core_has_fflags;  
    fflags_t [NUM_FPC-1:0][NUM_LANES-1:0] per_core_fflags;  

    reg [FPC_BITS-1:0] core_select;
    reg do_madd, do_sub, do_neg, is_itof, is_signed;

    always @(*) begin
        do_madd   = 0;
        do_sub    = 0;        
        do_neg    = 0;
        is_itof   = 0;
        is_signed = 0;
        case (op_type)
            `INST_FPU_ADD:    begin core_select = FPU_FMA; end
            `INST_FPU_SUB:    begin core_select = FPU_FMA; do_sub = 1; end
            `INST_FPU_MUL:    begin core_select = FPU_FMA; do_neg = 1; end
            `INST_FPU_MADD:   begin core_select = FPU_FMA; do_madd = 1; end
            `INST_FPU_MSUB:   begin core_select = FPU_FMA; do_madd = 1; do_sub = 1; end
            `INST_FPU_NMADD:  begin core_select = FPU_FMA; do_madd = 1; do_neg = 1; end
            `INST_FPU_NMSUB:  begin core_select = FPU_FMA; do_madd = 1; do_sub = 1; do_neg = 1; end
            `INST_FPU_DIV:    begin core_select = FPU_DIV; end
            `INST_FPU_SQRT:   begin core_select = FPU_SQRT; end
            `INST_FPU_CVTWS:  begin core_select = FPU_CVT; is_signed = 1; end
            `INST_FPU_CVTWUS: begin core_select = FPU_CVT; end
            `INST_FPU_CVTSW:  begin core_select = FPU_CVT; is_itof = 1; is_signed = 1; end
            `INST_FPU_CVTSWU: begin core_select = FPU_CVT; is_itof = 1; end
            default:          begin core_select = FPU_NCP; end
        endcase
    end

    `RESET_RELAY (fma_reset, reset);
    `RESET_RELAY (div_reset, reset);
    `RESET_RELAY (sqrt_reset, reset);
    `RESET_RELAY (cvt_reset, reset);
    `RESET_RELAY (ncp_reset, reset);

    VX_fpu_fma #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_fma (
        .clk        (clk), 
        .reset      (fma_reset),   
        .valid_in   (valid_in && (core_select == FPU_FMA)),
        .ready_in   (per_core_ready_in[FPU_FMA]),    
        .tag_in     (tag_in),  
        .frm        (frm),
        .do_madd    (do_madd),
        .do_sub     (do_sub),
        .do_neg     (do_neg),
        .dataa      (dataa), 
        .datab      (datab),    
        .datac      (datac),   
        .has_fflags (per_core_has_fflags[FPU_FMA]),
        .fflags     (per_core_fflags[FPU_FMA]),
        .result     (per_core_result[FPU_FMA]),
        .tag_out    (per_core_tag_out[FPU_FMA]),
        .ready_out  (per_core_ready_out[FPU_FMA]),
        .valid_out  (per_core_valid_out[FPU_FMA])
    );

    VX_fpu_div #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_div (
        .clk        (clk), 
        .reset      (div_reset),   
        .valid_in   (valid_in && (core_select == FPU_DIV)),
        .ready_in   (per_core_ready_in[FPU_DIV]),    
        .tag_in     (tag_in),
        .frm        (frm),  
        .dataa      (dataa), 
        .datab      (datab),   
        .has_fflags (per_core_has_fflags[FPU_DIV]),
        .fflags     (per_core_fflags[FPU_DIV]),   
        .result     (per_core_result[FPU_DIV]),
        .tag_out    (per_core_tag_out[FPU_DIV]),
        .valid_out  (per_core_valid_out[FPU_DIV]),
        .ready_out  (per_core_ready_out[FPU_DIV])
    );

    VX_fpu_sqrt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_sqrt (
        .clk        (clk), 
        .reset      (sqrt_reset),   
        .valid_in   (valid_in && (core_select == FPU_SQRT)),
        .ready_in   (per_core_ready_in[FPU_SQRT]),    
        .tag_in     (tag_in),
        .frm        (frm),    
        .dataa      (dataa), 
        .has_fflags (per_core_has_fflags[FPU_SQRT]),
        .fflags     (per_core_fflags[FPU_SQRT]),
        .result     (per_core_result[FPU_SQRT]),
        .tag_out    (per_core_tag_out[FPU_SQRT]),
        .valid_out  (per_core_valid_out[FPU_SQRT]),
        .ready_out  (per_core_ready_out[FPU_SQRT])
    );

    VX_fpu_cvt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_cvt (
        .clk        (clk), 
        .reset      (cvt_reset),   
        .valid_in   (valid_in && (core_select == FPU_CVT)),
        .ready_in   (per_core_ready_in[FPU_CVT]),    
        .tag_in     (tag_in), 
        .frm        (frm),
        .is_itof    (is_itof),   
        .is_signed  (is_signed),        
        .dataa      (dataa),  
        .has_fflags (per_core_has_fflags[FPU_CVT]),
        .fflags     (per_core_fflags[FPU_CVT]),
        .result     (per_core_result[FPU_CVT]),
        .tag_out    (per_core_tag_out[FPU_CVT]),
        .valid_out  (per_core_valid_out[FPU_CVT]),
        .ready_out  (per_core_ready_out[FPU_CVT])
    );

    VX_fpu_ncomp #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_ncomp (
        .clk        (clk),
        .reset      (ncp_reset),   
        .valid_in   (valid_in && (core_select == FPU_NCP)),
        .ready_in   (per_core_ready_in[FPU_NCP]),        
        .tag_in     (tag_in),        
        .op_type    (op_type),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),        
        .result     (per_core_result[FPU_NCP]), 
        .has_fflags (per_core_has_fflags[FPU_NCP]),
        .fflags     (per_core_fflags[FPU_NCP]),
        .tag_out    (per_core_tag_out[FPU_NCP]),
        .valid_out  (per_core_valid_out[FPU_NCP]),
        .ready_out  (per_core_ready_out[FPU_NCP])
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_FPC-1:0][RSP_ARB_DATAW-1:0] per_core_data_out;

    for (genvar i = 0; i < NUM_FPC; ++i) begin
        assign per_core_data_out[i] = {per_core_result[i], per_core_has_fflags[i], per_core_fflags[i], per_core_tag_out[i]};
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPC),
        .DATAW      (RSP_ARB_DATAW),        
        .ARBITER    ("R"),
        .BUFFERED   (2)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_core_valid_out),        
        .ready_in  (per_core_ready_out),
        .data_in   (per_core_data_out),
        .data_out  ({result, has_fflags, fflags, tag_out}),
        .valid_out (valid_out),
        .ready_out (ready_out)
    );

    // can accept new request?
    assign ready_in = per_core_ready_in[core_select];

endmodule
