`include "VX_fpu_define.vh"

module VX_fpu_fpga #( 
    parameter TAGW = 4
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_MOD_BITS-1:0] frm,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    input wire [`NUM_THREADS-1:0][31:0]  datac,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [`NUM_THREADS-1:0] fflags,

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
    
    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][`NUM_THREADS-1:0][31:0] per_core_result;
    wire [NUM_FPC-1:0][TAGW-1:0] per_core_tag_out;
    reg [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;
    
    wire [NUM_FPC-1:0] per_core_has_fflags;  
    fflags_t [NUM_FPC-1:0][`NUM_THREADS-1:0] per_core_fflags;  

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
            default:     begin core_select = FPU_NCP; end
        endcase
    end

    `RESET_RELAY (fma_reset);
    `RESET_RELAY (div_reset);
    `RESET_RELAY (sqrt_reset);
    `RESET_RELAY (cvt_reset);
    `RESET_RELAY (ncp_reset);

    VX_fp_fma #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
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

    VX_fp_div #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
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
        .ready_out  (per_core_ready_out[FPU_DIV]),
        .valid_out  (per_core_valid_out[FPU_DIV])
    );

    VX_fp_sqrt #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
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
        .ready_out  (per_core_ready_out[FPU_SQRT]),
        .valid_out  (per_core_valid_out[FPU_SQRT])
    );

    VX_fp_cvt #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
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
        .ready_out  (per_core_ready_out[FPU_CVT]),
        .valid_out  (per_core_valid_out[FPU_CVT])
    );

    VX_fp_ncomp #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
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
        .ready_out  (per_core_ready_out[FPU_NCP]),
        .valid_out  (per_core_valid_out[FPU_NCP])
    );

    reg has_fflags_n;
    fflags_t [`NUM_THREADS-1:0] fflags_n;
    reg [`NUM_THREADS-1:0][31:0] result_n;
    reg [TAGW-1:0] tag_out_n;

    always @(*) begin
        per_core_ready_out = 0;
        has_fflags_n       = 'x;
        fflags_n           = 'x;
        result_n           = 'x;
        tag_out_n          = 'x;
        for (integer i = 0; i < NUM_FPC; i++) begin
            if (per_core_valid_out[i]) begin                
                has_fflags_n = per_core_has_fflags[i];
                fflags_n     = per_core_fflags[i];
                result_n     = per_core_result[i];
                tag_out_n    = per_core_tag_out[i];
                per_core_ready_out[i] = ready_out;
                break;
            end
        end
    end

    assign valid_out  = (| per_core_valid_out);
    assign has_fflags = has_fflags_n;
    assign tag_out    = tag_out_n;
    assign result     = result_n;    
    assign fflags     = fflags_n;

    assign ready_in = per_core_ready_in[core_select];

endmodule