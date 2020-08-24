`include "VX_define.vh"
`include "dspba_library_ver.sv"

module VX_fp_fpga #( 
    parameter TAGW = 1
) (
	input wire clk,
	input wire reset,   

    input wire  valid_in,
    output wire ready_in,

    input wire [TAGW-1:0] tag_in,
	
    input wire [`FPU_BITS-1:0] op_type,
    input wire [`MOD_BITS-1:0] frm,

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
    localparam NUM_FPC  = 12;
    localparam FPC_BITS = `LOG2UP(NUM_FPC);
    
    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][`NUM_THREADS-1:0][31:0] per_core_result;
    wire [NUM_FPC-1:0][TAGW-1:0] per_core_tag_out;
    reg [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;
    
    wire fpnew_has_fflags;  
    fflags_t [`NUM_THREADS-1:0] fpnew_fflags;  

    reg [FPC_BITS-1:0] core_select;
    reg fmadd_negate;

    always @(*) begin
        core_select  = 0;
        fmadd_negate = 0;
        case (op_type)
            `FPU_ADD:    core_select = 1;
            `FPU_SUB:    core_select = 2;
            `FPU_MUL:    core_select = 3;
            `FPU_MADD:   core_select = 4;
            `FPU_MSUB:   core_select = 5;
            `FPU_NMSUB:  begin core_select = 4; fmadd_negate = 1; end
            `FPU_NMADD:  begin core_select = 5; fmadd_negate = 1; end           
            `FPU_DIV:    core_select = 6;
            `FPU_SQRT:   core_select = 7;
            `FPU_CVTWS:  core_select = 8;
            `FPU_CVTWUS: core_select = 9;
            `FPU_CVTSW:  core_select = 10;
            `FPU_CVTSWU: core_select = 11;
            default:;
        endcase
    end

    VX_fp_noncomp #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_noncomp (
        .clk        (clk),
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 0)),
        .ready_in   (per_core_ready_in[0]),        
        .tag_in     (tag_in),        
        .op_type    (op_type),
        .frm        (op_mod),
        .dataa      (dataa),
        .datab      (datab),
        .result     (per_core_result[0]), 
        .has_fflags (fpnew_has_fflags),
        .fflags     (fpnew_fflags),
        .tag_out    (per_core_tag_out[0]),
        .ready_out  (per_core_ready_out[0]),
        .valid_out  (per_core_valid_out[0])
    );
    
    VX_fp_add #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_add (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 1)),
        .ready_in   (per_core_ready_in[1]),    
        .tag_in     (tag_in),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (per_core_result[1]),
        .tag_out    (per_core_tag_out[1]),
        .ready_out  (per_core_ready_out[1]),
        .valid_out  (per_core_valid_out[1])
    );

    VX_fp_sub #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_sub (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 2)),
        .ready_in   (per_core_ready_in[2]),    
        .tag_in     (tag_in),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (per_core_result[2]),
        .tag_out    (per_core_tag_out[2]),
        .ready_out  (per_core_ready_out[2]),
        .valid_out  (per_core_valid_out[2])
    );

    VX_fp_mul #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_mul (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 3)),
        .ready_in   (per_core_ready_in[3]),    
        .tag_in     (tag_in),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (per_core_result[3]),
        .tag_out    (per_core_tag_out[3]),
        .ready_out  (per_core_ready_out[3]),
        .valid_out  (per_core_valid_out[3])
    );

    VX_fp_madd #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_madd (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 4)),
        .ready_in   (per_core_ready_in[4]),    
        .tag_in     (tag_in),    
        .negate     (fmadd_negate),
        .dataa      (dataa), 
        .datab      (datab),         
        .datac      (datac),        
        .result     (per_core_result[4]),
        .tag_out    (per_core_tag_out[4]),
        .ready_out  (per_core_ready_out[4]),
        .valid_out  (per_core_valid_out[4])
    );

    VX_fp_msub #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_msub (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 5)),
        .ready_in   (per_core_ready_in[5]),    
        .tag_in     (tag_in),    
        .negate     (fmadd_negate),
        .dataa      (dataa), 
        .datab      (datab),   
        .datac      (datac),              
        .result     (per_core_result[5]),
        .tag_out    (per_core_tag_out[5]),
        .ready_out  (per_core_ready_out[5]),
        .valid_out  (per_core_valid_out[5])
    );

    VX_fp_div #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_div (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 6)),
        .ready_in   (per_core_ready_in[6]),    
        .tag_in     (tag_in),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (per_core_result[6]),
        .tag_out    (per_core_tag_out[6]),
        .ready_out  (per_core_ready_out[6]),
        .valid_out  (per_core_valid_out[6])
    );

    VX_fp_sqrt #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_sqrt (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 7)),
        .ready_in   (per_core_ready_in[7]),    
        .tag_in     (tag_in),    
        .dataa      (dataa),  
        .result     (per_core_result[7]),
        .tag_out    (per_core_tag_out[7]),
        .ready_out  (per_core_ready_out[7]),
        .valid_out  (per_core_valid_out[7])
    );

    VX_fp_ftoi #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_ftoi (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 8)),
        .ready_in   (per_core_ready_in[8]),    
        .tag_in     (tag_in),    
        .dataa      (dataa),  
        .result     (per_core_result[8]),
        .tag_out    (per_core_tag_out[8]),
        .ready_out  (per_core_ready_out[8]),
        .valid_out  (per_core_valid_out[8])
    );

    VX_fp_ftou #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_ftou (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 9)),
        .ready_in   (per_core_ready_in[9]),    
        .tag_in     (tag_in),    
        .dataa      (dataa),  
        .result     (per_core_result[9]),
        .tag_out    (per_core_tag_out[9]),
        .ready_out  (per_core_ready_out[9]),
        .valid_out  (per_core_valid_out[9])
    );

    VX_fp_itof #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_itof (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 10)),
        .ready_in   (per_core_ready_in[10]),    
        .tag_in     (tag_in),    
        .dataa      (dataa),  
        .result     (per_core_result[10]),
        .tag_out    (per_core_tag_out[10]),
        .ready_out  (per_core_ready_out[10]),
        .valid_out  (per_core_valid_out[10])
    );

    VX_fp_utof #(
        .TAGW (TAGW),
        .LANES(`NUM_THREADS)
    ) fp_utof (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in && (core_select == 11)),
        .ready_in   (per_core_ready_in[11]),    
        .tag_in     (tag_in),    
        .dataa      (dataa),  
        .result     (per_core_result[11]),
        .tag_out    (per_core_tag_out[11]),
        .ready_out  (per_core_ready_out[11]),
        .valid_out  (per_core_valid_out[11])
    );

    reg valid_out_r;
    reg has_fflags_r;
    reg [`NUM_THREADS-1:0][31:0] result_r;
    reg [TAGW-1:0] tag_out_r;

    always @(*) begin
        per_core_ready_out = 0;
        valid_out_r  = 0;
        has_fflags_r = 0;
        result_r     = 'x;
        tag_out_r    = 'x;
        for (integer i = 0; i < NUM_FPC; i++) begin
            if (per_core_valid_out[i]) begin
                per_core_ready_out[i] = 1;
                valid_out_r  = i;
                has_fflags_r = fpnew_has_fflags && (i == 0);
                result_r     = per_core_result[i];
                tag_out_r    = per_core_tag_out[i];
                break;
            end
        end
    end

    assign ready_in   = (& per_core_ready_in);
    assign valid_out  = valid_out_r;
    assign has_fflags = has_fflags_r;
    assign tag_out    = tag_out_r;
    assign result     = result_r;    
    assign fflags     = fpnew_fflags;

endmodule