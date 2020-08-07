`include "VX_define.vh"
`include "dspba_library_ver.sv"

module VX_fp_fpga (
	input wire clk,
	input wire reset,   

    input wire  in_valid,
    output wire in_ready,

    input wire [`ISTAG_BITS-1:0] in_tag,
	
    input wire [`FPU_BITS-1:0] op,
    input wire [`FRM_BITS-1:0] frm,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    input wire [`NUM_THREADS-1:0][31:0]  datac,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [`NUM_THREADS-1:0] fflags,

    output wire [`ISTAG_BITS-1:0] out_tag,

    input wire  out_ready,
    output wire out_valid
);
    localparam NUM_FPC  = 12;
    localparam FPC_BITS = `LOG2UP(NUM_FPC);
    
    wire [NUM_FPC-1:0] core_in_ready;
    wire [NUM_FPC-1:0][`NUM_THREADS-1:0][31:0] core_result;
    wire fpnew_has_fflags;  
    fflags_t fpnew_fflags;  
    wire [NUM_FPC-1:0][`ISTAG_BITS-1:0] core_out_tag;
    wire [NUM_FPC-1:0] core_out_ready;
    wire [NUM_FPC-1:0] core_out_valid;

    reg [FPC_BITS-1:0] core_select;
    reg fmadd_negate;

    genvar i;

    always @(*) begin
        core_select  = 0;
        fmadd_negate = 0;
        case (op)
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

    VX_fp_noncomp fp_noncomp (
        .clk        (clk),
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 0)),
        .in_ready   (core_in_ready[0]),        
        .in_tag     (in_tag),        
        .op         (op),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),
        .result     (core_result[0]), 
        .has_fflags (fpnew_has_fflags),
        .fflags     (fpnew_fflags),
        .out_tag    (core_out_tag[0]),
        .out_ready  (core_out_ready[0]),
        .out_valid  (core_out_valid[0])
    );
    
    VX_fp_add fp_add (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 1)),
        .in_ready   (core_in_ready[1]),    
        .in_tag     (in_tag),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (core_result[1]),
        .out_tag    (core_out_tag[1]),
        .out_ready  (core_out_ready[1]),
        .out_valid  (core_out_valid[1])
    );

    VX_fp_sub fp_sub (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 2)),
        .in_ready   (core_in_ready[2]),    
        .in_tag     (in_tag),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (core_result[2]),
        .out_tag    (core_out_tag[2]),
        .out_ready  (core_out_ready[2]),
        .out_valid  (core_out_valid[2])
    );

    VX_fp_mul fp_mul (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 3)),
        .in_ready   (core_in_ready[3]),    
        .in_tag     (in_tag),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (core_result[3]),
        .out_tag    (core_out_tag[3]),
        .out_ready  (core_out_ready[3]),
        .out_valid  (core_out_valid[3])
    );

    VX_fp_madd fp_madd (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 4)),
        .in_ready   (core_in_ready[4]),    
        .in_tag     (in_tag),    
        .negate     (fmadd_negate),
        .dataa      (dataa), 
        .datab      (datab),         
        .datac      (datac),        
        .result     (core_result[4]),
        .out_tag    (core_out_tag[4]),
        .out_ready  (core_out_ready[4]),
        .out_valid  (core_out_valid[4])
    );

    VX_fp_msub fp_msub (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 5)),
        .in_ready   (core_in_ready[5]),    
        .in_tag     (in_tag),    
        .negate     (fmadd_negate),
        .dataa      (dataa), 
        .datab      (datab),   
        .datac      (datac),              
        .result     (core_result[5]),
        .out_tag    (core_out_tag[5]),
        .out_ready  (core_out_ready[5]),
        .out_valid  (core_out_valid[5])
    );

    VX_fp_div fp_div (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 6)),
        .in_ready   (core_in_ready[6]),    
        .in_tag     (in_tag),    
        .dataa      (dataa), 
        .datab      (datab),         
        .result     (core_result[6]),
        .out_tag    (core_out_tag[6]),
        .out_ready  (core_out_ready[6]),
        .out_valid  (core_out_valid[6])
    );

    VX_fp_sqrt fp_sqrt (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 7)),
        .in_ready   (core_in_ready[7]),    
        .in_tag     (in_tag),    
        .dataa      (dataa),  
        .result     (core_result[7]),
        .out_tag    (core_out_tag[7]),
        .out_ready  (core_out_ready[7]),
        .out_valid  (core_out_valid[7])
    );

    VX_fp_ftoi fp_ftoi (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 8)),
        .in_ready   (core_in_ready[8]),    
        .in_tag     (in_tag),    
        .dataa      (dataa),  
        .result     (core_result[8]),
        .out_tag    (core_out_tag[8]),
        .out_ready  (core_out_ready[8]),
        .out_valid  (core_out_valid[8])
    );

    VX_fp_ftou fp_ftou (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 9)),
        .in_ready   (core_in_ready[9]),    
        .in_tag     (in_tag),    
        .dataa      (dataa),  
        .result     (core_result[9]),
        .out_tag    (core_out_tag[9]),
        .out_ready  (core_out_ready[9]),
        .out_valid  (core_out_valid[9])
    );

    VX_fp_itof fp_itof (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 10)),
        .in_ready   (core_in_ready[10]),    
        .in_tag     (in_tag),    
        .dataa      (dataa),  
        .result     (core_result[10]),
        .out_tag    (core_out_tag[10]),
        .out_ready  (core_out_ready[10]),
        .out_valid  (core_out_valid[10])
    );

    VX_fp_utof fp_utof (
        .clk        (clk), 
        .reset      (reset),   
        .in_valid   (in_valid && (core_select == 11)),
        .in_ready   (core_in_ready[11]),    
        .in_tag     (in_tag),    
        .dataa      (dataa),  
        .result     (core_result[11]),
        .out_tag    (core_out_tag[11]),
        .out_ready  (core_out_ready[11]),
        .out_valid  (core_out_valid[11])
    );

    wire [FPC_BITS-1:0] fp_index;
    wire fp_valid;
    
    VX_priority_encoder #(
        .N(NUM_FPC)
    ) wb_select (
        .data_in   (core_out_valid),
        .data_out  (fp_index),
        .valid_out (fp_valid)
    );

    for (i = 0; i < NUM_FPC; i++) begin
        assign core_out_ready[i] = out_ready && (i == fp_index);
    end

    wire                          tmp_valid  = fp_valid;
    wire [`ISTAG_BITS-1:0]        tmp_tag    = core_out_tag[fp_index];
    wire [`NUM_THREADS-1:0][31:0] tmp_result = core_result[fp_index];
    wire                          tmp_has_fflags = fpnew_has_fflags && (fp_index == 0);
    fflags_t [`NUM_THREADS-1:0]   tmp_flags  = fpnew_fflags;            

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + (`NUM_THREADS * 32) + 1 + `FFG_BITS)
    ) nc_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({tmp_valid, tmp_tag, tmp_result, tmp_has_fflags, tmp_fflags}),
        .out   ({out_valid, out_tag, result,     has_fflags,     fflags})
    );

endmodule