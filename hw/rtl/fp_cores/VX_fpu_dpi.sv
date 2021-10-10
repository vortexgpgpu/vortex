`include "VX_fpu_define.vh"

module VX_fpu_dpi #( 
    parameter TAGW = 1
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

    reg is_fadd, is_fsub, is_fmul, is_fmadd, is_fmsub, is_fnmadd, is_fnmsub;
    reg is_itof, is_utof, is_ftoi, is_ftou;
    reg is_fclss, is_flt, is_fle, is_feq, is_fmin, is_fmax, is_fsgnj, is_fsgnjn, is_fsgnjx;

    always @(*) begin
        is_fadd   = 0;
        is_fsub   = 0;        
        is_fmul   = 0;        
        is_fmadd  = 0;
        is_fmsub  = 0;
        is_fnmadd = 0;           
        is_fnmsub = 0;        
        is_itof   = 0;
        is_utof   = 0;
        is_ftoi   = 0;
        is_ftou   = 0;
        is_fclss  = 0;
        is_flt    = 0;
        is_fle    = 0;
        is_feq    = 0;
        is_fmin   = 0;
        is_fmax   = 0;
        is_fsgnj  = 0;
        is_fsgnjn = 0;
        is_fsgnjx = 0;

        case (op_type)
            `INST_FPU_ADD:   begin core_select = FPU_FMA; is_fadd = 1; end
            `INST_FPU_SUB:   begin core_select = FPU_FMA; is_fsub = 1; end
            `INST_FPU_MUL:   begin core_select = FPU_FMA; is_fmul = 1; end
            `INST_FPU_MADD:  begin core_select = FPU_FMA; is_fmadd = 1; end
            `INST_FPU_MSUB:  begin core_select = FPU_FMA; is_fmsub = 1; end
            `INST_FPU_NMADD: begin core_select = FPU_FMA; is_fnmadd = 1; end
            `INST_FPU_NMSUB: begin core_select = FPU_FMA; is_fnmsub = 1; end
            `INST_FPU_DIV:   begin core_select = FPU_DIV; end
            `INST_FPU_SQRT:  begin core_select = FPU_SQRT; end
            `INST_FPU_CVTWS: begin core_select = FPU_CVT; is_ftoi = 1; end
            `INST_FPU_CVTWUS:begin core_select = FPU_CVT; is_ftou = 1; end
            `INST_FPU_CVTSW: begin core_select = FPU_CVT; is_itof = 1; end
            `INST_FPU_CVTSWU:begin core_select = FPU_CVT; is_utof = 1; end
            `INST_FPU_CLASS: begin core_select = FPU_NCP; is_fclss = 1; end  
            `INST_FPU_CMP:   begin core_select = FPU_NCP; 
                            is_fle = (frm == 0); 
                            is_flt = (frm == 1); 
                            is_feq = (frm == 2); 
                         end  
            default:   begin core_select = FPU_NCP; 
                            is_fsgnj  = (frm == 0);
                            is_fsgnjn = (frm == 1);
                            is_fsgnjx = (frm == 2);
                            is_fmin   = (frm == 3);
                            is_fmax   = (frm == 4);
                        end
        endcase
    end

    generate 
    begin : fma
        
        wire [`NUM_THREADS-1:0][31:0] result_fma;
        wire [`NUM_THREADS-1:0][31:0] result_fadd;
        wire [`NUM_THREADS-1:0][31:0] result_fsub;
        wire [`NUM_THREADS-1:0][31:0] result_fmul;
        wire [`NUM_THREADS-1:0][31:0] result_fmadd;
        wire [`NUM_THREADS-1:0][31:0] result_fmsub;
        wire [`NUM_THREADS-1:0][31:0] result_fnmadd;
        wire [`NUM_THREADS-1:0][31:0] result_fnmsub;
        
        fflags_t [`NUM_THREADS-1:0] fflags_fma;
        fflags_t [`NUM_THREADS-1:0] fflags_fadd;
        fflags_t [`NUM_THREADS-1:0] fflags_fsub;
        fflags_t [`NUM_THREADS-1:0] fflags_fmul;
        fflags_t [`NUM_THREADS-1:0] fflags_fmadd;
        fflags_t [`NUM_THREADS-1:0] fflags_fmsub;
        fflags_t [`NUM_THREADS-1:0] fflags_fnmadd;
        fflags_t [`NUM_THREADS-1:0] fflags_fnmsub;

        wire fma_valid = (valid_in && core_select == FPU_FMA);
        wire fma_ready = per_core_ready_out[FPU_FMA] || ~per_core_valid_out[FPU_FMA];

        wire fma_fire = fma_valid && fma_ready;

        always @(*) begin        
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                dpi_fadd   (fma_fire, dataa[i], datab[i], frm, result_fadd[i], fflags_fadd[i]);
                dpi_fsub   (fma_fire, dataa[i], datab[i], frm, result_fsub[i], fflags_fsub[i]);
                dpi_fmul   (fma_fire, dataa[i], datab[i], frm, result_fmul[i], fflags_fmul[i]);
                dpi_fmadd  (fma_fire, dataa[i], datab[i], datac[i], frm, result_fmadd[i], fflags_fmadd[i]);
                dpi_fmsub  (fma_fire, dataa[i], datab[i], datac[i], frm, result_fmsub[i], fflags_fmsub[i]);
                dpi_fnmadd (fma_fire, dataa[i], datab[i], datac[i], frm, result_fnmadd[i], fflags_fnmadd[i]);
                dpi_fnmsub (fma_fire, dataa[i], datab[i], datac[i], frm, result_fnmsub[i], fflags_fnmsub[i]);
            end
        end

        assign result_fma = is_fadd   ? result_fadd :
                            is_fsub   ? result_fsub :
                            is_fmul   ? result_fmul :
                            is_fmadd  ? result_fmadd :               
                            is_fmsub  ? result_fmsub :
                            is_fnmadd ? result_fnmadd :               
                            is_fnmsub ? result_fnmsub :
                                        0;

        assign fflags_fma = is_fadd   ? fflags_fadd :
                            is_fsub   ? fflags_fsub :
                            is_fmul   ? fflags_fmul :
                            is_fmadd  ? fflags_fmadd :               
                            is_fmsub  ? fflags_fmsub :
                            is_fnmadd ? fflags_fnmadd :               
                            is_fnmsub ? fflags_fnmsub : 
                                        0;                

        VX_shift_register #(
            .DATAW  (1 + TAGW + `NUM_THREADS * (32 + $bits(fflags_t))),
            .DEPTH  (`LATENCY_FMA),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fma_ready),
            .data_in  ({fma_valid,                   tag_in,                    result_fma,               fflags_fma}),
            .data_out ({per_core_valid_out[FPU_FMA], per_core_tag_out[FPU_FMA], per_core_result[FPU_FMA], per_core_fflags[FPU_FMA]})
        );

        assign per_core_has_fflags[FPU_FMA] = 1;
        assign per_core_ready_in[FPU_FMA] = fma_ready;

    end
    endgenerate

    generate 
    begin : fdiv

        wire [`NUM_THREADS-1:0][31:0] result_fdiv;
        fflags_t [`NUM_THREADS-1:0] fflags_fdiv;

        wire fdiv_valid = (valid_in && core_select == FPU_DIV);
        wire fdiv_ready = per_core_ready_out[FPU_DIV] || ~per_core_valid_out[FPU_DIV];

        wire fdiv_fire = fdiv_valid && fdiv_ready;
        
        always @(*) begin        
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                dpi_fdiv (fdiv_fire, dataa[i], datab[i], frm, result_fdiv[i], fflags_fdiv[i]);
            end
        end

        VX_shift_register #(
            .DATAW  (1 + TAGW + `NUM_THREADS * (32 + $bits(fflags_t))),
            .DEPTH  (`LATENCY_FDIV),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fdiv_ready),
            .data_in  ({fdiv_valid,                  tag_in,                    result_fdiv,              fflags_fdiv}),
            .data_out ({per_core_valid_out[FPU_DIV], per_core_tag_out[FPU_DIV], per_core_result[FPU_DIV], per_core_fflags[FPU_DIV]})
        );

        assign per_core_has_fflags[FPU_DIV] = 1;
        assign per_core_ready_in[FPU_DIV] = fdiv_ready;

    end
    endgenerate

    generate 
    begin : fsqrt

        wire [`NUM_THREADS-1:0][31:0] result_fsqrt;
        fflags_t [`NUM_THREADS-1:0] fflags_fsqrt;

        wire fsqrt_valid = (valid_in && core_select == FPU_SQRT);
        wire fsqrt_ready = per_core_ready_out[FPU_SQRT] || ~per_core_valid_out[FPU_SQRT];
                
        wire fsqrt_fire = fsqrt_valid && fsqrt_ready;
        
        always @(*) begin        
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                dpi_fsqrt (fsqrt_fire, dataa[i], frm, result_fsqrt[i], fflags_fsqrt[i]);
            end
        end

        VX_shift_register #(
            .DATAW  (1 + TAGW + `NUM_THREADS * (32 + $bits(fflags_t))),
            .DEPTH  (`LATENCY_FSQRT),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fsqrt_ready),
            .data_in  ({fsqrt_valid,                  tag_in,                     result_fsqrt,              fflags_fsqrt}),
            .data_out ({per_core_valid_out[FPU_SQRT], per_core_tag_out[FPU_SQRT], per_core_result[FPU_SQRT], per_core_fflags[FPU_SQRT]})
        );

        assign per_core_has_fflags[FPU_SQRT] = 1;
        assign per_core_ready_in[FPU_SQRT] = fsqrt_ready;

    end
    endgenerate

    generate
    begin : fcvt

        wire [`NUM_THREADS-1:0][31:0] result_fcvt;
        wire [`NUM_THREADS-1:0][31:0] result_itof;
        wire [`NUM_THREADS-1:0][31:0] result_utof;
        wire [`NUM_THREADS-1:0][31:0] result_ftoi;
        wire [`NUM_THREADS-1:0][31:0] result_ftou;
        
        fflags_t [`NUM_THREADS-1:0] fflags_fcvt;
        fflags_t [`NUM_THREADS-1:0] fflags_itof;
        fflags_t [`NUM_THREADS-1:0] fflags_utof;
        fflags_t [`NUM_THREADS-1:0] fflags_ftoi;
        fflags_t [`NUM_THREADS-1:0] fflags_ftou;

        wire fcvt_valid = (valid_in && core_select == FPU_CVT);
        wire fcvt_ready = per_core_ready_out[FPU_CVT] || ~per_core_valid_out[FPU_CVT];

        wire fcvt_fire = fcvt_valid && fcvt_ready;
                
        always @(*) begin        
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                dpi_itof (fcvt_fire, dataa[i], frm, result_itof[i], fflags_itof[i]);
                dpi_utof (fcvt_fire, dataa[i], frm, result_utof[i], fflags_utof[i]);
                dpi_ftoi (fcvt_fire, dataa[i], frm, result_ftoi[i], fflags_ftoi[i]);
                dpi_ftou (fcvt_fire, dataa[i], frm, result_ftou[i], fflags_ftou[i]);
            end
        end

        assign result_fcvt = is_itof ? result_itof :
                             is_utof ? result_utof :
                             is_ftoi ? result_ftoi :
                             is_ftou ? result_ftou : 
                                       0;

        assign fflags_fcvt = is_itof ? fflags_itof :
                             is_utof ? fflags_utof :
                             is_ftoi ? fflags_ftoi :
                             is_ftou ? fflags_ftou : 
                                       0;

        VX_shift_register #(
            .DATAW  (1 + TAGW + `NUM_THREADS * (32 + $bits(fflags_t))),
            .DEPTH  (`LATENCY_FCVT),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fcvt_ready),
            .data_in  ({fcvt_valid,                  tag_in,                    result_fcvt,              fflags_fcvt}),
            .data_out ({per_core_valid_out[FPU_CVT], per_core_tag_out[FPU_CVT], per_core_result[FPU_CVT], per_core_fflags[FPU_CVT]})
        );

        assign per_core_has_fflags[FPU_CVT] = 1;
        assign per_core_ready_in[FPU_CVT] = fcvt_ready;

    end
    endgenerate

    generate 
    begin : fncp

        wire [`NUM_THREADS-1:0][31:0] result_fncp;
        wire [`NUM_THREADS-1:0][31:0] result_fclss;
        wire [`NUM_THREADS-1:0][31:0] result_flt;
        wire [`NUM_THREADS-1:0][31:0] result_fle;
        wire [`NUM_THREADS-1:0][31:0] result_feq;
        wire [`NUM_THREADS-1:0][31:0] result_fmin;
        wire [`NUM_THREADS-1:0][31:0] result_fmax;
        wire [`NUM_THREADS-1:0][31:0] result_fsgnj;
        wire [`NUM_THREADS-1:0][31:0] result_fsgnjn;
        wire [`NUM_THREADS-1:0][31:0] result_fsgnjx;
        reg [`NUM_THREADS-1:0][31:0] result_fmv;

        fflags_t [`NUM_THREADS-1:0] fflags_fncp;
        fflags_t [`NUM_THREADS-1:0] fflags_flt;
        fflags_t [`NUM_THREADS-1:0] fflags_fle;
        fflags_t [`NUM_THREADS-1:0] fflags_feq;
        fflags_t [`NUM_THREADS-1:0] fflags_fmin;
        fflags_t [`NUM_THREADS-1:0] fflags_fmax;

        wire fncp_valid = (valid_in && core_select == FPU_NCP);
        wire fncp_ready = per_core_ready_out[FPU_NCP] || ~per_core_valid_out[FPU_NCP];

        wire fncp_fire = fncp_valid && fncp_ready;
                
        always @(*) begin        
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                dpi_fclss  (fncp_fire, dataa[i], result_fclss[i]);
                dpi_flt    (fncp_fire, dataa[i], datab[i], result_flt[i], fflags_flt[i]);
                dpi_fle    (fncp_fire, dataa[i], datab[i], result_fle[i], fflags_fle[i]);
                dpi_feq    (fncp_fire, dataa[i], datab[i], result_feq[i], fflags_feq[i]);
                dpi_fmin   (fncp_fire, dataa[i], datab[i], result_fmin[i], fflags_fmin[i]);
                dpi_fmax   (fncp_fire, dataa[i], datab[i], result_fmax[i], fflags_fmax[i]);            
                dpi_fsgnj  (fncp_fire, dataa[i], datab[i], result_fsgnj[i]);
                dpi_fsgnjn (fncp_fire, dataa[i], datab[i], result_fsgnjn[i]);
                dpi_fsgnjx (fncp_fire, dataa[i], datab[i], result_fsgnjx[i]);
                result_fmv[i] = dataa[i];
            end
        end

        assign result_fncp = is_fclss  ? result_fclss :
                             is_flt    ? result_flt :
                             is_fle    ? result_fle :
                             is_feq    ? result_feq :
                             is_fmin   ? result_fmin :
                             is_fmax   ? result_fmax :
                             is_fsgnj  ? result_fsgnj :
                             is_fsgnjn ? result_fsgnjn :
                             is_fsgnjx ? result_fsgnjx :
                                         result_fmv;

        wire has_fflags_fncp = (is_flt || is_fle || is_feq || is_fmin || is_fmax);

        assign fflags_fncp = is_flt  ? fflags_flt :
                             is_fle  ? fflags_fle :
                             is_feq  ? fflags_feq :
                             is_fmin ? fflags_fmin :
                             is_fmax ? fflags_fmax : 
                                       0;

        VX_shift_register #(
            .DATAW  (1 + TAGW + 1 + `NUM_THREADS * (32 + $bits(fflags_t))),
            .DEPTH  (`LATENCY_FNCP),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fncp_ready),
            .data_in  ({fncp_valid,                  tag_in,                    has_fflags_fncp,              result_fncp,              fflags_fncp}),
            .data_out ({per_core_valid_out[FPU_NCP], per_core_tag_out[FPU_NCP], per_core_has_fflags[FPU_NCP], per_core_result[FPU_NCP], per_core_fflags[FPU_NCP]})
        );
        
        assign per_core_ready_in[FPU_NCP] = fncp_ready;

    end
    endgenerate

    ///////////////////////////////////////////////////////////////////////////

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