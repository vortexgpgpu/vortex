`include "VX_define.vh"

module VX_mul_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_mul_req_if   mul_req_if,

    // Outputs
    VX_commit_if    mul_commit_if
); 

    wire [`MUL_BITS-1:0]          alu_op  = mul_req_if.op_type;
    wire                        is_div_op = `IS_DIV_OP(alu_op);
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = mul_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = mul_req_if.rs2_data;

    wire ready_out;
    
    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire [`NW_BITS-1:0] mul_wid_out;
    wire [`NUM_THREADS-1:0] mul_tmask_out;
    wire [31:0] mul_PC_out;
    wire [`NR_BITS-1:0] mul_rd_out;
    wire mul_wb_out;

    wire mul_valid_out;
    wire mul_valid_in = mul_req_if.valid && !is_div_op;    
    wire mul_ready_in = ready_out || ~mul_valid_out;
        
    wire is_mulh_in = (alu_op != `MUL_MUL);
    wire is_mulh_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] mul_in1 = {(alu_op != `MUL_MULHU)                          & alu_in1[i][31], alu_in1[i]};
        wire [32:0] mul_in2 = {(alu_op != `MUL_MULHU && alu_op != `MUL_MULHSU) & alu_in2[i][31], alu_in2[i]};
    `IGNORE_WARNINGS_BEGIN
        wire [65:0] mul_result_tmp;
    `IGNORE_WARNINGS_END

        VX_multiplier #(
            .WIDTHA  (33),
            .WIDTHB  (33),
            .WIDTHP  (66),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (mul_ready_in),
            .dataa  (mul_in1),
            .datab  (mul_in2),
            .result (mul_result_tmp)
        );

        assign mul_result[i] = is_mulh_out ? mul_result_tmp[63:32] : mul_result_tmp[31:0];
    end

    VX_shift_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,   mul_req_if.wid, mul_req_if.tmask, mul_req_if.PC, mul_req_if.rd, mul_req_if.wb, is_mulh_in}),
        .data_out ({mul_valid_out, mul_wid_out,    mul_tmask_out,    mul_PC_out,    mul_rd_out,    mul_wb_out,    is_mulh_out})
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] div_result_tmp, rem_result_tmp;    
    wire [`NW_BITS-1:0] div_wid_out;
    wire [`NUM_THREADS-1:0] div_tmask_out;
    wire [31:0] div_PC_out;
    wire [`NR_BITS-1:0] div_rd_out;
    wire div_wb_out;

    wire is_rem_op_in  = (alu_op == `MUL_REM) || (alu_op == `MUL_REMU);
    wire is_signed_div = (alu_op == `MUL_DIV) || (alu_op == `MUL_REM);     
    wire div_valid_in  = mul_req_if.valid && is_div_op; 
    wire div_ready_out = ready_out && ~mul_valid_out; // arbitration prioritizes MUL  
    wire div_ready_in;
    wire div_valid_out;
    wire is_rem_op_out;
    
    VX_serial_div #(
        .WIDTHN (32),
        .WIDTHD (32),
        .WIDTHQ (32),
        .WIDTHR (32),
        .LANES  (`NUM_THREADS),
        .TAGW   (`NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1)
    ) divide (
        .clk       (clk),
        .reset     (reset),        
        .valid_in  (div_valid_in),
        .ready_in  (div_ready_in),
        .signed_mode(is_signed_div),
        .tag_in    ({mul_req_if.wid, mul_req_if.tmask, mul_req_if.PC, mul_req_if.rd, mul_req_if.wb, is_rem_op_in}),
        .numer     (alu_in1),
        .denom     (alu_in2),
        .quotient  (div_result_tmp),
        .remainder (rem_result_tmp),
        .ready_out (div_ready_out),
        .valid_out (div_valid_out),
        .tag_out   ({div_wid_out, div_tmask_out, div_PC_out, div_rd_out, div_wb_out, is_rem_op_out})
    );

    wire [`NUM_THREADS-1:0][31:0] div_result = is_rem_op_out ? rem_result_tmp : div_result_tmp; 

    ///////////////////////////////////////////////////////////////////////////

    wire stall_out = ~mul_commit_if.ready && mul_commit_if.valid;
    assign ready_out = ~stall_out;

    wire                    rsp_valid = mul_valid_out || div_valid_out;  
    wire [`NW_BITS-1:0]     rsp_wid   = mul_valid_out ? mul_wid_out : div_wid_out;
    wire [`NUM_THREADS-1:0] rsp_tmask = mul_valid_out ? mul_tmask_out : div_tmask_out;
    wire [31:0]             rsp_PC    = mul_valid_out ? mul_PC_out : div_PC_out;
    wire [`NR_BITS-1:0]     rsp_rd    = mul_valid_out ? mul_rd_out : div_rd_out;
    wire                    rsp_wb    = mul_valid_out ? mul_wb_out : div_wb_out;
    wire [`NUM_THREADS-1:0][31:0] rsp_data = mul_valid_out ? mul_result : div_result;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({rsp_valid,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           rsp_data}),
        .data_out ({mul_commit_if.valid, mul_commit_if.wid, mul_commit_if.tmask, mul_commit_if.PC, mul_commit_if.rd, mul_commit_if.wb, mul_commit_if.data})
    );

    assign mul_commit_if.eop = 1'b1;

    // can accept new request?
    assign mul_req_if.ready = is_div_op ? div_ready_in : mul_ready_in;
    
endmodule