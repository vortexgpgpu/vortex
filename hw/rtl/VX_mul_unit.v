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
    localparam MULQ_BITS = `LOG2UP(`MULQ_SIZE);

    wire [`MUL_BITS-1:0]          alu_op  = mul_req_if.op_type;
    wire                        is_div_op = `IS_DIV_OP(alu_op);
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = mul_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = mul_req_if.rs2_data;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [31:0] rsp_PC;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;
    wire [MULQ_BITS-1:0] tag_in, tag_out;
    wire valid_out, ready_out;
    wire mulq_full;

    wire mulq_push = mul_req_if.valid && mul_req_if.ready;
    wire mulq_pop  = valid_out && ready_out;

    VX_cam_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1),
        .SIZE  (`MULQ_SIZE)
    ) req_metadata_buf  (
        .clk            (clk),
        .reset          (reset),
        .acquire_slot   (mulq_push),       
        .write_addr     (tag_in),                
        .read_addr      (tag_out),
        .release_addr   (tag_out),        
        .write_data     ({mul_req_if.wid, mul_req_if.tmask, mul_req_if.PC, mul_req_if.rd, mul_req_if.wb}),                    
        .read_data      ({rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd,        rsp_wb}),        
        .release_slot   (mulq_pop),     
        .full           (mulq_full)
    );

    wire valid_in = mul_req_if.valid && ~mulq_full;

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire [MULQ_BITS-1:0] mul_tag;
    wire is_mul_in = (alu_op == `MUL_MUL);    
    wire is_mul_out;   

    wire mul_valid_out;
    wire mul_valid_in = valid_in && !is_div_op;     
    wire mul_ready_in = ready_out || ~mul_valid_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    

        wire [32:0] mul_in1 = {(alu_op != `MUL_MULHU)                          & alu_in1[i][31], alu_in1[i]};
        wire [32:0] mul_in2 = {(alu_op != `MUL_MULHU && alu_op != `MUL_MULHSU) & alu_in2[i][31], alu_in2[i]};
    `IGNORE_WARNINGS_BEGIN
        wire [65:0] mul_result_tmp;
    `IGNORE_WARNINGS_END

        VX_multiplier #(
            .WIDTHA(33),
            .WIDTHB(33),
            .WIDTHP(66),
            .SIGNED(1),
            .LATENCY(`LATENCY_IMUL)
        ) multiplier (
            .clk(clk),
            .enable(mul_ready_in),
            .dataa(mul_in1),
            .datab(mul_in2),
            .result(mul_result_tmp)
        );

        assign mul_result[i] = is_mul_out ? mul_result_tmp[31:0] : mul_result_tmp[63:32];            
    end

    VX_shift_register #(
        .DATAW(1 + MULQ_BITS + 1),
        .DEPTH(`LATENCY_IMUL)
    ) mul_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(mul_ready_in),
        .in({mul_valid_in,   tag_in,  is_mul_in}),
        .out({mul_valid_out, mul_tag, is_mul_out})
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] div_result_tmp, rem_result_tmp;

    wire is_rem_op     = (alu_op == `MUL_REM) || (alu_op == `MUL_REMU);
    wire is_signed_div = (alu_op == `MUL_DIV) || (alu_op == `MUL_REM);     
    wire div_valid_in  = valid_in && is_div_op; 
    wire div_ready_out = ready_out && ~mul_valid_out; // arbitration prioritizes MUL  
    wire div_ready_in;
    wire div_valid_out;
    wire is_rem_op_out;
    wire [MULQ_BITS-1:0] div_tag;
    
    VX_serial_div #(
        .WIDTHN(32),
        .WIDTHD(32),
        .WIDTHQ(32),
        .WIDTHR(32),
        .LANES(`NUM_THREADS),
        .TAGW(MULQ_BITS + 1)
    ) divide (
        .clk(clk),
        .reset(reset),
        .ready_in(div_ready_in),
        .valid_in(div_valid_in),
        .signed_mode(is_signed_div),
        .tag_in({tag_in, is_rem_op}),
        .numer(alu_in1),
        .denom(alu_in2),
        .quotient(div_result_tmp),
        .remainder(rem_result_tmp),
        .ready_out(div_ready_out),
        .valid_out(div_valid_out),
        .tag_out({div_tag, is_rem_op_out})
    );

    wire [`NUM_THREADS-1:0][31:0] div_result = is_rem_op_out ? rem_result_tmp : div_result_tmp; 

    ///////////////////////////////////////////////////////////////////////////

    wire stall_out = ~mul_commit_if.ready && mul_commit_if.valid;
    assign ready_out = ~stall_out;

    assign valid_out = mul_valid_out || div_valid_out; 
    assign tag_out = mul_valid_out ? mul_tag : div_tag;
    wire [`NUM_THREADS-1:0][31:0] result = mul_valid_out ? mul_result : div_result;

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .R(1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .stall    (stall_out),
        .flush    (1'b0),
        .data_in  ({valid_out,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           result}),
        .data_out ({mul_commit_if.valid, mul_commit_if.wid, mul_commit_if.tmask, mul_commit_if.PC, mul_commit_if.rd, mul_commit_if.wb, mul_commit_if.data})
    );

    // can accept new request?
    assign mul_req_if.ready = (is_div_op ? div_ready_in : mul_ready_in) && ~mulq_full;
    
endmodule