`include "VX_define.vh"

module VX_branch_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_branch_req_if    branch_req_if,

    // Outputs    
    VX_branch_rsp_if    branch_rsp_if,
    VX_wb_if            branch_wb_if
);

    wire [`NT_BITS-1:0] br_result_index;

    VX_priority_encoder #(
        .N(`NUM_THREADS)
    ) choose_alu_result (
        .data_in  (alu_req_if.valid),
        .data_out (br_result_index),
        `UNUSED_PIN (valid_out)
    );

    wire [`BR_BITS-1:0] br_op = branch_req_if.br_op;
    wire [31:0] rs1_data = branch_req_if.rs1_data[br_result_index];
    wire [31:0] rs2_data = branch_req_if.rs2_data[br_result_index];

    wire [32:0] sub_in1 = {(br_op != `BR_LTU) & (br_op != `BR_GEU) & rs1_data[31], rs1_data};
    wire [32:0] sub_in2 = {(br_op != `BR_LTU) & (br_op != `BR_GEU) & rs2_data[31], rs2_data};
    wire [32:0] sub_res = $signed(sub_in1) - $signed(sub_in2);

    wire sub_sign  = sub_res[32];
    wire sub_nzero = (| sub_res[31:0]);
    
    reg br_taken;
    always @(*) begin
        case (br_op)            
            `BR_NE:  br_taken = sub_nzero;
            `BR_EQ:  br_taken = ~sub_nzero;
            `BR_LT, 
            `BR_LTU: br_taken = sub_sign;
            `BR_GE, 
            `BR_GEU: br_taken = ~sub_sign;
            default: br_taken = 1'b1;
        endcase
    end

    wire in_valid = (| branch_req_if.valid);

    wire [31:0] base_addr = (br_op == `BR_JALR) ? rs1_data : branch_req_if.curr_PC;
    wire [31:0] br_dest   = $signed(base_addr) + $signed(branch_req_if.offset);

    wire stall = (~branch_wb_if.ready && (| branch_wb_if.valid));                               

    VX_generic_register #(
        .N(1 + `NW_BITS + 1 + 32)
    ) rsp_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({in_valid,            branch_req_if.warp_num,  br_taken,            br_dest}),
        .out   ({branch_rsp_if.valid, branch_rsp_if.warp_num,  branch_rsp_if.taken, branch_rsp_if.dest})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `NR_BITS + `WB_BITS + (`NUM_THREADS * 32)),
    ) wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({branch_req_if.valid, branch_req_if.warp_num, branch_req_if.curr_PC, branch_req_if.rd, branch_req_if.wb, {`NUM_THREADS{branch_req_if.next_PC}}}),
        .out   ({branch_wb_if.valid,  branch_wb_if.warp_num,  branch_wb_if.curr_PC,  branch_wb_if.rd,  branch_wb_if.wb,  branch_wb_if.data})
    );    

    assign branch_req_if.ready = ~stall;

endmodule