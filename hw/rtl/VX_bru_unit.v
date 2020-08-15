`include "VX_define.vh"

module VX_bru_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_bru_req_if bru_req_if,

    // Outputs
    VX_branch_ctl_if branch_ctl_if,
    VX_exu_to_cmt_if bru_commit_if    
);    
    wire [`BRU_BITS-1:0] bru_op = bru_req_if.op;      
    wire bru_neg    = `BRU_NEG(bru_op);    
    wire bru_less   = `BRU_LESS(bru_op);
    wire bru_signed = `BRU_SIGNED(bru_op);
    wire bru_static = `BRU_STATIC(bru_op);

    wire [31:0] rs1_data = bru_req_if.rs1_data;
    wire [31:0] rs2_data = bru_req_if.rs2_data;

    wire [32:0] signed_in1 = {bru_signed & rs1_data[31], rs1_data};
    wire [32:0] signed_in2 = {bru_signed & rs2_data[31], rs2_data};
    wire is_less  = $signed(signed_in1) < $signed(signed_in2);

    wire is_equal = (rs1_data == rs2_data);
        
    wire taken = ((bru_less ? is_less : is_equal) ^ bru_neg) | bru_static;

    wire [31:0] base_addr = bru_req_if.rs1_is_PC ? bru_req_if.curr_PC : rs1_data;
    wire [31:0] dest = base_addr + bru_req_if.offset;

    wire [31:0] jal_result = bru_req_if.curr_PC + 4;
    wire [31:0] jal_result_r;

    VX_generic_register #(
        .N(1 + `NW_BITS + `ISTAG_BITS + 1 + 32 + 32)
    ) bru_reg (
        .clk   (clk),
        .reset (reset),
        .stall (0),
        .flush (0),
        .in    ({bru_req_if.valid,    bru_req_if.wid,    bru_req_if.issue_tag,    taken,               dest,               jal_result}),
        .out   ({bru_commit_if.valid, branch_ctl_if.wid, bru_commit_if.issue_tag, branch_ctl_if.taken, branch_ctl_if.dest, jal_result_r})
    );

    assign branch_ctl_if.valid = bru_commit_if.valid;    

    assign bru_commit_if.data = {`NUM_THREADS{jal_result_r}};    
    
    assign bru_req_if.ready = 1'b1;

endmodule