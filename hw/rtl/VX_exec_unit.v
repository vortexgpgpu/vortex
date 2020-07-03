`include "VX_define.vh"

module VX_exec_unit (
    input wire            clk,
    input wire            reset,
        // Request
    VX_exec_unit_req_if   exec_unit_req_if,

    // Output
    VX_wb_if              inst_exec_wb_if,
    VX_jal_rsp_if         jal_rsp_if,
    VX_branch_rsp_if      branch_rsp_if,

    input wire            no_slot_exec,
    output wire           delay
);

    wire[`NUM_THREADS-1:0][31:0] in_a_reg_data;
    wire[`NUM_THREADS-1:0][31:0] in_b_reg_data;
    wire[4:0]            in_alu_op;
    wire                 in_rs2_src;
    wire[31:0]           in_itype_immed;
`DEBUG_BEGIN
    wire[2:0]            in_branch_type;
`DEBUG_END
    wire[19:0]           in_upper_immed;
    wire                 in_jal;
    wire[31:0]           in_jal_offset;
    wire[31:0]           in_curr_PC;

    assign in_a_reg_data  = exec_unit_req_if.a_reg_data;
    assign in_b_reg_data  = exec_unit_req_if.b_reg_data;
    assign in_alu_op      = exec_unit_req_if.alu_op;
    assign in_rs2_src     = exec_unit_req_if.rs2_src;
    assign in_itype_immed = exec_unit_req_if.itype_immed;
    assign in_branch_type = exec_unit_req_if.branch_type;
    assign in_upper_immed = exec_unit_req_if.upper_immed;
    assign in_jal         = exec_unit_req_if.jal;
    assign in_jal_offset  = exec_unit_req_if.jal_offset;
    assign in_curr_PC     = exec_unit_req_if.curr_PC;

    wire[`NUM_THREADS-1:0][31:0]  alu_result;
    wire[`NUM_THREADS-1:0]  alu_stall;

    genvar i;
    generate
        for (i = 0; i < `NUM_THREADS; i++) begin : alu_defs
            VX_alu_unit alu_unit (
                .clk            (clk),
                .reset          (reset),
                .src_a          (in_a_reg_data[i]),
                .src_b          (in_b_reg_data[i]),
                .src_rs2        (in_rs2_src),
                .itype_immed    (in_itype_immed),
                .upper_immed    (in_upper_immed),
                .alu_op         (in_alu_op),
                .curr_PC        (in_curr_PC),
                .alu_result     (alu_result[i]),
                .alu_stall      (alu_stall[i])
            );
        end
    endgenerate

    wire internal_stall = (| alu_stall);

    assign delay = no_slot_exec || internal_stall;

`DEBUG_BEGIN
    wire [$clog2(`NUM_THREADS)-1:0] jal_branch_use_index;
    wire jal_branch_found_valid;
`DEBUG_END

    VX_priority_encoder #(
        .N(`NUM_THREADS)
    ) choose_alu_result (
        .data_in   (exec_unit_req_if.valid),
        .data_out  (jal_branch_use_index),
        .valid_out (jal_branch_found_valid)
    );

    wire[31:0] branch_use_alu_result = alu_result[jal_branch_use_index];

    reg temp_branch_dir;
    always @(*)
    begin
        case (exec_unit_req_if.branch_type)
            `BR_EQ:  temp_branch_dir = (branch_use_alu_result     == 0);
            `BR_NE:  temp_branch_dir = (branch_use_alu_result     != 0);
            `BR_LT:  temp_branch_dir = (branch_use_alu_result[31] != 0);
            `BR_GT:  temp_branch_dir = (branch_use_alu_result[31] == 0);
            `BR_LTU: temp_branch_dir = (branch_use_alu_result[31] != 0); 
            `BR_GTU: temp_branch_dir = (branch_use_alu_result[31] == 0);
            `BR_NO:  temp_branch_dir = 0;
            default: temp_branch_dir = 0;
        endcase // in_branch_type
    end

    wire[`NUM_THREADS-1:0][31:0] duplicate_PC_data;

    generate
        for (i = 0; i < `NUM_THREADS; i++) begin
            assign duplicate_PC_data[i] = exec_unit_req_if.next_PC;
        end
    endgenerate
  
    VX_jal_rsp_if       jal_rsp_temp_if();
    VX_branch_rsp_if    branch_rsp_temp_if();

    // Actual Writeback
    assign inst_exec_wb_if.rd       = exec_unit_req_if.rd;
    assign inst_exec_wb_if.wb       = exec_unit_req_if.wb;
    assign inst_exec_wb_if.valid    = exec_unit_req_if.valid & {`NUM_THREADS{!internal_stall}};
    assign inst_exec_wb_if.warp_num = exec_unit_req_if.warp_num;
    assign inst_exec_wb_if.data     = exec_unit_req_if.jal ? duplicate_PC_data : alu_result;
    assign inst_exec_wb_if.curr_PC  = in_curr_PC;

    // Jal rsp
    assign jal_rsp_temp_if.valid    = in_jal;
    assign jal_rsp_temp_if.dest     = $signed(in_a_reg_data[jal_branch_use_index]) + $signed(in_jal_offset);
    assign jal_rsp_temp_if.warp_num = exec_unit_req_if.warp_num;

    // Branch rsp
    assign branch_rsp_temp_if.valid    = (exec_unit_req_if.branch_type != `BR_NO) && (| exec_unit_req_if.valid);
    assign branch_rsp_temp_if.dir      = temp_branch_dir;
    assign branch_rsp_temp_if.warp_num = exec_unit_req_if.warp_num;
    assign branch_rsp_temp_if.dest     = $signed(exec_unit_req_if.curr_PC) + ($signed(exec_unit_req_if.itype_immed) << 1); // itype_immed = branch_offset

    VX_generic_register #(
        .N(33 + `NW_BITS-1 + 1)
    ) jal_reg (
        .clk   (clk),
        .reset (reset),
        .stall (1'b0),
        .flush (1'b0),
        .in    ({jal_rsp_temp_if.valid, jal_rsp_temp_if.dest, jal_rsp_temp_if.warp_num}),
        .out   ({jal_rsp_if.valid     , jal_rsp_if.dest     , jal_rsp_if.warp_num})
    );

    VX_generic_register #(
        .N(34 + `NW_BITS-1 + 1)
    ) branch_reg (
        .clk   (clk),
        .reset (reset),
        .stall (1'b0),
        .flush (1'b0),
        .in    ({branch_rsp_temp_if.valid, branch_rsp_temp_if.dir, branch_rsp_temp_if.warp_num, branch_rsp_temp_if.dest}),
        .out   ({branch_rsp_if.valid     , branch_rsp_if.dir     , branch_rsp_if.warp_num     , branch_rsp_if.dest     })
    );

endmodule : VX_exec_unit