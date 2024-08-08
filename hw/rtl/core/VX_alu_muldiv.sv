// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_alu_muldiv #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_commit_if.master commit_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS  = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH = `UP(PID_BITS);
    localparam TAG_WIDTH = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + PID_WIDTH + 1 + 1;

    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [`INST_M_BITS-1:0] muldiv_op = `INST_M_BITS'(execute_if.data.op_type);

    wire is_mulx_op = `INST_M_IS_MULX(muldiv_op);
    wire is_signed_op = `INST_M_SIGNED(muldiv_op);
`ifdef XLEN_64
    wire is_alu_w = execute_if.data.op_args.alu.is_w;
`else
    wire is_alu_w = 0;
`endif

    wire [NUM_LANES-1:0][`XLEN-1:0] mul_result_out;
    wire [`UUID_WIDTH-1:0] mul_uuid_out;
    wire [`NW_WIDTH-1:0] mul_wid_out;
    wire [NUM_LANES-1:0] mul_tmask_out;
    wire [`PC_BITS-1:0] mul_PC_out;
    wire [`NR_BITS-1:0] mul_rd_out;
    wire mul_wb_out;
    wire [PID_WIDTH-1:0] mul_pid_out;
    wire mul_sop_out, mul_eop_out;

    wire mul_valid_in = execute_if.valid && is_mulx_op;
    wire mul_ready_in;
    wire mul_valid_out;
    wire mul_ready_out;

    wire is_mulh_in      = `INST_M_IS_MULH(muldiv_op);
    wire is_signed_mul_a = `INST_M_SIGNED_A(muldiv_op);
    wire is_signed_mul_b = is_signed_op;

`ifdef IMUL_DPI

    wire [NUM_LANES-1:0][`XLEN-1:0] mul_result_tmp;

    wire mul_fire_in = mul_valid_in && mul_ready_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [`XLEN-1:0] mul_resultl, mul_resulth;
        wire [`XLEN-1:0] mul_in1 = is_alu_w ? (execute_if.data.rs1_data[i] & `XLEN'hFFFFFFFF) : execute_if.data.rs1_data[i];
        wire [`XLEN-1:0] mul_in2 = is_alu_w ? (execute_if.data.rs2_data[i] & `XLEN'hFFFFFFFF) : execute_if.data.rs2_data[i];
        always @(*) begin
            dpi_imul (mul_fire_in, is_signed_mul_a, is_signed_mul_b, mul_in1, mul_in2, mul_resultl, mul_resulth);
        end
        assign mul_result_tmp[i] = is_mulh_in ? mul_resulth : (is_alu_w ? `XLEN'($signed(mul_resultl[31:0])) : mul_resultl);
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + (NUM_LANES * `XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in, execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, mul_result_tmp}),
        .data_out ({mul_valid_out, mul_uuid_out, mul_wid_out, mul_tmask_out, mul_PC_out, mul_rd_out, mul_wb_out, mul_pid_out, mul_sop_out, mul_eop_out, mul_result_out})
    );

    assign mul_ready_in = mul_ready_out || ~mul_valid_out;

`else

    wire [NUM_LANES-1:0][2*(`XLEN+1)-1:0] mul_result_tmp;
    wire is_mulh_out;
    wire is_mul_w_out;

`ifdef XLEN_64

    wire [NUM_LANES-1:0][`XLEN:0] mul_in1;
    wire [NUM_LANES-1:0][`XLEN:0] mul_in2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign mul_in1[i] = is_alu_w ? {{(`XLEN-31){execute_if.data.rs1_data[i][31]}}, execute_if.data.rs1_data[i][31:0]} : {is_signed_mul_a && execute_if.data.rs1_data[i][`XLEN-1], execute_if.data.rs1_data[i]};
        assign mul_in2[i] = is_alu_w ? {{(`XLEN-31){execute_if.data.rs2_data[i][31]}}, execute_if.data.rs2_data[i][31:0]} : {is_signed_mul_b && execute_if.data.rs2_data[i][`XLEN-1], execute_if.data.rs2_data[i]};
    end

    wire mul_strode;
    wire mul_busy;

    VX_elastic_adapter mul_elastic_adapter (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mul_valid_in),
        .ready_in  (mul_ready_in),
        .valid_out (mul_valid_out),
        .ready_out (mul_ready_out),
        .strobe    (mul_strode),
        .busy      (mul_busy)
    );

    VX_serial_mul #(
        .A_WIDTH (`XLEN+1),
        .LANES   (NUM_LANES),
        .SIGNED  (1)
    ) serial_mul (
        .clk       (clk),
        .reset     (reset),

        .strobe    (mul_strode),
        .busy      (mul_busy),

        .dataa     (mul_in1),
        .datab     (mul_in2),
        .result    (mul_result_tmp)
    );

    reg [TAG_WIDTH+2-1:0] mul_tag_r;
    always @(posedge clk) begin
        if (mul_valid_in && mul_ready_in) begin
            mul_tag_r <= {execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, is_mulh_in, is_alu_w, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop};
        end
    end

    assign {mul_uuid_out, mul_wid_out, mul_tmask_out, mul_PC_out, mul_rd_out, mul_wb_out, is_mulh_out, is_mul_w_out, mul_pid_out, mul_sop_out, mul_eop_out} = mul_tag_r;

`else

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [`XLEN:0] mul_in1 = {is_signed_mul_a && execute_if.data.rs1_data[i][`XLEN-1], execute_if.data.rs1_data[i]};
        wire [`XLEN:0] mul_in2 = {is_signed_mul_b && execute_if.data.rs2_data[i][`XLEN-1], execute_if.data.rs2_data[i]};

        VX_multiplier #(
            .A_WIDTH (`XLEN+1),
            .B_WIDTH (`XLEN+1),
            .R_WIDTH (2*(`XLEN+1)),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (mul_ready_in),
            .dataa  (mul_in1),
            .datab  (mul_in2),
            .result (mul_result_tmp[i])
        );
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + 1 + 1),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in, execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, is_mulh_in, is_alu_w}),
        .data_out ({mul_valid_out, mul_uuid_out, mul_wid_out, mul_tmask_out, mul_PC_out, mul_rd_out, mul_wb_out, mul_pid_out, mul_sop_out, mul_eop_out, is_mulh_out, is_mul_w_out})
    );

    assign mul_ready_in = mul_ready_out || ~mul_valid_out;

`endif

    for (genvar i = 0; i < NUM_LANES; ++i) begin
    `ifdef XLEN_64
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`XLEN)-1:`XLEN] :
                                                 (is_mul_w_out ? `XLEN'($signed(mul_result_tmp[i][31:0])) :
                                                                 mul_result_tmp[i][`XLEN-1:0]);
    `else
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`XLEN)-1:`XLEN] : mul_result_tmp[i][`XLEN-1:0];
        `UNUSED_VAR (is_mul_w_out)
    `endif
    end

`endif

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`XLEN-1:0] div_result_out;
    wire [`UUID_WIDTH-1:0] div_uuid_out;
    wire [`NW_WIDTH-1:0] div_wid_out;
    wire [NUM_LANES-1:0] div_tmask_out;
    wire [`PC_BITS-1:0] div_PC_out;
    wire [`NR_BITS-1:0] div_rd_out;
    wire div_wb_out;
    wire [PID_WIDTH-1:0] div_pid_out;
    wire div_sop_out, div_eop_out;

    wire is_rem_op = `INST_M_IS_REM(muldiv_op);

    wire div_valid_in = execute_if.valid && ~is_mulx_op;
    wire div_ready_in;
    wire div_valid_out;
    wire div_ready_out;

    wire [NUM_LANES-1:0][`XLEN-1:0] div_in1;
    wire [NUM_LANES-1:0][`XLEN-1:0] div_in2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
    `ifdef XLEN_64
        assign div_in1[i] = is_alu_w ? {{(`XLEN-32){is_signed_op && execute_if.data.rs1_data[i][31]}}, execute_if.data.rs1_data[i][31:0]}: execute_if.data.rs1_data[i];
        assign div_in2[i] = is_alu_w ? {{(`XLEN-32){is_signed_op && execute_if.data.rs2_data[i][31]}}, execute_if.data.rs2_data[i][31:0]}: execute_if.data.rs2_data[i];
    `else
        assign div_in1[i] = execute_if.data.rs1_data[i];
        assign div_in2[i] = execute_if.data.rs2_data[i];
    `endif
    end

`ifdef IDIV_DPI

    wire [NUM_LANES-1:0][`XLEN-1:0] div_result_in;
    wire div_fire_in = div_valid_in && div_ready_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [`XLEN-1:0] div_quotient, div_remainder;
        always @(*) begin
            dpi_idiv (div_fire_in, is_signed_op, div_in1[i], div_in2[i], div_quotient, div_remainder);
        end
        assign div_result_in[i] = is_rem_op ? (is_alu_w ? `XLEN'($signed(div_remainder[31:0])) : div_remainder) :
                                              (is_alu_w ? `XLEN'($signed(div_quotient[31:0])) : div_quotient);
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + (NUM_LANES * `XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) div_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (div_ready_in),
        .data_in  ({div_valid_in, execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, div_result_in}),
        .data_out ({div_valid_out, div_uuid_out, div_wid_out, div_tmask_out, div_PC_out, div_rd_out, div_wb_out, div_pid_out, div_sop_out, div_eop_out, div_result_out})
    );

    assign div_ready_in = div_ready_out || ~div_valid_out;

`else

    wire [NUM_LANES-1:0][`XLEN-1:0] div_quotient, div_remainder;
    wire is_rem_op_out;
    wire is_div_w_out;
    wire div_strode;
    wire div_busy;

    VX_elastic_adapter div_elastic_adapter (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (div_valid_in),
        .ready_in  (div_ready_in),
        .valid_out (div_valid_out),
        .ready_out (div_ready_out),
        .strobe    (div_strode),
        .busy      (div_busy)
    );

    VX_serial_div #(
        .WIDTHN (`XLEN),
        .WIDTHD (`XLEN),
        .WIDTHQ (`XLEN),
        .WIDTHR (`XLEN),
        .LANES  (NUM_LANES)
    ) serial_div (
        .clk       (clk),
        .reset     (reset),

        .strobe    (div_strode),
        .busy      (div_busy),

        .is_signed (is_signed_op),
        .numer     (div_in1),
        .denom     (div_in2),

        .quotient  (div_quotient),
        .remainder (div_remainder)
    );

    reg [TAG_WIDTH+2-1:0] div_tag_r;
    always @(posedge clk) begin
        if (div_valid_in && div_ready_in) begin
            div_tag_r <= {execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, is_rem_op, is_alu_w, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop};
        end
    end

    assign {div_uuid_out, div_wid_out, div_tmask_out, div_PC_out, div_rd_out, div_wb_out, is_rem_op_out, is_div_w_out, div_pid_out, div_sop_out, div_eop_out} = div_tag_r;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
    `ifdef XLEN_64
        assign div_result_out[i] = is_rem_op_out ? (is_div_w_out ? `XLEN'($signed(div_remainder[i][31:0])) : div_remainder[i]) :
                                                   (is_div_w_out ? `XLEN'($signed(div_quotient[i][31:0])) : div_quotient[i]);
    `else
        assign div_result_out[i] = is_rem_op_out ? div_remainder[i] : div_quotient[i];
        `UNUSED_VAR (is_div_w_out)
    `endif
    end

`endif

    // can accept new request?
    assign execute_if.ready = is_mulx_op ? mul_ready_in : div_ready_in;

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW (TAG_WIDTH + (NUM_LANES * `XLEN)),
        .ARBITER ("F"),
        .OUT_BUF (1)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({div_valid_out, mul_valid_out}),
        .ready_in  ({div_ready_out, mul_ready_out}),
        .data_in   ({{div_uuid_out, div_wid_out, div_tmask_out, div_PC_out, div_rd_out, div_wb_out, div_pid_out, div_sop_out, div_eop_out, div_result_out},
                     {mul_uuid_out, mul_wid_out, mul_tmask_out, mul_PC_out, mul_rd_out, mul_wb_out, mul_pid_out, mul_sop_out, mul_eop_out, mul_result_out}}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.rd, commit_if.data.wb, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop, commit_if.data.data}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
