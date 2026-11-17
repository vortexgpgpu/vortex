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

module VX_alu_muldiv import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1,
    parameter IDIV_PE_RATIO = `VX_CFG_IDIV_PE_RATIO
) (
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.rs3_data)

    localparam TAG_WIDTH = $bits(alu_header_t);

    wire [INST_M_BITS-1:0] muldiv_op = INST_M_BITS'(execute_if.data.op_type);

    wire is_mulx_op = inst_m_is_mulx(muldiv_op);
    wire is_signed_op = inst_m_signed(muldiv_op);
`ifdef VX_CFG_XLEN_64
    wire is_alu_w = execute_if.data.op_args.alu.is_w;
`else
    wire is_alu_w = 0;
`endif

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] mul_result_out;
    alu_header_t mul_hdr_out;

    wire mul_valid_in = execute_if.valid && is_mulx_op;
    wire mul_ready_in;
    wire mul_valid_out;
    wire mul_ready_out;

    wire is_mulh_in      = inst_m_is_mulh(muldiv_op);
    wire is_signed_mul_a = inst_m_signed_a(muldiv_op);
    wire is_signed_mul_b = is_signed_op;

`ifdef IMUL_DPI

    import "DPI-C" function void dpi_imul(input logic enable, input logic is_signed_a, input logic is_signed_b, input int a, input int b, output int resultl, output int resulth);
    import "DPI-C" function void dpi_lmul(input logic enable, input logic is_signed_a, input logic is_signed_b, input longint a, input longint b, output longint resultl, output longint resulth);

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] mul_result_tmp;

    wire mul_fire_in = mul_valid_in && mul_ready_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mul_result_tmp
        reg [`VX_CFG_XLEN-1:0] mul_resultl, mul_resulth;
        wire [`VX_CFG_XLEN-1:0] mul_in1 = is_alu_w ? (execute_if.data.rs1_data[i] & `VX_CFG_XLEN'(32'hFFFFFFFF)) : execute_if.data.rs1_data[i];
        wire [`VX_CFG_XLEN-1:0] mul_in2 = is_alu_w ? (execute_if.data.rs2_data[i] & `VX_CFG_XLEN'(32'hFFFFFFFF)) : execute_if.data.rs2_data[i];
        always @(*) begin
        `ifdef VX_CFG_XLEN_64
            dpi_lmul (mul_fire_in, is_signed_mul_a, is_signed_mul_b, mul_in1, mul_in2, mul_resultl, mul_resulth);
        `else
            dpi_imul (mul_fire_in, is_signed_mul_a, is_signed_mul_b, mul_in1, mul_in2, mul_resultl, mul_resulth);
        `endif
        end
        assign mul_result_tmp[i] = is_mulh_in ? mul_resulth : (is_alu_w ? `VX_CFG_XLEN'($signed(mul_resultl[31:0])) : mul_resultl);
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + (NUM_LANES * `VX_CFG_XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,  execute_if.data.header, mul_result_tmp}),
        .data_out ({mul_valid_out, mul_hdr_out,            mul_result_out})
    );

    assign mul_ready_in = mul_ready_out || ~mul_valid_out;

`else

    wire [NUM_LANES-1:0][2*(`VX_CFG_XLEN+1)-1:0] mul_result_tmp;
    wire is_mulh_out;
    wire is_mul_w_out;

`ifdef VX_CFG_XLEN_64

    wire [NUM_LANES-1:0][`VX_CFG_XLEN:0] mul_in1;
    wire [NUM_LANES-1:0][`VX_CFG_XLEN:0] mul_in2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mul_in
        assign mul_in1[i] = is_alu_w ? {{(`VX_CFG_XLEN-31){execute_if.data.rs1_data[i][31]}}, execute_if.data.rs1_data[i][31:0]} : {is_signed_mul_a && execute_if.data.rs1_data[i][`VX_CFG_XLEN-1], execute_if.data.rs1_data[i]};
        assign mul_in2[i] = is_alu_w ? {{(`VX_CFG_XLEN-31){execute_if.data.rs2_data[i][31]}}, execute_if.data.rs2_data[i][31:0]} : {is_signed_mul_b && execute_if.data.rs2_data[i][`VX_CFG_XLEN-1], execute_if.data.rs2_data[i]};
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
        .A_WIDTH (`VX_CFG_XLEN+1),
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
            mul_tag_r <= {execute_if.data.header, is_mulh_in, is_alu_w};
        end
    end

    assign {mul_hdr_out, is_mulh_out, is_mul_w_out} = mul_tag_r;

`else

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_multiplier
        wire [`VX_CFG_XLEN:0] mul_in1 = {is_signed_mul_a && execute_if.data.rs1_data[i][`VX_CFG_XLEN-1], execute_if.data.rs1_data[i]};
        wire [`VX_CFG_XLEN:0] mul_in2 = {is_signed_mul_b && execute_if.data.rs2_data[i][`VX_CFG_XLEN-1], execute_if.data.rs2_data[i]};

        // Where the target has no pipelined multiplier blocks, the first of the
        // LATENCY_IMUL stages registers the operands, so the multiply array
        // gets a cycle of its own rather than following the operand select.
        wire [`VX_CFG_XLEN:0] mul_in1_r, mul_in2_r;
        VX_pipe_register #(
            .DATAW (2 * (`VX_CFG_XLEN+1)),
            .DEPTH (`IMUL_IN_REG)
        ) mul_in_reg (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (mul_ready_in),
            .data_in  ({mul_in1,   mul_in2}),
            .data_out ({mul_in1_r, mul_in2_r})
        );

        VX_multiplier #(
            .A_WIDTH (`VX_CFG_XLEN+1),
            .B_WIDTH (`VX_CFG_XLEN+1),
            .R_WIDTH (2*(`VX_CFG_XLEN+1)),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL - `IMUL_IN_REG)
        ) multiplier (
            .clk    (clk),
            .enable (mul_ready_in),
            .dataa  (mul_in1_r),
            .datab  (mul_in2_r),
            .result (mul_result_tmp[i])
        );
    end
    `STATIC_ASSERT (`LATENCY_IMUL >= `IMUL_IN_REG, ("invalid parameter"))

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + 1 + 1),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,  execute_if.data.header, is_mulh_in,  is_alu_w}),
        .data_out ({mul_valid_out, mul_hdr_out,            is_mulh_out, is_mul_w_out})
    );

    assign mul_ready_in = mul_ready_out || ~mul_valid_out;

`endif

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mul_result_out
    `ifdef VX_CFG_XLEN_64
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`VX_CFG_XLEN)-1:`VX_CFG_XLEN] :
                                                 (is_mul_w_out ? `VX_CFG_XLEN'($signed(mul_result_tmp[i][31:0])) :
                                                                 mul_result_tmp[i][`VX_CFG_XLEN-1:0]);
    `else
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`VX_CFG_XLEN)-1:`VX_CFG_XLEN] : mul_result_tmp[i][`VX_CFG_XLEN-1:0];
        `UNUSED_VAR (is_mul_w_out)
    `endif
    end

`endif

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] div_result_out;
    alu_header_t div_hdr_out;

    wire is_rem_op = inst_m_is_rem(muldiv_op);

    wire div_valid_in = execute_if.valid && ~is_mulx_op;
    wire div_ready_in;
    wire div_valid_out;
    wire div_ready_out;

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] div_in1;
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] div_in2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_div_in
    `ifdef VX_CFG_XLEN_64
        assign div_in1[i] = is_alu_w ? {{(`VX_CFG_XLEN-32){is_signed_op && execute_if.data.rs1_data[i][31]}}, execute_if.data.rs1_data[i][31:0]}: execute_if.data.rs1_data[i];
        assign div_in2[i] = is_alu_w ? {{(`VX_CFG_XLEN-32){is_signed_op && execute_if.data.rs2_data[i][31]}}, execute_if.data.rs2_data[i][31:0]}: execute_if.data.rs2_data[i];
    `else
        assign div_in1[i] = execute_if.data.rs1_data[i];
        assign div_in2[i] = execute_if.data.rs2_data[i];
    `endif
    end

    // A divide runs on NUM_PES_DIV divider lanes, one batch of lanes after
    // another, and holds the input until its last batch starts.
    localparam NUM_PES_DIV = `UP(NUM_LANES / IDIV_PE_RATIO);
    localparam BATCHES_DIV = NUM_LANES / NUM_PES_DIV;
    localparam BATCH_BITS  = `LOG2UP(BATCHES_DIV);
    `STATIC_ASSERT ((NUM_LANES % NUM_PES_DIV) == 0, ("invalid parameter"))

    wire div_last;

`ifdef IDIV_DPI

    import "DPI-C" function void dpi_idiv(input logic enable, input logic is_signed, input int a, input int b, output int quotient, output int remainder);
    import "DPI-C" function void dpi_ldiv(input logic enable, input logic is_signed, input longint a, input longint b, output longint quotient, output longint remainder);

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] div_result_in;
    wire div_pipe_ready = div_ready_out || ~div_valid_out;

    if (BATCHES_DIV > 1) begin : g_batch
        reg [BATCH_BITS-1:0] batch_idx;
        always @(posedge clk) begin
            if (reset) begin
                batch_idx <= '0;
            end else if (div_valid_in && div_pipe_ready) begin
                batch_idx <= div_last ? '0 : (batch_idx + 1'b1);
            end
        end
        assign div_last = (batch_idx == BATCH_BITS'(BATCHES_DIV-1));
    end else begin : g_no_batch
        assign div_last = 1'b1;
    end

    assign div_ready_in = div_pipe_ready && div_last;
    wire div_fire_in = div_valid_in && div_ready_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_div_result_in
        reg [`VX_CFG_XLEN-1:0] div_quotient, div_remainder;
        always @(*) begin
        `ifdef VX_CFG_XLEN_64
            dpi_ldiv (div_fire_in, is_signed_op, div_in1[i], div_in2[i], div_quotient, div_remainder);
        `else
            dpi_idiv (div_fire_in, is_signed_op, div_in1[i], div_in2[i], div_quotient, div_remainder);
        `endif
        end
        assign div_result_in[i] = is_rem_op ? (is_alu_w ? `VX_CFG_XLEN'($signed(div_remainder[31:0])) : div_remainder) :
                                              (is_alu_w ? `VX_CFG_XLEN'($signed(div_quotient[31:0])) : div_quotient);
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH + (NUM_LANES * `VX_CFG_XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) div_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (div_pipe_ready),
        .data_in  ({div_valid_in && div_last, execute_if.data.header, div_result_in}),
        .data_out ({div_valid_out,            div_hdr_out,            div_result_out})
    );

`else

    wire [BATCHES_DIV-1:0][NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] div_in1_b = div_in1;
    wire [BATCHES_DIV-1:0][NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] div_in2_b = div_in2;
    wire [NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] pe_numer, pe_denom;
    wire [NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] div_quotient, div_remainder;
    wire is_rem_op_out;
    wire is_div_w_out;
    wire div_strobe;
    wire div_busy;
    wire batch_run;

    // the last batch started: the divide owns the divider until its result leaves
    reg div_loaded;
    always @(posedge clk) begin
        if (reset) begin
            div_loaded <= 0;
        end else begin
            if (div_valid_in && div_ready_in) begin
                div_loaded <= 1;
            end
            if (div_valid_out && div_ready_out) begin
                div_loaded <= 0;
            end
        end
    end

    assign div_strobe    = div_valid_in && ~div_loaded && ~batch_run;
    assign div_ready_in  = ~div_loaded && ~batch_run && div_last;
    assign div_valid_out = div_loaded && ~div_busy;

    // Each divider lane's result: the quotient or remainder, sign-extended from
    // 32 bits for the W forms. pe_result selects by the held input, for the
    // batches before the last; pe_result_out by the last batch's tag.
    wire [NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] pe_result, pe_result_out;
    for (genvar j = 0; j < NUM_PES_DIV; ++j) begin : g_pe_result
        wire [`VX_CFG_XLEN-1:0] value     = is_rem_op ? div_remainder[j] : div_quotient[j];
        wire [`VX_CFG_XLEN-1:0] value_out = is_rem_op_out ? div_remainder[j] : div_quotient[j];
    `ifdef VX_CFG_XLEN_64
        assign pe_result[j]     = is_alu_w ? `VX_CFG_XLEN'($signed(value[31:0])) : value;
        assign pe_result_out[j] = is_div_w_out ? `VX_CFG_XLEN'($signed(value_out[31:0])) : value_out;
    `else
        assign pe_result[j]     = value;
        assign pe_result_out[j] = value_out;
    `endif
    end
`ifndef VX_CFG_XLEN_64
    `UNUSED_VAR (is_div_w_out)
`endif

    if (BATCHES_DIV > 1) begin : g_batch
        reg [BATCH_BITS-1:0] batch_idx;
        reg batch_run_r;
        // the results of the batches before the last, held until the last finishes
        reg [BATCHES_DIV-2:0][NUM_PES_DIV-1:0][`VX_CFG_XLEN-1:0] batch_res;
        wire batch_done = batch_run_r && ~div_busy;

        always @(posedge clk) begin
            if (reset) begin
                batch_idx   <= '0;
                batch_run_r <= 0;
            end else begin
                if (div_strobe) begin
                    batch_run_r <= ~div_last;
                    if (div_last) begin
                        batch_idx <= '0;
                    end
                end
                if (batch_done) begin
                    batch_run_r <= 0;
                    batch_idx   <= batch_idx + 1'b1;
                end
            end
        end

        // the input is held through the batches, so its op still selects the result
        for (genvar b = 0; b < BATCHES_DIV-1; ++b) begin : g_batch_res
            for (genvar j = 0; j < NUM_PES_DIV; ++j) begin : g_lanes
                always @(posedge clk) begin
                    if (batch_done && batch_idx == BATCH_BITS'(b)) begin
                        batch_res[b][j] <= pe_result[j];
                    end
                end
                assign div_result_out[b * NUM_PES_DIV + j] = batch_res[b][j];
            end
        end

        assign div_last  = (batch_idx == BATCH_BITS'(BATCHES_DIV-1));
        assign batch_run = batch_run_r;
        assign pe_numer  = div_in1_b[batch_idx];
        assign pe_denom  = div_in2_b[batch_idx];
    end else begin : g_no_batch
        assign div_last  = 1'b1;
        assign batch_run = 1'b0;
        assign pe_numer  = div_in1_b[0];
        assign pe_denom  = div_in2_b[0];
        `UNUSED_VAR (pe_result)
    end

    VX_serial_div #(
        .WIDTHN (`VX_CFG_XLEN),
        .WIDTHD (`VX_CFG_XLEN),
        .WIDTHQ (`VX_CFG_XLEN),
        .WIDTHR (`VX_CFG_XLEN),
        .LANES  (NUM_PES_DIV)
    ) serial_div (
        .clk       (clk),
        .reset     (reset),

        .strobe    (div_strobe),
        .busy      (div_busy),

        .is_signed (is_signed_op),
        .numer     (pe_numer),
        .denom     (pe_denom),

        .quotient  (div_quotient),
        .remainder (div_remainder)
    );

    reg [TAG_WIDTH+2-1:0] div_tag_r;
    always @(posedge clk) begin
        if (div_valid_in && div_ready_in) begin
            div_tag_r <= {execute_if.data.header, is_rem_op, is_alu_w};
        end
    end

    assign {div_hdr_out, is_rem_op_out, is_div_w_out} = div_tag_r;

    // the last batch's lanes leave straight from the divider
    for (genvar j = 0; j < NUM_PES_DIV; ++j) begin : g_div_result_out
        assign div_result_out[(BATCHES_DIV-1) * NUM_PES_DIV + j] = pe_result_out[j];
    end

`endif

    // can accept new request?
    assign execute_if.ready = is_mulx_op ? mul_ready_in : div_ready_in;

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW (TAG_WIDTH + (NUM_LANES * `VX_CFG_XLEN)),
        .ARBITER ("P"),
        .OUT_BUF (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({div_valid_out, mul_valid_out}),
        .ready_in  ({div_ready_out, mul_ready_out}),
        .data_in   ({{div_hdr_out, div_result_out},
                     {mul_hdr_out, mul_result_out}}),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
