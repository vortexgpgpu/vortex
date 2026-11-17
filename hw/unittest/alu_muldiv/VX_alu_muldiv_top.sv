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

// Self-checking harness for VX_alu_muldiv's synthesizable divider: one unit per
// IDIV_PE_RATIO of 1, 2, 4, 8 and 16 (past the lane count, one divider lane).
// Each gets random multiplies, divides and remainders, with corner operands,
// random input gaps and random output stalls, and checks every lane of every
// result against a reference model. Results are matched by an op id carried in
// the PC field, since multiplies and divides can leave out of order.
module VX_alu_muldiv_top import VX_gpu_pkg::*; #(
    parameter NUM_OPS = 500
) (
    input  wire        clk,
    input  wire        reset,
    output wire [31:0] errors,
    output wire        done
);
    localparam L          = `VX_CFG_NUM_ALU_LANES;
    localparam X          = `VX_CFG_XLEN;
    localparam NUM_SHAPES = 5;
    localparam ID_W       = `CLOG2(NUM_OPS);

    function automatic logic [X-1:0] ref_op(
        input logic [INST_M_BITS-1:0] op,
        input logic is_w,
        input logic [X-1:0] a_in,
        input logic [X-1:0] b_in
    );
        logic [X-1:0] a, b, r;
        logic sgn = inst_m_signed(op);
        a = a_in;
        b = b_in;
        if (X == 64 && is_w) begin
            a = sgn ? X'($signed(a_in[31:0])) : X'(a_in[31:0]);
            b = sgn ? X'($signed(b_in[31:0])) : X'(b_in[31:0]);
        end
        case (op)
        INST_M_MUL:  r = a * b;
        INST_M_DIV:  r = (b == 0) ? '1 : ((a == {1'b1, {(X-1){1'b0}}} && b == '1) ? a : X'($signed(a) / $signed(b)));
        INST_M_DIVU: r = (b == 0) ? '1 : (a / b);
        INST_M_REM:  r = (b == 0) ? a : ((a == {1'b1, {(X-1){1'b0}}} && b == '1) ? '0 : X'($signed(a) % $signed(b)));
        default:     r = (b == 0) ? a : (a % b);
        endcase
        if (X == 64 && is_w) begin
            r = X'($signed(r[31:0]));
        end
        return r;
    endfunction

    function automatic logic [X-1:0] rand_operand();
        logic [X-1:0] v = X'({$urandom, $urandom});
        case ($urandom % 8)
        0: v = '0;
        1: v = '1;
        2: v = {1'b1, {(X-1){1'b0}}};
        3: v = X'($urandom % 7);
        4: v = X'({$urandom, 32'h80000000 | $urandom}); // bit 31 set, bit 63 random
        default:;
        endcase
        return v;
    endfunction

    wire [NUM_SHAPES-1:0][31:0] shape_errors;
    wire [NUM_SHAPES-1:0]       shape_done;

    for (genvar s = 0; s < NUM_SHAPES; ++s) begin : g_shapes
        VX_execute_if #(.data_t (alu_execute_t)) execute_if();
        VX_result_if  #(.data_t (alu_result_t))  result_if();

        VX_alu_muldiv #(
            .NUM_LANES     (L),
            .IDIV_PE_RATIO (1 << s)
        ) dut (
            .clk        (clk),
            .reset      (reset),
            .execute_if (execute_if),
            .result_if  (result_if)
        );

        logic [L-1:0][X-1:0] expected [NUM_OPS];
        logic [NUM_OPS-1:0]  received;
        alu_execute_t        in_data;
        logic                in_valid;
        logic                out_ready;
        int                  sent, recv, errs;

        `UNUSED_VAR (execute_if.data)

        assign execute_if.valid = in_valid;
        assign execute_if.data  = in_data;
        assign result_if.ready  = out_ready;

        wire in_fire  = in_valid && execute_if.ready;
        wire out_fire = result_if.valid && out_ready;

        always @(posedge clk) begin
            if (reset) begin
                in_valid  <= 0;
                out_ready <= 0;
                sent      <= 0;
            end else begin
                out_ready <= ($urandom % 4) != 0;
                if (in_fire) begin
                    sent     <= sent + 1;
                    in_valid <= 0;
                end
                if ((~in_valid || in_fire) && (sent + int'(in_fire)) < NUM_OPS && ($urandom % 3) != 0) begin
                    automatic logic [ID_W-1:0] id = ID_W'(sent + int'(in_fire));
                    automatic logic [INST_M_BITS-1:0] op = (($urandom % 4) == 0) ? INST_M_MUL
                                                         : (INST_M_DIV | INST_M_BITS'($urandom % 4));
                    automatic logic is_w = (X == 64) ? 1'($urandom) : 1'b0;
                    automatic alu_execute_t next = '0;
                    next.header.PC = PC_BITS'(id);
                    next.op_type   = INST_OP_BITS'(op);
                `ifdef VX_CFG_XLEN_64
                    next.op_args.alu.is_w = is_w;
                `endif
                    for (int i = 0; i < L; ++i) begin
                        next.rs1_data[i] = rand_operand();
                        next.rs2_data[i] = rand_operand();
                        expected[id][i] <= ref_op(op, is_w, next.rs1_data[i], next.rs2_data[i]);
                    end
                    in_data  <= next;
                    in_valid <= 1;
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                recv     <= 0;
                errs     <= 0;
                received <= '0;
            end else if (out_fire) begin
                automatic int id = int'(result_if.data.header.PC);
                automatic int bad = 0;
                if (id >= NUM_OPS || received[id]) begin
                    $display("ratio %0d: unexpected result id %0d", 1 << s, id);
                    bad = 1;
                end else begin
                    for (int i = 0; i < L; ++i) begin
                        if (result_if.data.data[i] != expected[id][i]) begin
                            $display("ratio %0d: op %0d lane %0d: got 0x%h, expected 0x%h",
                                1 << s, id, i, result_if.data.data[i], expected[id][i]);
                            bad = 1;
                        end
                    end
                    received[id] <= 1;
                end
                recv <= recv + 1;
                errs <= errs + bad;
            end
        end

        assign shape_errors[s] = 32'(errs);
        assign shape_done[s]   = (recv == NUM_OPS);
    end

    logic [31:0] total_errors;
    always @(*) begin
        total_errors = '0;
        for (integer s = 0; s < NUM_SHAPES; ++s) begin
            total_errors += shape_errors[s];
        end
    end

    assign errors = total_errors;
    assign done   = (& shape_done);

endmodule
