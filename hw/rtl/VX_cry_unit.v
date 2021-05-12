`include "VX_define.vh"

/*
Vortex Crypto Unit

Inspiration taken from: https://github.com/riscv/riscv-crypto
*/

module VX_cry_unit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_cry_req_if       cry_req_if,

    // Outputs
    // VX_branch_ctl_if    branch_ctl_if,
    VX_commit_if        cry_commit_if
);
    `define ROR32(a,b) ((a >> b) | (a << 32-b))
    // `define ROR32(a,b) (({(a),(a)} >> (b))[31:0])
    `define ROL32(a,b) ((a << b) | (a >> 32-b))
    `define SRL32(a,b) ((a >> b))

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (cry_req_if.next_PC)
    `UNUSED_VAR (cry_req_if.op_mod)
    `UNUSED_VAR (cry_req_if.use_PC)
    `UNUSED_VAR (cry_req_if.tid)

    wire stall_in, stall_out;

    reg [`NUM_THREADS-1:0][31:0]  cry_result;

    wire [`NUM_THREADS-1:0][31:0] sha_sig0_result;
    wire [`NUM_THREADS-1:0][31:0] sha_sig1_result;
    wire [`NUM_THREADS-1:0][31:0] sha_sum0_result;
    wire [`NUM_THREADS-1:0][31:0] sha_sum1_result;
    wire [`NUM_THREADS-1:0][31:0] aes_result;
    wire [`NUM_THREADS-1:0][31:0] ror_result;
    wire [`NUM_THREADS-1:0][31:0] rol_result;

    wire [`CRY_MOD_BITS - 1:0] bytesel = `CRY_MOD(cry_req_if.op_mod);

    wire aes_ready_in;
    wire is_aes = `CRY_AES(cry_req_if.op_type);
    wire aes_ready_out = ~stall_out;
    wire aes_valid_out;
    wire do_esi = (`CRY_OP(cry_req_if.op_type)) == `CRY_AES32ESI;
    wire do_esmi = (`CRY_OP(cry_req_if.op_type)) == `CRY_AES32ESMI;
    wire do_dsi = (`CRY_OP(cry_req_if.op_type)) == `CRY_AES32DSI;
    wire do_dsmi = (`CRY_OP(cry_req_if.op_type)) == `CRY_AES32DSMI;

    wire [`NW_BITS-1:0] aes_wid;
    wire [`NUM_THREADS-1:0] aes_tmask;
    wire [31:0] aes_PC;
    wire [`NR_BITS-1:0] aes_rd;
    wire aes_wb;

    VX_aes aes_logic (
        .clk(clk),
        .reset(reset),
        .rs1_data(cry_req_if.rs1_data),
        .rs2_data(cry_req_if.rs2_data),
        .bs(bytesel),
        .op_saes32_encs(do_esi),
        .op_saes32_encsm(do_esmi),
        .op_saes32_decs(do_dsi),
        .op_saes32_decsm(do_dsmi),

        .wid_in(cry_req_if.wid),
        .tmask_in(cry_req_if.tmask),
        .PC_in(cry_req_if.PC),
        .rd_in(cry_req_if.rd),
        .wb_in(cry_req_if.wb),

        .wid_out(aes_wid),
        .tmask_out(aes_tmask),
        .PC_out(aes_PC),
        .rd_out(aes_rd),
        .wb_out(aes_wb),

        .result(aes_result),
        .valid_in(cry_req_if.valid && is_aes),
        .ready_in(aes_ready_in),
        .valid_out(aes_valid_out),
        .ready_out(aes_ready_out)
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin

        assign sha_sig0_result[i]  = `ROR32(cry_req_if.rs1_data[i],7) ^
                                    `ROR32(cry_req_if.rs1_data[i],18) ^
                                    `SRL32(cry_req_if.rs1_data[i],3);
        assign sha_sig1_result[i]  = `ROR32(cry_req_if.rs1_data[i],17) ^
                                    `ROR32(cry_req_if.rs1_data[i],19) ^
                                    `SRL32(cry_req_if.rs1_data[i],10);
        assign sha_sum0_result[i]  = `ROR32(cry_req_if.rs1_data[i],2) ^
                                    `ROR32(cry_req_if.rs1_data[i],13) ^
                                    `ROR32(cry_req_if.rs1_data[i],22);
        assign sha_sum1_result[i]  = `ROR32(cry_req_if.rs1_data[i],6) ^
                                    `ROR32(cry_req_if.rs1_data[i],11) ^
                                    `ROR32(cry_req_if.rs1_data[i],25);

        assign ror_result[i]       = !cry_req_if.use_imm ?
                                                `ROR32(cry_req_if.rs1_data[i],cry_req_if.rs2_data[i])
                                                : `ROR32(cry_req_if.rs1_data[i],cry_req_if.imm);
        // assign rol_result[i]       = cry_req_if.use_imm ?
        //                                         `ROL32(cry_req_if.rs1_data[i],cry_req_if.rs2_data[i])
        //                                         : `ROL32(cry_req_if.rs1_data[i],cry_req_if.imm);
        assign rol_result[i] = 0;
    end

    wire [`CRY_BITS - 1:0] cry_op = `CRY_OP(cry_req_if.op_type);

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            if (aes_valid_out) begin
                cry_result[i] = aes_result[i];
            end else begin
                case (cry_op)
                    `CRY_SHA256SUM0:    cry_result[i] = sha_sum0_result[i];
                    `CRY_SHA256SUM1:    cry_result[i] = sha_sum1_result[i];
                    `CRY_SHA256SIG0:    cry_result[i] = sha_sig0_result[i];
                    `CRY_SHA256SIG1:    cry_result[i] = sha_sig1_result[i];
                    `CRY_ROR:           cry_result[i] = ror_result[i];
                    `CRY_ROL:           cry_result[i] = rol_result[i];
                    default:            cry_result[i] = 32'hffff0000; // magic number for debugging
                endcase
            end
        end
    end

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (cry_req_if.valid) begin
            case (cry_op)
                `CRY_AES32ESI:      $display("AES_ESI: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_AES32ESMI:     $display("AES_ESMI: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_AES32DSI:      $display("AES_DSI: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_AES32DSMI:     $display("AES_DSMI: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_SHA256SUM0:    $display("SHA_SUM0: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_SHA256SUM1:    $display("SHA_SUM1: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_SHA256SIG0:    $display("SHA_SIG0: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_SHA256SIG1:    $display("SHA_SIG1: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_ROR:           $display("ROR: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                `CRY_ROL:           $display("ROL: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
                default:            $display("INVALID: %t: core%0d-commit: wid=%0d, PC=%0h, ex=CRY, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cry_req_if.wid, cry_req_if.PC, cry_req_if.tmask, cry_req_if.wb, cry_req_if.rd, cry_result);
            endcase
        end
    end
`endif

    // output

    wire                          result_valid;
    wire [`NW_BITS-1:0]           result_wid;
    wire [`NUM_THREADS-1:0]       result_tmask;
    wire [31:0]                   result_PC;
    wire [`NR_BITS-1:0]           result_rd;
    wire                          result_wb;
    wire [`NUM_THREADS-1:0][31:0] result_data;

    assign stall_in = (is_aes && ~aes_ready_in)
                      || (~is_aes && (aes_valid_out || stall_out));

    assign result_valid = aes_valid_out || (~is_aes && cry_req_if.valid);
    assign result_wid   = aes_valid_out ? aes_wid : cry_req_if.wid;
    assign result_tmask = aes_valid_out ? aes_tmask : cry_req_if.tmask;
    assign result_PC    = aes_valid_out ? aes_PC : cry_req_if.PC;
    assign result_rd    = aes_valid_out ? aes_rd : cry_req_if.rd;
    assign result_wb    = aes_valid_out ? aes_wb : cry_req_if.wb;
    assign result_data  = cry_result;

    assign stall_out = ~cry_commit_if.ready && cry_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({result_valid,        result_wid,        result_tmask,        result_PC,        result_rd,        result_wb,        result_data}),
        .data_out ({cry_commit_if.valid, cry_commit_if.wid, cry_commit_if.tmask, cry_commit_if.PC, cry_commit_if.rd, cry_commit_if.wb, cry_commit_if.data})
    );

    assign cry_commit_if.eop = 1'b1;

    // can accept new request?
    assign cry_req_if.ready = ~stall_in;

    `undef ROR32
    `undef SRL32
    `undef ROL32
endmodule
