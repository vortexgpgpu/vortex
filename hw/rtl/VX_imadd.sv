`include "VX_define.vh"

module VX_imadd (
    input wire clk,
    input wire reset,
    
    // Inputs
    input wire [`INST_MOD_BITS-1:0]     op_mod,
    input wire [`UUID_BITS-1:0]         uuid_in,
    input wire [`NW_BITS-1:0]           wid_in,
    input wire [`NUM_THREADS-1:0]       tmask_in,
    input wire [31:0]                   PC_in,
    input wire [`NR_BITS-1:0]           rd_in,
    input wire                          wb_in,
    input wire [`NUM_THREADS-1:0][31:0] data_in1, 
    input wire [`NUM_THREADS-1:0][31:0] data_in2,
    input wire [`NUM_THREADS-1:0][31:0] data_in3,

    // Outputs
    output wire [`UUID_BITS-1:0]         uuid_out,
    output wire [`NW_BITS-1:0]           wid_out,
    output wire [`NUM_THREADS-1:0]       tmask_out,
    output wire [31:0]                   PC_out,
    output wire [`NR_BITS-1:0]           rd_out,
    output wire                          wb_out,
    output wire [`NUM_THREADS-1:0][31:0] data_out,

    // handshake
    input wire  valid_in,
    output wire ready_in,
    output wire valid_out,
    input wire  ready_out
); 

    wire                    valid_s;
    wire [`UUID_BITS-1:0]   uuid_s;
    wire [`NW_BITS-1:0]     wid_s;
    wire [`NUM_THREADS-1:0] tmask_s;
    wire [31:0]             PC_s;
    wire [`NR_BITS-1:0]     rd_s;
    wire                    wb_s;
    wire [1:0] shift, shift_s;
    wire [`NUM_THREADS-1:0][31:0] data_in3_s;

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire [`NUM_THREADS-1:0][31:0] add_result;

    assign shift = op_mod[1:0];
    `UNUSED_VAR (op_mod)

    wire stall_out;    

    wire mul_ready_in = ~stall_out || ~valid_s;

    ///////////////////////////////////////////////////////////////////////////

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [31:0] mul_in1 = data_in1[i];
        wire [31:0] mul_in2 = data_in2[i];
        wire [55:0] mul_result_tmp;

        VX_multiplier #(
            .WIDTHA  (32),
            .WIDTHB  (32),
            .WIDTHP  (56), // 56-24=32
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL+1)
        ) multiplier (
            .clk    (clk),
            .enable (mul_ready_in),
            .dataa  (mul_in1),
            .datab  (mul_in2),
            .result (mul_result_tmp)
        );

        assign mul_result[i] = 32'($signed(mul_result_tmp) >> $signed({shift_s, 3'b0}));
    end

    VX_shift_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 2 + (`NUM_THREADS * 32)),
        .DEPTH  (`LATENCY_IMUL+1),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({valid_in, uuid_in, wid_in, tmask_in, PC_in, rd_in, wb_in, shift,   data_in3}),
        .data_out ({valid_s,  uuid_s,  wid_s,  tmask_s,  PC_s,  rd_s,  wb_s,  shift_s, data_in3_s})
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign add_result[i] = mul_result[i] + data_in3_s[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    assign stall_out = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s,   uuid_s,   wid_s,   tmask_s,   PC_s,   rd_s,   wb_s,   add_result}),
        .data_out ({valid_out, uuid_out, wid_out, tmask_out, PC_out, rd_out, wb_out, data_out})
    );

    // can accept new request?
    assign ready_in = mul_ready_in;
    
endmodule
