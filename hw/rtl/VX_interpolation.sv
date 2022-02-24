// TODO

`include "VX_define.vh"

module VX_interpolation (
    input wire clk,
    input wire reset,
    
    // Inputs    
    input wire [`INST_MUL_BITS-1:0]     alu_op,
    input wire [`UUID_BITS-1:0]         uuid_in,
    input wire [`NW_BITS-1:0]           wid_in,
    input wire [`NUM_THREADS-1:0]       tmask_in,
    input wire [31:0]                   PC_in,
    input wire [`NR_BITS-1:0]           rd_in,
    input wire                          wb_in,
    input wire [`NUM_THREADS-1:0][31:0] alu_in1, 
    input wire [`NUM_THREADS-1:0][31:0] alu_in2,
    input wire [`NUM_THREADS-1:0][31:0] alu_in3,

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

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire [`NUM_THREADS-1:0][31:0] add_result;
    wire [`UUID_BITS-1:0] mul_uuid_out;
    wire [`NW_BITS-1:0] mul_wid_out;
    wire [`NUM_THREADS-1:0] mul_tmask_out;
    wire [31:0] mul_PC_out;
    wire [`NR_BITS-1:0] mul_rd_out;
    wire mul_wb_out;

    wire stall_out;

    wire mul_valid_out;
    wire mul_valid_in = valid_in;    
    wire mul_ready_in = ~stall_out || ~mul_valid_out;

    wire is_mulh_in      = (alu_op != `INST_MUL_MUL);

    ///////////////////////////////////////////////////////////////////////////
    
    wire is_mulh_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [31:0] mul_in1 = alu_in1[i];
        wire [31:0] mul_in2 = alu_in2[i];
    `IGNORE_UNUSED_BEGIN
        wire [63:0] mul_result_tmp;
    `IGNORE_UNUSED_END

        VX_multiplier #(
            .WIDTHA  (32),
            .WIDTHB  (32),
            .WIDTHP  (64),
            .SIGNED  (0),
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
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1 + (`NUM_THREADS * 32)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,  uuid_in,      wid_in,      tmask_in,       PC_in,      rd_in,      wb_in,      is_mulh_in,  mul_result}),
        .data_out ({mul_valid_out, mul_uuid_out, mul_wid_out, mul_tmask_out,  mul_PC_out, mul_rd_out, mul_wb_out, is_mulh_out, mul_result_out})
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign add_result[i] = mul_result_out[i] + alu_in3[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    wire                    rsp_valid = mul_valid_out;  
    wire [`UUID_BITS-1:0]   rsp_uuid  = mul_uuid_out;
    wire [`NW_BITS-1:0]     rsp_wid   = mul_wid_out;
    wire [`NUM_THREADS-1:0] rsp_tmask = mul_tmask_out;
    wire [31:0]             rsp_PC    = mul_PC_out;
    wire [`NR_BITS-1:0]     rsp_rd    = mul_rd_out;
    wire                    rsp_wb    = mul_wb_out;
    wire [`NUM_THREADS-1:0][31:0] rsp_data = mul_result;

    assign stall_out = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({rsp_valid, rsp_uuid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_wb, rsp_data}),
        .data_out ({valid_out, uuid_out, wid_out, tmask_out, PC_out, rd_out, wb_out, data_out})
    );

    // can accept new request?
    assign ready_in = mul_ready_in;
    
endmodule