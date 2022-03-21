`include "VX_define.vh"

module VX_interpolation (
    input wire clk,
    input wire reset,
    
    // Inputs    
    input wire [`INST_MUL_BITS-1:0]     interp_op,
    input wire [`UUID_BITS-1:0]         uuid_in,
    input wire [`NW_BITS-1:0]           wid_in,
    input wire [`NUM_THREADS-1:0]       tmask_in,
    input wire [31:0]                   PC_in,
    input wire [`NR_BITS-1:0]           rd_in,
    input wire                          wb_in,
    input wire [`NUM_THREADS-1:0][31:0] interp_in1, 
    input wire [`NUM_THREADS-1:0][31:0] interp_in2,
    input wire [`NUM_THREADS-1:0][31:0] interp_in3,

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
    wire [`NUM_THREADS-1:0][31:0] mul_result_out;
    wire [`NUM_THREADS-1:0][31:0] add_result;
    wire [`NR_BITS-1:0] mul_rd_out;
    wire mul_wb_out;

    wire stall_out;

    wire imadd_valid_out;
    wire mul_ready_in = ~stall_out || ~imadd_valid_out;

    wire is_mulh_in      = (interp_op != `INST_MUL_MUL);

    ///////////////////////////////////////////////////////////////////////////
    
    wire is_mulh_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [31:0] mul_in1 = interp_in1[i];
        wire [31:0] mul_in2 = interp_in2[i];
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

    reg [`LATENCY_IMUL-1:0] mul_shift_reg;

    always @(posedge clk) // wait for multiplier
        begin
            mul_shift_reg <= { mul_shift_reg[`LATENCY_IMUL-2:0], valid_in & mul_ready_in };
        end
    
    assign imadd_valid_out = mul_shift_reg[`LATENCY_IMUL-1];


    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign add_result[i] = mul_result[i] + interp_in3[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] rsp_data = add_result;

    assign stall_out = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({imadd_valid_out, uuid_in, wid_in, tmask_in, PC_in, rd_in, wb_in, rsp_data}),
        .data_out ({valid_out, uuid_out, wid_out, tmask_out, PC_out, rd_out, wb_out, data_out})
    );

    // can accept new request?
    assign ready_in = mul_ready_in;
    
endmodule
