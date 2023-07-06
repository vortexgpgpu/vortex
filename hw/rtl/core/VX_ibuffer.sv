`include "VX_define.vh"

module VX_ibuffer #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_decode_if.slave  decode_if,  

    // outputs
    VX_scoreboard_if.master scoreboard_if,
    VX_ibuffer_if.master ibuffer_if
);
    `UNUSED_PARAM (CORE_ID)

    localparam NW_WIDTH = `UP(`NW_BITS);    
    localparam DATAW = `UP(`UUID_BITS) + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + (`NR_BITS * 4) + `XLEN + 1 + 1;

    wire [`NUM_WARPS-1:0][DATAW-1:0] q_data_out;
    wire [DATAW-1:0] q_data_in;
    wire [`NUM_WARPS-1:0] q_full, q_empty;
    wire [`NUM_WARPS-1:0] deq_valid_in, deq_ready_in;  
        
    assign q_data_in = {decode_if.uuid,
                        decode_if.tmask, 
                        decode_if.PC, 
                        decode_if.ex_type, 
                        decode_if.op_type, 
                        decode_if.op_mod, 
                        decode_if.wb,
                        decode_if.use_PC,
                        decode_if.use_imm,
                        decode_if.imm,
                        decode_if.rd, 
                        decode_if.rs1, 
                        decode_if.rs2, 
                        decode_if.rs3};

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin

        wire q_push = decode_if.valid && decode_if.ready && (i == decode_if.wid);
        wire q_pop  = deq_valid_in[i] && deq_ready_in[i];

        VX_fifo_queue #(
            .DATAW   (DATAW),
            .DEPTH   (`IBUF_SIZE),
            .OUT_REG (1)
        ) inst_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (q_push),
            .pop      (q_pop),
            .data_in  (q_data_in),
            .data_out (q_data_out[i]),
            .full     (q_full[i]),
            .empty    (q_empty[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );

        assign decode_if.ibuf_pop[i] = q_pop;
    end

    assign decode_if.ready = ~q_full[decode_if.wid];

    // scoreboad access

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign scoreboard_if.valid[i] = ~q_empty[i];
        assign scoreboard_if.rd[i]    = q_data_out[i][3*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs1[i]   = q_data_out[i][2*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs2[i]   = q_data_out[i][1*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs3[i]   = q_data_out[i][0*`NR_BITS +: `NR_BITS];
    end

    // round-robin select

    wire [`NUM_WARPS-1:0][(NW_WIDTH+DATAW)-1:0] deq_data_in;
    
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign deq_valid_in[i] = scoreboard_if.valid[i] && scoreboard_if.ready[i];
        assign deq_data_in[i]  = {NW_WIDTH'(i), q_data_out[i]};
    end   

    VX_stream_arb #(            
        .NUM_INPUTS  (`NUM_WARPS),
        .DATAW       (NW_WIDTH+DATAW),
        .ARBITER     ("R"),
        .LOCK_ENABLE (0),
        .BUFFERED    (2)        
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (deq_valid_in),
        .ready_in  (deq_ready_in),
        .data_in   (deq_data_in),                
        .data_out  ({
            ibuffer_if.wid,
            ibuffer_if.uuid,            
            ibuffer_if.tmask, 
            ibuffer_if.PC, 
            ibuffer_if.ex_type, 
            ibuffer_if.op_type, 
            ibuffer_if.op_mod, 
            ibuffer_if.wb,
            ibuffer_if.use_PC,
            ibuffer_if.use_imm,
            ibuffer_if.imm,
            ibuffer_if.rd, 
            ibuffer_if.rs1, 
            ibuffer_if.rs2, 
            ibuffer_if.rs3}),
        .valid_out (ibuffer_if.valid),
        .ready_out (ibuffer_if.ready)
    );

endmodule
