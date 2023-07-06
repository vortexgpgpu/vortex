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

    localparam NW_WIDTH   = `UP(`NW_BITS);    
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam DATAW = UUID_WIDTH + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + (`NR_BITS * 4) + `XLEN + 1 + 1;

    wire [`NUM_WARPS-1:0][DATAW-1:0] q_data_out;
    wire [DATAW-1:0] q_data_in;
    wire [`NUM_WARPS-1:0] q_full, q_empty, q_alm_empty;

    wire enq_fire = decode_if.valid && decode_if.ready;
    wire deq_fire = ibuffer_if.valid && ibuffer_if.ready;
        
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

        wire q_push = enq_fire && (i == decode_if.wid);
        wire q_pop  = deq_fire && (i == ibuffer_if.wid);

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
            .alm_empty (q_alm_empty[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign decode_if.ibuf_pop[i] = q_pop;
    end

    assign decode_if.ready = ~q_full[decode_if.wid];

    // scoreboad access

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign scoreboard_if.valid[i] = ~q_empty[i];
        assign scoreboard_if.uuid[i]  = q_data_out[i][DATAW-1 -: UUID_WIDTH];
        assign scoreboard_if.tmask[i] = q_data_out[i][DATAW-UUID_WIDTH-1 -: `NUM_THREADS];
        assign scoreboard_if.PC[i]    = q_data_out[i][DATAW-UUID_WIDTH-`NUM_THREADS-1 -: `XLEN];
        assign scoreboard_if.rd[i]    = q_data_out[i][3*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs1[i]   = q_data_out[i][2*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs2[i]   = q_data_out[i][1*`NR_BITS +: `NR_BITS];
        assign scoreboard_if.rs3[i]   = q_data_out[i][0*`NR_BITS +: `NR_BITS];
    end

    // round-robin select

    reg [`NUM_WARPS-1:0] valid_table_n, scb_ready;
    logic deq_valid, deq_valid_n;
    logic [NW_WIDTH-1:0] deq_wid, deq_wid_n;
    reg [DATAW-1:0] deq_instr;
    
    always @(*) begin
        valid_table_n = scoreboard_if.valid;        
        if (deq_fire) begin
            valid_table_n[ibuffer_if.wid] = ~q_alm_empty[ibuffer_if.wid];
        end
    end
    
    VX_rr_arbiter #(
        .NUM_REQS (`NUM_WARPS)
    ) rr_arbiter (
        .clk      (clk),
        .reset    (reset),          
        .requests (valid_table_n), 
        .grant_index (deq_wid_n),
        .grant_valid (deq_valid_n),
        `UNUSED_PIN (grant_onehot),
        `UNUSED_PIN (unlock)
    );    

    always @(posedge clk) begin
        if (reset)  begin
            deq_valid <= 0;
        end else begin
            deq_valid <= deq_valid_n;
        end
        deq_wid   <= deq_wid_n;
        deq_instr <= q_data_out[deq_wid_n];
        scb_ready <= scoreboard_if.ready;
    end

    assign ibuffer_if.valid = deq_valid && scb_ready[deq_wid];
    assign ibuffer_if.wid   = deq_wid;
    assign {ibuffer_if.uuid,
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
            ibuffer_if.rs3} = deq_instr;

endmodule
