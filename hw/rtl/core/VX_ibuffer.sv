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

    localparam SIZE      = `IBUF_SIZE + 1;
    localparam ALM_FULL  = SIZE - 1;
    localparam ALM_EMPTY = 1;

    localparam NW_WIDTH   = `UP(`NW_BITS);
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam ADDRW      = $clog2(SIZE);
    localparam NWARPSW    = $clog2(`NUM_WARPS+1);
    localparam DATAW = UUID_WIDTH + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + (`NR_BITS * 4) + `XLEN + 1 + 1;
    
    wire [`NUM_WARPS-1:0] q_full, q_empty, q_alm_full, q_alm_empty;
    wire [`NUM_WARPS-1:0][DATAW-1:0] q_data_prev;
    reg [`NUM_WARPS-1:0][DATAW-1:0] q_data_out;
    wire [DATAW-1:0] q_data_in;

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

        reg [ADDRW-1:0] used_r;
        reg full_r, empty_r, alm_full_r, alm_empty_r;

        wire push = enq_fire && (i == decode_if.wid); 
        wire pop  = deq_fire && (i == ibuffer_if.wid);

        wire going_empty = empty_r || (alm_empty_r && pop);

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (SIZE-1),
            .OUT_REG (1)
        ) queue (
            .clk      (clk),
            .reset    (reset),
            .valid_in (push && ~going_empty),
            .data_in  (q_data_in),
            .ready_out(pop),
            .data_out (q_data_prev[i]),           
            `UNUSED_PIN (ready_in),
            `UNUSED_PIN (valid_out)
        );

        always @(posedge clk) begin
            if (reset) begin            
                used_r      <= '0;
                full_r      <= 0; 
                alm_full_r  <= 0;
                empty_r     <= 1; 
                alm_empty_r <= 1;
            end else begin  
                if (push) begin
                    if (!pop) begin
                        empty_r <= 0;
                        if (used_r == ADDRW'(ALM_EMPTY))
                            alm_empty_r <= 0;
                        if (used_r == ADDRW'(SIZE-1))
                            full_r <= 1;
                        if (used_r == ADDRW'(ALM_FULL-1))
                            alm_full_r <= 1;
                    end
                end else if (pop) begin
                    full_r <= 0; 
                    if (used_r == ADDRW'(ALM_FULL))
                        alm_full_r <= 0;
                    if (used_r == ADDRW'(1))
                        empty_r <= 1;
                    if (used_r == ADDRW'(ALM_EMPTY+1))
                        alm_empty_r <= 1;
                end
                used_r <= $signed(used_r) + ADDRW'($signed(2'(push) - 2'(pop)));                
            end 

            if (push && going_empty) begin                                                       
                q_data_out[i] <= q_data_in;
            end else if (pop) begin
                q_data_out[i] <= q_data_prev[i];
            end                  
        end
        
        assign q_full[i]      = full_r;
        assign q_empty[i]     = empty_r;
        assign q_alm_full[i]  = alm_full_r;
        assign q_alm_empty[i] = alm_empty_r;

        assign decode_if.ibuf_pop[i] = pop;
    end
    
    `UNUSED_VAR (q_alm_full)

    assign decode_if.ready = ~q_full[decode_if.wid];

    // scoreboad access

    assign scoreboard_if.valid = deq_valid_n;
    assign scoreboard_if.wid   = deq_wid_n;
    assign scoreboard_if.uuid  = deq_instr_n[DATAW-1 -: UUID_WIDTH];    
    assign scoreboard_if.tmask = deq_instr_n[DATAW-UUID_WIDTH-1 -: `NUM_THREADS];
    assign scoreboard_if.PC    = deq_instr_n[DATAW-UUID_WIDTH-`NUM_THREADS-1 -: `XLEN];
    assign scoreboard_if.rd    = deq_instr_n[3*`NR_BITS +: `NR_BITS];
    assign scoreboard_if.rs1   = deq_instr_n[2*`NR_BITS +: `NR_BITS];
    assign scoreboard_if.rs2   = deq_instr_n[1*`NR_BITS +: `NR_BITS];
    assign scoreboard_if.rs3   = deq_instr_n[0*`NR_BITS +: `NR_BITS];

    ///////////////////////////////////////////////////////////////////////////

    reg [`NUM_WARPS-1:0] valid_table, valid_table_n;
    reg [NW_WIDTH-1:0] deq_wid, deq_wid_n;
    reg [NW_WIDTH-1:0] deq_wid_rr;
    wire [NW_WIDTH-1:0] deq_wid_rr_n;
    reg deq_valid, deq_valid_n;
    reg [DATAW-1:0] deq_instr, deq_instr_n;
    reg [NWARPSW-1:0] num_warps;
    reg scb_ready;

    `UNUSED_VAR (deq_instr)

    // calculate valid table
    always @(*) begin
        valid_table_n = valid_table;        
        if (deq_fire) begin
            valid_table_n[deq_wid] = ~q_alm_empty[deq_wid];
        end
        if (enq_fire) begin
            valid_table_n[decode_if.wid] = 1;
        end
    end

    // round-robin warp scheduling
    VX_rr_arbiter #(
        .NUM_REQS (`NUM_WARPS)
    ) rr_arbiter (
        .clk      (clk),
        .reset    (reset),          
        .requests (valid_table_n), 
        .grant_index (deq_wid_rr_n),
        `UNUSED_PIN (grant_valid),
        `UNUSED_PIN (grant_onehot),
        `UNUSED_PIN (unlock)
    );

    // schedule the next instruction to issue
    wire is_single_warp = (1 == num_warps) && !(deq_fire && q_alm_empty[deq_wid]);
    always @(*) begin
        if (num_warps > 1) begin
            deq_valid_n = 1;
            deq_wid_n   = deq_wid_rr;          
            deq_instr_n = q_data_out[deq_wid_rr];
        end else begin
            deq_valid_n = is_single_warp;
            deq_wid_n   = is_single_warp ? deq_wid : decode_if.wid;
            deq_instr_n = deq_fire ? q_data_prev[deq_wid] : q_data_out[deq_wid];
        end
    end

    wire warp_inc = enq_fire && q_empty[decode_if.wid];
    wire warp_dec = deq_fire && q_alm_empty[deq_wid] && ~(enq_fire && decode_if.wid == deq_wid);
    
    always @(posedge clk) begin
        if (reset)  begin
            valid_table <= '0;
            deq_valid   <= 0;
            num_warps   <= '0;
        end else begin
            valid_table <= valid_table_n;
            deq_valid   <= deq_valid_n;
            num_warps   <= num_warps + NWARPSW'($signed(2'(warp_inc) - 2'(warp_dec)));
        end        
        deq_wid    <= deq_wid_n;
        deq_wid_rr <= deq_wid_rr_n;
        deq_instr  <= deq_instr_n;
        scb_ready  <= scoreboard_if.ready;
    end

    assign ibuffer_if.valid = deq_valid && scb_ready;
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
