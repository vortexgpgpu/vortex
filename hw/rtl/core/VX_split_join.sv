`include "VX_platform.vh"

module VX_split_join #(
    parameter CORE_ID = 0
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [`UP(`NW_BITS)-1:0] wid,
    input  gpu_split_t              split,
    input  gpu_join_t               sjoin,
    output wire                     split_is_dvg,
    output wire [`NUM_THREADS-1:0]  split_tmask,
    output wire [`PD_STACK_SIZEW-1:0] split_ret,
    output wire                     join_is_dvg,
    output wire                     join_is_else,
    output wire [`NUM_THREADS-1:0]  join_tmask,
    output wire [`XLEN-1:0]         join_pc
);
    `UNUSED_PARAM (CORE_ID)
    
    wire [(`XLEN+`NUM_THREADS)-1:0] ipdom_data [`NUM_WARPS-1:0];     
    wire [`PD_STACK_SIZEW-1:0] ipdom_q_ptr [`NUM_WARPS-1:0];
    wire ipdom_index [`NUM_WARPS-1:0];

    wire [`NUM_THREADS-1:0] then_tmask;
    wire [`NUM_THREADS-1:0] else_tmask;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign then_tmask[i] = split.tmask[i] && split.taken[i];
        assign else_tmask[i] = split.tmask[i] && ~split.taken[i];
    end

    wire [`CLOG2(`NUM_THREADS+1)-1:0] then_tmask_cnt, else_tmask_cnt;
    `POP_COUNT(then_tmask_cnt, then_tmask);
    `POP_COUNT(else_tmask_cnt, else_tmask);
    wire then_first = (then_tmask_cnt >= else_tmask_cnt);
    
    assign split_is_dvg = (then_tmask != 0) && (else_tmask != 0);
    assign split_tmask = then_first ? then_tmask : else_tmask;
    assign split_ret = ipdom_q_ptr[wid];

    assign join_is_dvg = (sjoin.stack_ptr != ipdom_q_ptr[wid]);
    assign {join_pc, join_tmask} = ipdom_data[wid];
    assign join_is_else = (ipdom_index[wid] == 0);

    wire [`NUM_THREADS-1:0] split_tmask_n = then_first ? else_tmask : then_tmask;
    wire [(`XLEN+`NUM_THREADS)-1:0] ipdom_q0 = {split.next_pc, split_tmask_n};
    wire [(`XLEN+`NUM_THREADS)-1:0] ipdom_q1 = {`XLEN'(0),                 split.tmask};

    wire ipdom_push = split.valid && split_is_dvg;
    wire ipdom_pop = sjoin.valid && join_is_dvg;

    `RESET_RELAY (ipdom_reset, reset);
    
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_ipdom_stack #(
            .WIDTH (`XLEN+`NUM_THREADS), 
            .DEPTH (`PD_STACK_SIZE)
        ) ipdom_stack (
            .clk   (clk),
            .reset (ipdom_reset),
            .push  (ipdom_push && (i == wid)),
            .pop   (ipdom_pop && (i == wid)),
            .q0    (ipdom_q0),
            .q1    (ipdom_q1),
            .d     (ipdom_data[i]),
            .d_idx (ipdom_index[i]),
            .q_ptr (ipdom_q_ptr[i]),
            `UNUSED_PIN (d_ptr),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full)
        );
    end

endmodule
