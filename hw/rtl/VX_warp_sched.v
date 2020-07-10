`include "VX_define.vh"

module VX_warp_sched (
    input wire                        clk,
    input wire                        reset,
    input wire                        stall,

    // Wspawn
    input wire                        wspawn,
    input wire[31:0]                  wsapwn_pc,
    input wire[`NUM_WARPS-1:0]        wspawn_new_active,

    // CTM
    input  wire                       ctm,
    input  wire[`NUM_THREADS-1:0]     ctm_mask,
    input  wire[`NW_BITS-1:0]         ctm_warp_num,

    // WHALT
    input  wire                       whalt,
    input  wire[`NW_BITS-1:0]         whalt_warp_num,

    input wire                        is_barrier,
`DEBUG_BEGIN
    input wire[31:0]                  barrier_id,
`DEBUG_END
    input wire[$clog2(`NUM_WARPS):0]  num_warps,
    input wire[`NW_BITS-1:0]          barrier_warp_num,

    // WSTALL
    input  wire                       wstall,
    input  wire [`NW_BITS-1:0]        wstall_warp_num,

    // Split
    input wire                        is_split,
    input wire                        dont_split,
    input wire [`NUM_THREADS-1:0]     split_new_mask,
    input wire [`NUM_THREADS-1:0]     split_later_mask,
    input wire [31:0]                 split_save_pc,    
    input wire [`NW_BITS-1:0]         split_warp_num,

    // Join
    input wire                        is_join,
    input wire [`NW_BITS-1:0]         join_warp_num,

    // JAL
    input wire                        jal,
    input wire [31:0]                 dest,
    input wire [`NW_BITS-1:0]         jal_warp_num,

    // Branch
    input wire                        branch_valid,
    input wire                        branch_dir,
    input wire [31:0]                 branch_dest,
    input wire [`NW_BITS-1:0]         branch_warp_num,

    output wire [`NUM_THREADS-1:0]    thread_mask,
    output wire [`NW_BITS-1:0]        warp_num,
    output wire [31:0]                warp_pc,
    output wire                       busy,
    output wire                       scheduled_warp,

    input  wire [`NW_BITS-1:0]        icache_stage_wid,
    input  wire                       icache_stage_response
);
    wire update_use_wspawn;
    wire update_visible_active;

    wire[(1+32+`NUM_THREADS-1):0] d[`NUM_WARPS-1:0];

    wire           join_fall;
    wire[31:0]     join_pc;
    wire[`NUM_THREADS-1:0] join_tm;

`DEBUG_BEGIN
    wire in_wspawn = wspawn;
    wire in_ctm    = ctm;
    wire in_whalt  = whalt;
    wire in_wstall = wstall;
`DEBUG_END

    reg[`NUM_WARPS-1:0]   warp_active;
    reg[`NUM_WARPS-1:0]   warp_stalled;

    reg [`NUM_WARPS-1:0]  visible_active;
    wire[`NUM_WARPS-1:0]  use_active;

    reg [`NUM_WARPS-1:0]  warp_lock;

    wire wstall_this_cycle;

    reg [`NUM_THREADS-1:0] thread_masks[`NUM_WARPS-1:0];
    reg [31:0] warp_pcs[`NUM_WARPS-1:0];

    // barriers
    reg [`NUM_WARPS-1:0] barrier_stall_mask[(`NUM_BARRIERS-1):0];
    wire reached_barrier_limit;
    wire [`NUM_WARPS-1:0] b_mask;
    wire [$clog2(`NUM_WARPS):0] b_count;

    // wsapwn
    reg [31:0]           use_wsapwn_pc;
    reg [`NUM_WARPS-1:0] use_wsapwn;

    wire [`NW_BITS-1:0] warp_to_schedule;
    wire                schedule;

    wire hazard;
    wire global_stall;

    wire real_schedule;

    wire [31:0] new_pc;

    reg [`NUM_WARPS-1:0] total_barrier_stall;

    reg didnt_split;

    integer w, b;
    
    always @(posedge clk) begin
        if (reset) begin
            for (b = 0; b < `NUM_BARRIERS; b=b+1) begin
                barrier_stall_mask[b] <= 0;
            end
            use_wsapwn_pc         <= 0;
            use_wsapwn            <= 0;
            warp_pcs[0]           <= `STARTUP_ADDR;
            warp_active[0]        <= 1; // Activating first warp
            visible_active[0]     <= 1; // Activating first warp
            thread_masks[0]       <= 1; // Activating first thread in first warp
            warp_stalled          <= 0;
            didnt_split           <= 0;
            warp_lock             <= 0;      
            // total_barrier_stall    = 0;
            for (w = 1; w < `NUM_WARPS; w=w+1) begin
                warp_pcs[w]        <= 0;
                warp_active[w]     <= 0; // Activating first warp
                visible_active[w]  <= 0; // Activating first warp
                thread_masks[w]    <= 1; // Activating first thread in first warp
            end

        end else begin
            // Wsapwning warps
            if (wspawn) begin
                warp_active    <= wspawn_new_active;
                use_wsapwn_pc  <= wsapwn_pc;
                use_wsapwn     <= wspawn_new_active & (~`NUM_WARPS'b1);
            end

            if (is_barrier) begin
                warp_stalled[barrier_warp_num]     <= 0;
                if (reached_barrier_limit) begin
                    barrier_stall_mask[barrier_id] <= 0;
                end else begin
                    barrier_stall_mask[barrier_id][barrier_warp_num] <= 1;
                end
            end else if (ctm) begin
                thread_masks[ctm_warp_num] <= ctm_mask;
                warp_stalled[ctm_warp_num] <= 0;
            end else if (is_join && !didnt_split) begin
                if (!join_fall) begin
                    warp_pcs[join_warp_num] <= join_pc;
                end
                thread_masks[join_warp_num] <= join_tm;
                didnt_split                    <= 0;
            end else if (is_split) begin
                warp_stalled[split_warp_num]   <= 0;
                if (!dont_split) begin
                    thread_masks[split_warp_num] <= split_new_mask;
                    didnt_split                <= 0;
                end else begin
                    didnt_split                <= 1;
                end
            end
            
            if (whalt) begin
                warp_active[whalt_warp_num]    <= 0;
                visible_active[whalt_warp_num] <= 0;
            end

            if (update_use_wspawn) begin
                use_wsapwn[warp_to_schedule]   <= 0;
                thread_masks[warp_to_schedule] <= 1;
            end


            // Stalling the scheduling of warps
            if (wstall) begin
                warp_stalled[wstall_warp_num]   <= 1;
                visible_active[wstall_warp_num] <= 0;
            end

            // Refilling active warps
            if (update_visible_active) begin
                visible_active <= warp_active & (~warp_stalled) & (~total_barrier_stall) & ~warp_lock;
            end

            // Don't change state if stall
            if (!global_stall && real_schedule && (thread_mask != 0)) begin
                visible_active[warp_to_schedule] <= 0;
                warp_pcs[warp_to_schedule]       <= new_pc;
            end

            // Jal
            if (jal) begin
                warp_pcs[jal_warp_num]     <= dest;
                warp_stalled[jal_warp_num] <= 0;
            end

            // Branch
            if (branch_valid) begin
                if (branch_dir) begin
                    warp_pcs[branch_warp_num] <= branch_dest;
                end
                warp_stalled[branch_warp_num] <= 0;
            end

            // Lock/Release
            if (scheduled_warp && !stall) begin
                warp_lock[warp_num] <= 1'b1;
            end
            if (icache_stage_response) begin
                warp_lock[icache_stage_wid] <= 1'b0;
            end

        end
    end

    VX_countones #(
        .N(`NUM_WARPS)
    ) barrier_count (
        .valids(b_mask),
        .count (b_count)
    );

    wire [$clog2(`NUM_WARPS):0] count_visible_active;

    VX_countones #(
        .N(`NUM_WARPS)
    ) num_visible (
        .valids(visible_active),
        .count (count_visible_active)
    );

    // assign b_count = $countones(b_mask);

    assign b_mask = barrier_stall_mask[barrier_id][`NUM_WARPS-1:0];
    assign reached_barrier_limit = b_count == (num_warps);

    assign wstall_this_cycle = wstall && (wstall_warp_num == warp_to_schedule); // Maybe bug

    assign total_barrier_stall = barrier_stall_mask[0] | barrier_stall_mask[1] | barrier_stall_mask[2] | barrier_stall_mask[3];

    assign update_visible_active = (0 == count_visible_active) && !(stall || wstall_this_cycle || hazard || is_join);

    wire [(1+32+`NUM_THREADS-1):0] q1 = {1'b1, 32'b0, thread_masks[split_warp_num]};
    wire [(1+32+`NUM_THREADS-1):0] q2 = {1'b0, split_save_pc, split_later_mask};

    assign {join_fall, join_pc, join_tm} = d[join_warp_num];

    genvar i;
    generate
    for (i = 0; i < `NUM_WARPS; i++) begin : stacks
        wire correct_warp_s = (i == split_warp_num);
        wire correct_warp_j = (i == join_warp_num);

        wire push = (is_split && !dont_split) && correct_warp_s;
        wire pop  = is_join  && correct_warp_j;

        VX_generic_stack #(
            .WIDTH(1+32+`NUM_THREADS), 
            .DEPTH($clog2(`NUM_THREADS)+1)
        ) ipdom_stack(
            .clk  (clk),
            .reset(reset),
            .push (push),
            .pop  (pop),
            .d    (d[i]),
            .q1   (q1),
            .q2   (q2)
        );
    end
    endgenerate

    wire should_jal = (jal && (warp_to_schedule == jal_warp_num));
    wire should_bra = (branch_valid && branch_dir && (warp_to_schedule == branch_warp_num));

    assign hazard = (should_jal || should_bra) && schedule;

    assign real_schedule = schedule && !warp_stalled[warp_to_schedule] && !total_barrier_stall[warp_to_schedule] && !warp_lock[0];

    assign global_stall = (stall || wstall_this_cycle || hazard || !real_schedule || is_join);

    assign scheduled_warp = !(wstall_this_cycle || hazard || !real_schedule || is_join) && !reset;

    wire real_use_wspawn = use_wsapwn[warp_to_schedule];

    assign warp_pc     = real_use_wspawn ? use_wsapwn_pc : warp_pcs[warp_to_schedule];
    assign thread_mask = (global_stall) ? 0 : (real_use_wspawn ? `NUM_THREADS'b1 : thread_masks[warp_to_schedule]);
    assign warp_num    = warp_to_schedule;

    assign update_use_wspawn = use_wsapwn[warp_to_schedule] && !global_stall;

    assign new_pc = warp_pc + 4;

    assign use_active = (count_visible_active != 0) ? visible_active : (warp_active & (~warp_stalled) & (~total_barrier_stall) & (~warp_lock));

    // Choosing a warp to schedule
    VX_fixed_arbiter #(
        .N(`NUM_WARPS)
    ) choose_schedule (
        .clk         (clk),
        .reset       (reset),
        .requests    (use_active),
        .grant_index (warp_to_schedule),
        .grant_valid (schedule),
        `UNUSED_PIN  (grant_onehot)
    );

    // always @(*) begin
    //     $display("WarpPC: %h",warp_pc);
    //     $display("real_schedule: %d, schedule: %d, warp_stalled: %d, warp_to_schedule: %d, total_barrier_stall: %d",real_schedule, schedule, warp_stalled[warp_to_schedule], warp_to_schedule,  total_barrier_stall[warp_to_schedule]);
    // end

    assign busy = (warp_active != 0);

endmodule