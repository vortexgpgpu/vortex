`include "VX_define.vh"

module VX_warp_sched #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_warp_ctl_if      warp_ctl_if,
    VX_wstall_if        wstall_if,
    VX_join_if          join_if,
    VX_branch_ctl_if    branch_ctl_if,

    VX_ifetch_rsp_if    ifetch_rsp_if,
    VX_ifetch_req_if    ifetch_req_if,

    output wire         busy
);
    wire update_use_wspawn;
    wire update_visible_active;

    wire [(1+32+`NUM_THREADS-1):0] ipdom[`NUM_WARPS-1:0];

    wire                    join_fall;
    wire [31:0]             join_pc;
    wire [`NUM_THREADS-1:0] join_tm;

    reg [`NUM_WARPS-1:0] warp_active;
    reg [`NUM_WARPS-1:0] warp_stalled;

    reg [`NUM_WARPS-1:0]  visible_active;
    wire [`NUM_WARPS-1:0] use_active;

    reg [`NUM_WARPS-1:0] warp_lock;

    wire wstall_this_cycle;

    reg [`NUM_THREADS-1:0] thread_masks[`NUM_WARPS-1:0];
    reg [31:0] warp_pcs[`NUM_WARPS-1:0];

    // barriers
    reg [`NUM_WARPS-1:0] barrier_stall_mask[`NUM_BARRIERS-1:0];
    wire [`NUM_WARPS-1:0] b_mask;
    wire [`NW_BITS:0] b_count;

    wire reached_barrier_limit;

    // wspawn
    reg [31:0]              use_wspawn_pc;
    reg [`NUM_WARPS-1:0]    use_wspawn;

    wire [`NW_BITS-1:0]     warp_to_schedule;
    wire                    schedule;

    wire [`NUM_THREADS-1:0] thread_mask;
    wire [`NW_BITS-1:0]     warp_num;
    wire [31:0]             warp_pc;
    wire                    scheduled_warp;

    wire hazard;
    wire global_stall;

    wire real_schedule;

    wire [31:0] new_pc;

    reg [`NUM_WARPS-1:0] total_barrier_stall;

    reg didnt_split;

    wire stall;

    always @(posedge clk) begin
        integer i;
        if (reset) begin
            for (i = 0; i < `NUM_BARRIERS; i++) begin
                barrier_stall_mask[i] <= 0;
            end

            use_wspawn_pc         <= 0;
            use_wspawn            <= 0;
            warp_pcs[0]           <= `STARTUP_ADDR;
            warp_active[0]        <= 1; // Activating first warp
            visible_active[0]     <= 1; // Activating first warp
            thread_masks[0]       <= 1; // Activating first thread in first warp
            warp_stalled          <= 0;
            didnt_split           <= 0;
            warp_lock             <= 0;      
            
            for (i = 1; i < `NUM_WARPS; i++) begin
                warp_pcs[i]       <= 0;
                warp_active[i]    <= 0; // Activating first warp
                visible_active[i] <= 0; // Activating first warp
                thread_masks[i]   <= 1; // Activating first thread in first warp
            end

        end else begin
            
            if (warp_ctl_if.wspawn) begin
                warp_active   <= warp_ctl_if.wspawn_wmask;                
                use_wspawn    <= warp_ctl_if.wspawn_wmask & (~`NUM_WARPS'(1));
                use_wspawn_pc <= warp_ctl_if.wspawn_pc;
            end

            if (warp_ctl_if.is_barrier) begin
                warp_stalled[warp_ctl_if.warp_num]     <= 0;
                if (reached_barrier_limit) begin
                    barrier_stall_mask[warp_ctl_if.barrier_id] <= 0;
                end else begin
                    barrier_stall_mask[warp_ctl_if.barrier_id][warp_ctl_if.warp_num] <= 1;
                end
            end else if (warp_ctl_if.change_mask) begin
                thread_masks[warp_ctl_if.warp_num] <= warp_ctl_if.thread_mask;
                warp_stalled[warp_ctl_if.warp_num] <= 0;
                if (0 == warp_ctl_if.thread_mask) begin
                    warp_active[warp_ctl_if.warp_num]    <= 0;
                    visible_active[warp_ctl_if.warp_num] <= 0;
                end
            end else if (join_if.is_join && !didnt_split) begin
                if (!join_fall) begin
                    warp_pcs[join_if.warp_num] <= join_pc;
                end
                thread_masks[join_if.warp_num] <= join_tm;
                didnt_split                 <= 0;
            end else if (warp_ctl_if.is_split) begin
                warp_stalled[warp_ctl_if.warp_num]   <= 0;
                if (warp_ctl_if.do_split) begin
                    thread_masks[warp_ctl_if.warp_num] <= warp_ctl_if.split_new_mask;
                    didnt_split                <= 0;
                end else begin
                    didnt_split                <= 1;
                end
            end          

            if (update_use_wspawn) begin
                use_wspawn[warp_to_schedule]   <= 0;
                thread_masks[warp_to_schedule] <= 1;
            end

            // Stalling the scheduling of warps
            if (wstall_if.wstall) begin
                warp_stalled[wstall_if.warp_num]   <= 1;
                visible_active[wstall_if.warp_num] <= 0;
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

            // Branch
            if (branch_ctl_if.valid) begin
                if (branch_ctl_if.taken) begin
                    warp_pcs[branch_ctl_if.warp_num] <= branch_ctl_if.dest;
                end
                warp_stalled[branch_ctl_if.warp_num] <= 0;
            end

            // Lock/Release
            if (scheduled_warp && !stall) begin
                warp_lock[warp_num] <= 1;
            end
            if (ifetch_rsp_if.valid && ifetch_rsp_if.ready) begin
                warp_lock[ifetch_rsp_if.warp_num] <= 0;
            end

        end
    end

    VX_countones #(
        .N(`NUM_WARPS)
    ) barrier_count (
        .valids(b_mask),
        .count (b_count)
    );

    wire [`NW_BITS:0] count_visible_active;

    VX_countones #(
        .N(`NUM_WARPS)
    ) num_visible (
        .valids(visible_active),
        .count (count_visible_active)
    );

    assign b_mask = barrier_stall_mask[warp_ctl_if.barrier_id][`NUM_WARPS-1:0];
    
    assign reached_barrier_limit = (b_count == warp_ctl_if.barrier_num_warps);

    assign wstall_this_cycle = wstall_if.wstall && (wstall_if.warp_num == warp_to_schedule); // Maybe bug

    assign total_barrier_stall = barrier_stall_mask[0] | barrier_stall_mask[1] | barrier_stall_mask[2] | barrier_stall_mask[3];

    assign update_visible_active = (0 == count_visible_active) && !(stall || wstall_this_cycle || hazard || join_if.is_join);

    wire [(1+32+`NUM_THREADS-1):0] q1 = {1'b1, 32'b0, thread_masks[warp_ctl_if.warp_num]};
    wire [(1+32+`NUM_THREADS-1):0] q2 = {1'b0, warp_ctl_if.split_save_pc, warp_ctl_if.split_later_mask};

    assign {join_fall, join_pc, join_tm} = ipdom[join_if.warp_num];

    genvar i;
    for (i = 0; i < `NUM_WARPS; i++) begin
        wire correct_warp_s = (i == warp_ctl_if.warp_num);
        wire correct_warp_j = (i == join_if.warp_num);

        wire push = (warp_ctl_if.is_split && warp_ctl_if.do_split) && correct_warp_s;
        wire pop  = join_if.is_join && correct_warp_j;

        VX_ipdom_stack #(
            .WIDTH(1+32+`NUM_THREADS), 
            .DEPTH(`NT_BITS+1)
        ) ipdom_stack (
            .clk  (clk),
            .reset(reset),
            .push (push),
            .pop  (pop),
            .d    (ipdom[i]),
            .q1   (q1),
            .q2   (q2),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full)
        );
    end

    wire should_bra = (branch_ctl_if.valid && branch_ctl_if.taken && (warp_to_schedule == branch_ctl_if.warp_num));

    assign hazard = should_bra && schedule;

    assign real_schedule = schedule && !warp_stalled[warp_to_schedule] && !total_barrier_stall[warp_to_schedule] && !warp_lock[0];

    assign global_stall = (stall || wstall_this_cycle || hazard || !real_schedule || join_if.is_join);

    assign scheduled_warp = !(wstall_this_cycle || hazard || !real_schedule || join_if.is_join) && !reset;

    wire real_use_wspawn = use_wspawn[warp_to_schedule];

    assign warp_pc     = real_use_wspawn ? use_wspawn_pc : warp_pcs[warp_to_schedule];
    
    assign thread_mask = global_stall ? 0 : (real_use_wspawn ? `NUM_THREADS'(1) : thread_masks[warp_to_schedule]);

    assign warp_num    = warp_to_schedule;

    assign update_use_wspawn = use_wspawn[warp_to_schedule] && !global_stall;

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

    assign stall = ~ifetch_req_if.ready && ifetch_req_if.valid;    

    VX_generic_register #( 
        .N(1 + `NUM_THREADS + 32 + `NW_BITS)
    ) fetch_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({(| thread_mask),     thread_mask,               warp_pc,               warp_num}),
        .out   ({ifetch_req_if.valid, ifetch_req_if.thread_mask, ifetch_req_if.curr_PC, ifetch_req_if.warp_num})
    );

    assign busy = (warp_active != 0); 

endmodule