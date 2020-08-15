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
    wire                    join_fall;
    wire [31:0]             join_pc;
    wire [`NUM_THREADS-1:0] join_tm;

    reg [`NUM_WARPS-1:0] warp_active;
    reg [`NUM_WARPS-1:0] warp_stalled;
    reg [`NUM_WARPS-1:0] visible_active;
    wire update_visible_active;
    
    reg [`NUM_WARPS-1:0] warp_lock;

    reg [`NUM_THREADS-1:0] thread_masks[`NUM_WARPS-1:0];
    reg [31:0] warp_pcs[`NUM_WARPS-1:0];

    // barriers
    reg [`NUM_WARPS-1:0] barrier_stall_mask[`NUM_BARRIERS-1:0];    
    wire reached_barrier_limit;
    reg [`NUM_WARPS-1:0] total_barrier_stall;

    // wspawn
    reg [31:0]              use_wspawn_pc;
    reg [`NUM_WARPS-1:0]    use_wspawn;

    
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [31:0]             warp_pc;
    wire [`NW_BITS-1:0]     warp_to_schedule;
    wire                    scheduled_warp;

    wire stall_out;
    wire global_stall;
    wire real_schedule; 

    reg didnt_split;        

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `NUM_BARRIERS; i++) begin
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
            
            for (integer i = 1; i < `NUM_WARPS; i++) begin
                warp_pcs[i]       <= 0;
                warp_active[i]    <= 0; // Activating first warp
                visible_active[i] <= 0; // Activating first warp
                thread_masks[i]   <= 1; // Activating first thread in first warp
            end
        end else begin            
            if (warp_ctl_if.wspawn.valid) begin
                warp_active   <= warp_ctl_if.wspawn.wmask;                
                use_wspawn    <= warp_ctl_if.wspawn.wmask & (~`NUM_WARPS'(1));
                use_wspawn_pc <= warp_ctl_if.wspawn.pc;
            end

            if (warp_ctl_if.barrier.valid) begin
                warp_stalled[warp_ctl_if.wid] <= 0;
                if (reached_barrier_limit) begin
                    barrier_stall_mask[warp_ctl_if.barrier.id] <= 0;
                end else begin
                    barrier_stall_mask[warp_ctl_if.barrier.id][warp_ctl_if.wid] <= 1;
                end
            end else if (warp_ctl_if.tmc.valid) begin
                thread_masks[warp_ctl_if.wid] <= warp_ctl_if.tmc.thread_mask;
                warp_stalled[warp_ctl_if.wid] <= 0;
                if (0 == warp_ctl_if.tmc.thread_mask) begin
                    warp_active[warp_ctl_if.wid]    <= 0;
                    visible_active[warp_ctl_if.wid] <= 0;
                end
            end else if (join_if.is_join && !didnt_split) begin
                if (!join_fall) begin
                    warp_pcs[join_if.wid] <= join_pc;
                end
                thread_masks[join_if.wid] <= join_tm;
                didnt_split <= 0;
            end else if (warp_ctl_if.split.valid) begin
                warp_stalled[warp_ctl_if.wid] <= 0;
                if (warp_ctl_if.split.diverged) begin
                    thread_masks[warp_ctl_if.wid] <= warp_ctl_if.split.then_mask;
                    didnt_split <= 0;
                end else begin
                    didnt_split <= 1;
                end
            end          

            if (use_wspawn[warp_to_schedule] && !global_stall) begin
                use_wspawn[warp_to_schedule]   <= 0;
                thread_masks[warp_to_schedule] <= 1;
            end

            // Stalling the scheduling of warps
            if (wstall_if.wstall) begin
                warp_stalled[wstall_if.wid]   <= 1;
                visible_active[wstall_if.wid] <= 0;
            end

            // Refilling active warps
            if (update_visible_active) begin
                visible_active <= warp_active & ~warp_stalled & ~total_barrier_stall & ~warp_lock;
            end

            // Don't change state if stall
            if (!global_stall && real_schedule && (thread_mask != 0)) begin
                visible_active[warp_to_schedule] <= 0;
                warp_pcs[warp_to_schedule]       <= warp_pc + 4;
            end

            // Branch
            if (branch_ctl_if.valid) begin
                if (branch_ctl_if.taken) begin
                    warp_pcs[branch_ctl_if.wid] <= branch_ctl_if.dest;
                end
                warp_stalled[branch_ctl_if.wid] <= 0;
            end

            // Lock/Release
            if (scheduled_warp && !stall_out) begin
                warp_lock[warp_to_schedule] <= 1;
            end
            if (ifetch_rsp_if.valid && ifetch_rsp_if.ready) begin
                warp_lock[ifetch_rsp_if.wid] <= 0;
            end
        end
    end

    wire [`NUM_WARPS-1:0] b_mask = barrier_stall_mask[warp_ctl_if.barrier.id][`NUM_WARPS-1:0];
    wire [`NW_BITS:0] b_count;

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
    
    assign reached_barrier_limit = (b_count == warp_ctl_if.barrier.num_warps);

    assign total_barrier_stall = barrier_stall_mask[0] | barrier_stall_mask[1] | barrier_stall_mask[2] | barrier_stall_mask[3];

    wire [(1+32+`NUM_THREADS-1):0] ipdom[`NUM_WARPS-1:0];
    wire [(1+32+`NUM_THREADS-1):0] q1 = {1'b1, 32'b0, thread_masks[warp_ctl_if.wid]};
    wire [(1+32+`NUM_THREADS-1):0] q2 = {1'b0, warp_ctl_if.split.pc, warp_ctl_if.split.else_mask};

    assign {join_fall, join_pc, join_tm} = ipdom[join_if.wid];

    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        wire push = warp_ctl_if.split.valid 
                 && warp_ctl_if.split.diverged 
                 && (i == warp_ctl_if.wid);

        wire pop = join_if.is_join 
                && (i == join_if.wid);

        VX_ipdom_stack #(
            .WIDTH(1+32+`NUM_THREADS), 
            .DEPTH(`NT_BITS+1)
        ) ipdom_stack (
            .clk  (clk),
            .reset(reset),
            .push (push),
            .pop  (pop),
            .q1   (q1),
            .q2   (q2),
            .d    (ipdom[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full)
        );
    end

    wire schedule;

    wire branch_hazard = schedule
                      && branch_ctl_if.valid 
                      && branch_ctl_if.taken 
                      && (branch_ctl_if.wid == warp_to_schedule);

    assign real_schedule = schedule 
                        && !warp_stalled[warp_to_schedule] 
                        && !total_barrier_stall[warp_to_schedule] 
                        && !warp_lock[0];

    wire wstall_this_cycle = wstall_if.wstall && (wstall_if.wid == warp_to_schedule); // Maybe bug

    assign update_visible_active = (0 == count_visible_active) && !(stall_out || wstall_this_cycle || branch_hazard || join_if.is_join);

    assign global_stall = stall_out || wstall_this_cycle || branch_hazard || !real_schedule || join_if.is_join;

    assign scheduled_warp = !(wstall_this_cycle || branch_hazard || !real_schedule || join_if.is_join) && !reset;

    assign warp_pc = use_wspawn[warp_to_schedule] ? use_wspawn_pc : warp_pcs[warp_to_schedule];
    
    assign thread_mask = global_stall ? 0 : (use_wspawn[warp_to_schedule] ? `NUM_THREADS'(1) : thread_masks[warp_to_schedule]);

    wire [`NUM_WARPS-1:0] use_active = (count_visible_active != 0) ? visible_active : 
                                            (warp_active & ~warp_stalled & ~total_barrier_stall & ~warp_lock);

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

    assign stall_out = ~ifetch_req_if.ready && ifetch_req_if.valid;    

    VX_generic_register #( 
        .N(1 + `NUM_THREADS + 32 + `NW_BITS)
    ) fetch_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_out),
        .flush (0),
        .in    ({(| thread_mask),     thread_mask,               warp_pc,               warp_to_schedule}),
        .out   ({ifetch_req_if.valid, ifetch_req_if.thread_mask, ifetch_req_if.curr_PC, ifetch_req_if.wid})
    );

    assign busy = (warp_active != 0); 

endmodule