`include "VX_define.vh"

module VX_warp_sched #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_warp_sched
    
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

    `UNUSED_PARAM (CORE_ID)

    wire                    join_else;
    wire [31:0]             join_pc;
    wire [`NUM_THREADS-1:0] join_tm;

    reg [`NUM_WARPS-1:0] active_warps, active_warps_n;   // real active warps (updated when a warp is activated or disabled)
    reg [`NUM_WARPS-1:0] schedule_table, schedule_table_n; // enforces round-robin, barrier, and non-speculating branches
    reg [`NUM_WARPS-1:0] stalled_warps;  // asserted when a branch/gpgpu instructions are issued
    
    // Lock warp until instruction decode to resolve branches
    reg [`NUM_WARPS-1:0] fetch_lock;

    reg [`NUM_THREADS-1:0] thread_masks [`NUM_WARPS-1:0];
    reg [31:0] warp_pcs [`NUM_WARPS-1:0];

    // barriers
    reg [`NUM_WARPS-1:0] barrier_stall_mask [`NUM_BARRIERS-1:0]; // warps waiting on barrier
    wire reached_barrier_limit; // the expected number of warps reached the barrier
    
    // wspawn
    reg [31:0]              use_wspawn_pc;
    reg [`NUM_WARPS-1:0]    use_wspawn;   
    reg [`NW_BITS-1:0]      scheduled_warp;
    wire                    warp_scheduled;

    wire ifetch_rsp_fire = ifetch_rsp_if.valid && ifetch_rsp_if.ready;    

    always @(*) begin
        active_warps_n = active_warps;
        if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
            active_warps_n = warp_ctl_if.wspawn.wmask;
        end
        if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
            active_warps_n[warp_ctl_if.wid] = (warp_ctl_if.tmc.tmask != 0);
        end        
    end

    always @(*) begin
        schedule_table_n = schedule_table;
        if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
            schedule_table_n[warp_ctl_if.wid] = (warp_ctl_if.tmc.tmask != 0);
        end        
        if (warp_scheduled) begin // remove scheduled warp (round-robin)
            schedule_table_n[scheduled_warp] = 0;
        end
    end    

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `NUM_BARRIERS; i++) begin
                barrier_stall_mask[i] <= 0;
            end

            use_wspawn_pc     <= 0;
            use_wspawn        <= 0;
            warp_pcs[0]       <= `STARTUP_ADDR;
            active_warps[0]   <= 1; // Activating first warp
            schedule_table[0] <= 1; // set first warp as ready
            thread_masks[0]   <= 1; // Activating first thread in first warp
            stalled_warps     <= 0;
            fetch_lock        <= 0;      
            
            for (integer i = 1; i < `NUM_WARPS; i++) begin
                warp_pcs[i]       <= 0;
                active_warps[i]   <= 0;
                schedule_table[i] <= 0;
                thread_masks[i]   <= 0;
            end
        end else begin            
            if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
                use_wspawn    <= warp_ctl_if.wspawn.wmask & (~`NUM_WARPS'(1));
                use_wspawn_pc <= warp_ctl_if.wspawn.pc;
            end

            if (warp_ctl_if.valid && warp_ctl_if.barrier.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (reached_barrier_limit) begin
                    barrier_stall_mask[warp_ctl_if.barrier.id] <= 0;
                end else begin
                    barrier_stall_mask[warp_ctl_if.barrier.id][warp_ctl_if.wid] <= 1;
                end
            end else if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
                thread_masks[warp_ctl_if.wid] <= warp_ctl_if.tmc.tmask;
                stalled_warps[warp_ctl_if.wid] <= 0;
            end else if (warp_ctl_if.valid && warp_ctl_if.split.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (warp_ctl_if.split.diverged) begin
                    thread_masks[warp_ctl_if.wid] <= warp_ctl_if.split.then_mask;
                end
            end          

            if (use_wspawn[scheduled_warp] && warp_scheduled) begin
                use_wspawn[scheduled_warp]   <= 0;
                thread_masks[scheduled_warp] <= 1;
            end

            // Stalling the scheduling of warps
            if (wstall_if.valid) begin
                stalled_warps[wstall_if.wid] <= 1;                
            end

            // Branch
            if (branch_ctl_if.valid) begin
                if (branch_ctl_if.taken) begin
                    warp_pcs[branch_ctl_if.wid] <= branch_ctl_if.dest;
                end
                stalled_warps[branch_ctl_if.wid] <= 0;
            end

            // Lock warp until instruction decode to resolve branches
            if (warp_scheduled) begin
                fetch_lock[scheduled_warp] <= 1;
            end
            if (ifetch_rsp_fire) begin
                fetch_lock[ifetch_rsp_if.wid] <= 0;
                warp_pcs[ifetch_rsp_if.wid] <= ifetch_rsp_if.PC + 4;
            end

            // join handling
            if (join_if.valid) begin
                if (join_else) begin
                    warp_pcs[join_if.wid] <= join_pc;
                end
                thread_masks[join_if.wid] <= join_tm;
            end

            active_warps <= active_warps_n;

            // reset 'schedule_table' when it goes to zero
            schedule_table <= (| schedule_table_n) ? schedule_table_n : active_warps_n;
        end
    end

    // calculate active barrier status

`IGNORE_WARNINGS_BEGIN
    wire [`NW_BITS:0] active_barrier_count;
`IGNORE_WARNINGS_END
    assign active_barrier_count = $countones(barrier_stall_mask[warp_ctl_if.barrier.id]);

    assign reached_barrier_limit = (active_barrier_count[`NW_BITS-1:0] == warp_ctl_if.barrier.size_m1);

    reg [`NUM_WARPS-1:0] total_barrier_stall;
    always @(*) begin
        total_barrier_stall = barrier_stall_mask[0];
        for (integer i = 1; i < `NUM_BARRIERS; ++i) begin
            total_barrier_stall |= barrier_stall_mask[i];
        end
    end

    // split/join stack management    

    wire [(1+32+`NUM_THREADS-1):0] ipdom [`NUM_WARPS-1:0];
    
    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        wire push = warp_ctl_if.valid 
                 && warp_ctl_if.split.valid
                 && (i == warp_ctl_if.wid);

        wire pop = join_if.valid && (i == join_if.wid);

        wire [`NUM_THREADS-1:0] else_mask = warp_ctl_if.split.diverged ? warp_ctl_if.split.else_mask : thread_masks[warp_ctl_if.wid];
        wire [(1+32+`NUM_THREADS-1):0] q_end  = {1'b0, 32'b0,                thread_masks[warp_ctl_if.wid]};
        wire [(1+32+`NUM_THREADS-1):0] q_else = {1'b1, warp_ctl_if.split.pc, else_mask};

        VX_ipdom_stack #(
            .WIDTH (1+32+`NUM_THREADS), 
            .DEPTH (2 ** (`NT_BITS+1))
        ) ipdom_stack (
            .clk   (clk),
            .reset (reset),
            .push  (push),
            .pop   (pop),
            .q1    (q_end),
            .q2    (q_else),
            .d     (ipdom[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full)
        );
    end

    assign {join_else, join_pc, join_tm} = ipdom [join_if.wid];

    // calculate next warp schedule

    reg [`NUM_THREADS-1:0] thread_mask;
    reg schedule_valid;
    reg [31:0] warp_pc;
    
    wire [`NUM_WARPS-1:0] schedule_ready = schedule_table & ~(stalled_warps | total_barrier_stall | fetch_lock);

    always @(*) begin
        schedule_valid = 0;
        thread_mask    = 'x;
        warp_pc        = 'x;
        scheduled_warp = 'x;
        for (integer i = 0; i < `NUM_WARPS; ++i) begin
            if (schedule_ready[i]) begin
                schedule_valid = 1;
                thread_mask = use_wspawn[i] ? `NUM_THREADS'(1) : thread_masks[i];
                warp_pc = use_wspawn[i] ? use_wspawn_pc : warp_pcs[i];
                scheduled_warp = `NW_BITS'(i);
                break;
            end
        end    
    end

    wire stall_out = ~ifetch_req_if.ready && ifetch_req_if.valid;   

    assign warp_scheduled = schedule_valid && ~stall_out;

    VX_pipe_register #( 
        .DATAW  (1 + `NUM_THREADS + 32 + `NW_BITS),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({warp_scheduled,      thread_mask,         warp_pc,          scheduled_warp}),
        .data_out ({ifetch_req_if.valid, ifetch_req_if.tmask, ifetch_req_if.PC, ifetch_req_if.wid})
    );

    assign busy = (active_warps != 0); 

    `SCOPE_ASSIGN (wsched_scheduled_warp, warp_scheduled);
    `SCOPE_ASSIGN (wsched_active_warps,   active_warps);
    `SCOPE_ASSIGN (wsched_schedule_table, schedule_table);
    `SCOPE_ASSIGN (wsched_schedule_ready, schedule_ready);
    `SCOPE_ASSIGN (wsched_warp_to_schedule, scheduled_warp);
    `SCOPE_ASSIGN (wsched_warp_pc, warp_pc);

endmodule