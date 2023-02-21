`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_warp_sched #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

    input base_dcrs_t       base_dcrs,

    VX_warp_ctl_if.slave    warp_ctl_if,
    VX_wrelease_if.slave    wrelease_if,
    VX_join_if.slave        join_if,
    VX_branch_ctl_if.slave  branch_ctl_if,

    VX_ifetch_req_if.master ifetch_req_if,

    VX_fetch_to_csr_if.master fetch_to_csr_if,

    VX_cmt_to_fetch_if.slave cmt_to_fetch_if,

    // Status
    output wire             busy
);
    `UNUSED_PARAM (CORE_ID)

    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    wire                    join_else;
    wire [31:0]             join_pc;
    wire [`NUM_THREADS-1:0] join_tmask;

    reg [`NUM_WARPS-1:0] active_warps, active_warps_n;   // real active warps (updated when a warp is activated or disabled)
    reg [`NUM_WARPS-1:0] stalled_warps;  // asserted when a branch/gpgpu instructions are issued
    
    reg [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;
    reg [`NUM_WARPS-1:0][`XLEN-1:0] warp_pcs;

    // barriers
    reg [`NUM_BARRIERS-1:0][`NUM_WARPS-1:0] barrier_masks; // warps waiting on barrier
    wire reached_barrier_limit; // the expected number of warps reached the barrier
    
    // wspawn
    reg [31:0]              wspawn_pc;
    reg [`NUM_WARPS-1:0]    use_wspawn;   

    wire [NW_WIDTH-1:0]     schedule_wid;
    wire [`NUM_THREADS-1:0] schedule_tmask;
    wire [31:0]             schedule_pc;
    wire                    schedule_valid;
    wire                    schedule_ready;

    reg [`PERF_CTR_BITS-1:0] cycles;

    reg [`NUM_WARPS-1:0][UUID_WIDTH-1:0] issued_instrs;
    wire [UUID_WIDTH-1:0] instr_uuid;

    wire schedule_fire = schedule_valid && schedule_ready;

    wire ifetch_req_fire = ifetch_req_if.valid && ifetch_req_if.ready;

    wire tmc_active = (warp_ctl_if.tmc.tmask != 0);   

    always @(*) begin
        active_warps_n = active_warps;
        if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
            active_warps_n = warp_ctl_if.wspawn.wmask;
        end
        if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
            active_warps_n[warp_ctl_if.wid] = tmc_active;
        end        
    end

    `UNUSED_VAR (base_dcrs)

    always @(posedge clk) begin
        if (reset) begin
            barrier_masks   <= '0;
            use_wspawn      <= '0;
            stalled_warps   <= '0;
            warp_pcs        <= '0;
            active_warps    <= '0;
            thread_masks    <= '0;
            issued_instrs   <= '0;
            cycles          <= '0;

            // activate first warp
            warp_pcs[0]     <= base_dcrs.startup_addr;
            active_warps[0] <= 1;
            thread_masks[0] <= 1;
        end else begin
            // join handling
            if (join_if.valid) begin
                if (join_else) begin
                    warp_pcs[join_if.wid] <= `XLEN'(join_pc);
                end
                thread_masks[join_if.wid] <= join_tmask;
            end

            if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
                use_wspawn <= warp_ctl_if.wspawn.wmask & (~`NUM_WARPS'(1));
                wspawn_pc  <= warp_ctl_if.wspawn.pc;
            end

            if (warp_ctl_if.valid && warp_ctl_if.barrier.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (reached_barrier_limit) begin
                    barrier_masks[warp_ctl_if.barrier.id] <= '0;
                end else begin
                    barrier_masks[warp_ctl_if.barrier.id][warp_ctl_if.wid] <= 1;
                end
            end 
            
            if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
                thread_masks[warp_ctl_if.wid]  <= warp_ctl_if.tmc.tmask;
                stalled_warps[warp_ctl_if.wid] <= 0;
            end 
            
            if (warp_ctl_if.valid && warp_ctl_if.split.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (warp_ctl_if.split.diverged) begin
                    thread_masks[warp_ctl_if.wid] <= warp_ctl_if.split.then_tmask;
                end
            end

            // Branch
            if (branch_ctl_if.valid) begin
                if (branch_ctl_if.taken) begin
                    warp_pcs[branch_ctl_if.wid] <= branch_ctl_if.dest;
                end
                stalled_warps[branch_ctl_if.wid] <= 0;
            end

            if (schedule_fire) begin
                // stall the warp until decode stage
                stalled_warps[schedule_wid] <= 1;

                // release wspawn
                if (use_wspawn[schedule_wid]) begin
                    thread_masks[schedule_wid] <= 1;
                end
                use_wspawn[schedule_wid] <= 0;

                issued_instrs[schedule_wid] <= issued_instrs[schedule_wid] + UUID_WIDTH'(1);
            end

            if (ifetch_req_fire) begin
                warp_pcs[ifetch_req_if.wid] <= `XLEN'(`XLEN'(ifetch_req_if.PC) + 4);
            end

            if (wrelease_if.valid) begin
                stalled_warps[wrelease_if.wid] <= 0;
            end

            if (busy) begin
                cycles <= cycles + 1;
            end

            active_warps <= active_warps_n;
        end
    end

    // export cycles counter
    assign fetch_to_csr_if.cycles = cycles;

    // calculate active barrier status

`IGNORE_UNUSED_BEGIN
    wire [$clog2(`NUM_WARPS+1)-1:0] active_barrier_count;
`IGNORE_UNUSED_END
    wire [`NUM_WARPS-1:0] barrier_mask = barrier_masks[warp_ctl_if.barrier.id];
    `POP_COUNT(active_barrier_count, barrier_mask);

    assign reached_barrier_limit = (active_barrier_count[NW_WIDTH-1:0] == warp_ctl_if.barrier.size_m1);

    reg [`NUM_WARPS-1:0] barrier_stalls;
    always @(*) begin
        barrier_stalls = barrier_masks[0];
        for (integer i = 1; i < `NUM_BARRIERS; ++i) begin
            barrier_stalls |= barrier_masks[i];
        end
    end

    // split/join stack management    

    wire [(32+`NUM_THREADS)-1:0] ipdom_data [`NUM_WARPS-1:0]; 
    wire ipdom_index [`NUM_WARPS-1:0];

    `RESET_RELAY (ipdom_stack_reset, reset);
    
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        wire push = warp_ctl_if.valid 
                 && warp_ctl_if.split.valid
                 && (i == warp_ctl_if.wid);

        wire pop = join_if.valid && (i == join_if.wid);

        wire [`NUM_THREADS-1:0] else_tmask = warp_ctl_if.split.else_tmask;
        wire [`NUM_THREADS-1:0] orig_tmask = thread_masks[warp_ctl_if.wid];

        wire [(32+`NUM_THREADS)-1:0] q_else = {warp_ctl_if.split.pc, else_tmask};
        wire [(32+`NUM_THREADS)-1:0] q_end  = {32'b0,                orig_tmask};

        VX_ipdom_stack #(
            .WIDTH (32+`NUM_THREADS), 
            .DEPTH (`IPDOM_STACK_SIZE)
        ) ipdom_stack (
            .clk   (clk),
            .reset (ipdom_stack_reset),
            .push  (push),
            .pop   (pop),
            .pair  (warp_ctl_if.split.diverged),
            .q1    (q_end),
            .q2    (q_else),
            .d     (ipdom_data[i]),
            .index (ipdom_index[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full)
        );
    end

    assign {join_pc, join_tmask} = ipdom_data[join_if.wid];
    assign join_else = ~ipdom_index[join_if.wid];

    // schedule the next ready warp

    wire [`NUM_WARPS-1:0] ready_warps = active_warps & ~(stalled_warps | barrier_stalls);

    VX_lzc #(
        .N       (`NUM_WARPS),
        .REVERSE (1)
    ) wid_select (
        .data_in   (ready_warps),
        .data_out  (schedule_wid),
        .valid_out (schedule_valid)
    );

    wire [`NUM_WARPS-1:0][(`NUM_THREADS + `XLEN)-1:0] schedule_data;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign schedule_data[i] = {(use_wspawn[i] ? `NUM_THREADS'(1) : thread_masks[i]),
                                   (use_wspawn[i] ? `XLEN'(wspawn_pc) : warp_pcs[i])};
    end

    assign {schedule_tmask, schedule_pc} = {schedule_data[schedule_wid][(`NUM_THREADS + `XLEN)-1:(`NUM_THREADS + `XLEN)-4], schedule_data[schedule_wid][(`NUM_THREADS + 32)-5:0]};

`ifndef NDEBUG
    assign instr_uuid = UUID_WIDTH'(issued_instrs[schedule_wid] * `NUM_WARPS * `NUM_CORES * `NUM_CLUSTERS)
                      + UUID_WIDTH'(`NUM_WARPS * CORE_ID)
                      + UUID_WIDTH'(schedule_wid);
`else
    assign instr_uuid = '0;
`endif

    VX_generic_buffer #( 
        .DATAW   (UUID_WIDTH + `NUM_THREADS + 32 + NW_WIDTH),
        .OUT_REG (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .valid_in (schedule_valid),
        .ready_in (schedule_ready),
        .data_in  ({instr_uuid,         schedule_tmask,      schedule_pc,      schedule_wid}),
        .data_out ({ifetch_req_if.uuid, ifetch_req_if.tmask, ifetch_req_if.PC, ifetch_req_if.wid}),
        .valid_out (ifetch_req_if.valid),
        .ready_out (ifetch_req_if.ready)
    );

    reg [7:0] pending_instrs;

    always @(posedge clk) begin
		if (reset) begin
			pending_instrs <= '0;
		end else begin
			pending_instrs <= pending_instrs 
                            + 8'(schedule_fire) 
                            - ({8{cmt_to_fetch_if.valid}} & 8'(cmt_to_fetch_if.committed));
		end
	end

    `BUFFER_BUSY ((active_warps != 0 || pending_instrs != 0), 1);
          
    reg [31:0] timeout_ctr;
    reg timeout_enable;
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr    <= '0;
            timeout_enable <= 0;
        end else begin
            if (wrelease_if.valid) begin
                timeout_enable <= 1;
            end
            if (timeout_enable && active_warps !=0 && active_warps == stalled_warps) begin
                timeout_ctr <= timeout_ctr + 1;
            end else if (active_warps == 0 || active_warps != stalled_warps) begin
                timeout_ctr <= '0;
            end
        end
    end
    `RUNTIME_ASSERT(timeout_ctr < `STALL_TIMEOUT, ("%t: *** core%0d-scheduler-timeout: stalled_warps=%b", $time, CORE_ID, stalled_warps));

endmodule
