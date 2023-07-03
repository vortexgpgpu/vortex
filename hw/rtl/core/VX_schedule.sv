`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_schedule #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

    // configuration
    input base_dcrs_t       base_dcrs,

    // inputsdecode_if
    VX_warp_ctl_if.slave    warp_ctl_if,    
    VX_branch_ctl_if.slave  branch_ctl_if,
    VX_decode_sched_if.slave decode_sched_if,
    VX_commit_sched_if.slave commit_sched_if,

    // outputs
    VX_schedule_if.master   schedule_if,
    VX_gbar_bus_if.master   gbar_bus_if,
    VX_sched_csr_if.master  sched_csr_if,

    // status
    output wire             busy
);
    `UNUSED_PARAM (CORE_ID)

    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NC_WIDTH   = `UP(`NC_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    reg [`NUM_WARPS-1:0] active_warps, active_warps_n; // updated when a warp is activated or disabled
    reg [`NUM_WARPS-1:0] stalled_warps;  // set when branch/gpgpu instructions are issued
    
    reg [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;
    reg [`NUM_WARPS-1:0][`XLEN-1:0] warp_pcs;

    // barriers
    reg [`NUM_BARRIERS-1:0][`NUM_WARPS-1:0] barrier_masks;
    wire [$clog2(`NUM_WARPS+1)-1:0] active_barrier_count;
    wire [`NUM_WARPS-1:0] curr_barrier_mask;
    reg [`NUM_WARPS-1:0] curr_barrier_mask_n;
    reg gbar_req_valid;
    reg [`NB_BITS-1:0] gbar_req_id;
    reg [NC_WIDTH-1:0] gbar_req_size_m1;
    
    // wspawn
    reg [`XLEN-1:0]         wspawn_pc;
    reg [`NUM_WARPS-1:0]    use_wspawn;   

    wire [NW_WIDTH-1:0]     schedule_wid;
    wire [`NUM_THREADS-1:0] schedule_tmask;
    wire [`XLEN-1:0]        schedule_pc;
    wire                    schedule_valid;
    wire                    schedule_ready;

    // split/join
    wire                    split_is_dvg;
    wire [`NUM_THREADS-1:0] split_tmask;
    wire                    join_is_dvg;
    wire                    join_is_else;    
    wire [`NUM_THREADS-1:0] join_tmask;
    wire [`XLEN-1:0]        join_pc;

    reg [`PERF_CTR_BITS-1:0] cycles;

    reg [`NUM_WARPS-1:0][UUID_WIDTH-1:0] issued_instrs;
    wire [UUID_WIDTH-1:0] instr_uuid;

    wire schedule_fire = schedule_valid && schedule_ready;
    wire schedule_if_fire = schedule_if.valid && schedule_if.ready;

    wire tmc_active = (warp_ctl_if.tmc.tmask != 0);   

    always @(*) begin
        active_warps_n = active_warps;
        if (warp_ctl_if.wspawn.valid) begin
            active_warps_n = warp_ctl_if.wspawn.wmask;
        end
        if (warp_ctl_if.tmc.valid) begin
            active_warps_n[warp_ctl_if.wid] = tmc_active;
        end
    end

    `UNUSED_VAR (base_dcrs)

    always @(posedge clk) begin
        if (reset) begin
            barrier_masks   <= '0;
            gbar_req_valid  <= 0;
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
            if (warp_ctl_if.sjoin.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (join_is_dvg) begin
                    if (join_is_else) begin
                        warp_pcs[warp_ctl_if.wid] <= `XLEN'(join_pc);
                    end
                    thread_masks[warp_ctl_if.wid] <= join_tmask;
                end
            end

            if (warp_ctl_if.wspawn.valid) begin
                use_wspawn <= warp_ctl_if.wspawn.wmask & (~`NUM_WARPS'(1));
                wspawn_pc  <= warp_ctl_if.wspawn.pc;
            end

            // barrier handling
            if (warp_ctl_if.barrier.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (warp_ctl_if.barrier.is_global 
                 && (curr_barrier_mask_n == active_warps)) begin
                    gbar_req_valid <= 1;
                    gbar_req_id <= warp_ctl_if.barrier.id;
                    gbar_req_size_m1 <= warp_ctl_if.barrier.size_m1[NC_WIDTH-1:0];
                    barrier_masks[warp_ctl_if.barrier.id][warp_ctl_if.wid] <= 1;
                end else
                if (~warp_ctl_if.barrier.is_global 
                 && (active_barrier_count[NW_WIDTH-1:0] == warp_ctl_if.barrier.size_m1[NW_WIDTH-1:0])) begin
                    barrier_masks[warp_ctl_if.barrier.id] <= '0;
                end else begin
                    barrier_masks[warp_ctl_if.barrier.id][warp_ctl_if.wid] <= 1;
                end
            end
            if (gbar_bus_if.rsp_valid && (gbar_req_id == gbar_bus_if.rsp_id)) begin
                barrier_masks[gbar_bus_if.rsp_id] <= '0;
            end
            
            // TMC handling
            if (warp_ctl_if.tmc.valid) begin
                thread_masks[warp_ctl_if.wid]  <= warp_ctl_if.tmc.tmask;
                stalled_warps[warp_ctl_if.wid] <= 0;
            end 
            
            // split handling
            if (warp_ctl_if.split.valid) begin
                stalled_warps[warp_ctl_if.wid] <= 0;
                if (split_is_dvg) begin
                    thread_masks[warp_ctl_if.wid] <= split_tmask;
                end
            end

            // Branch handling
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

            if (schedule_if_fire) begin
                warp_pcs[schedule_if.wid] <= `XLEN'(`XLEN'(schedule_if.PC) + 4);
            end

            if (decode_sched_if.valid && ~decode_sched_if.is_wstall) begin
                stalled_warps[decode_sched_if.wid] <= 0;
            end

            if (busy) begin
                cycles <= cycles + 1;
            end

            active_warps <= active_warps_n;
        end

        if (gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            gbar_req_valid <= 0;
        end 
    end

    // export cycles counter
    assign sched_csr_if.cycles = cycles;

    // barrier handling

    assign curr_barrier_mask = barrier_masks[warp_ctl_if.barrier.id];
    `POP_COUNT(active_barrier_count, curr_barrier_mask);
    `UNUSED_VAR (active_barrier_count)

    reg [`NUM_WARPS-1:0] barrier_stalls;
    always @(*) begin
        curr_barrier_mask_n = curr_barrier_mask;        
        curr_barrier_mask_n[warp_ctl_if.wid] = 1;

        barrier_stalls = barrier_masks[0];
        for (integer i = 1; i < `NUM_BARRIERS; ++i) begin
            barrier_stalls |= barrier_masks[i];
        end
    end

    assign gbar_bus_if.req_valid   = gbar_req_valid;
    assign gbar_bus_if.req_id      = gbar_req_id;
    assign gbar_bus_if.req_size_m1 = gbar_req_size_m1;
    assign gbar_bus_if.req_core_id = NC_WIDTH'(CORE_ID % `NUM_CORES);

    // split/join handling  

    VX_split_join #(
        .CORE_ID (CORE_ID)
    ) split_join (
        .clk        (clk),
        .reset      (reset),
        .wid        (warp_ctl_if.wid),
        .split      (warp_ctl_if.split),
        .sjoin      (warp_ctl_if.sjoin),
        .split_is_dvg (split_is_dvg),
        .split_tmask (split_tmask),
        .split_ret  (warp_ctl_if.split_ret),
        .join_is_dvg (join_is_dvg),
        .join_is_else (join_is_else),
        .join_tmask (join_tmask),
        .join_pc    (join_pc)
    );

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

    assign {schedule_tmask, schedule_pc} = {
        schedule_data[schedule_wid][(`NUM_THREADS + `XLEN)-1:(`NUM_THREADS + `XLEN)-4], 
        schedule_data[schedule_wid][(`NUM_THREADS + `XLEN)-5:0]
    };

`ifndef NDEBUG
    assign instr_uuid = UUID_WIDTH'(issued_instrs[schedule_wid] * `NUM_WARPS * `NUM_CORES * `NUM_CLUSTERS)
                      + UUID_WIDTH'(`NUM_WARPS * CORE_ID)
                      + UUID_WIDTH'(schedule_wid);
`else
    assign instr_uuid = '0;
`endif

    VX_generic_buffer #( 
        .DATAW   (UUID_WIDTH + `NUM_THREADS + `XLEN + NW_WIDTH),
        .OUT_REG (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .valid_in (schedule_valid),
        .ready_in (schedule_ready),
        .data_in  ({instr_uuid,       schedule_tmask,    schedule_pc,    schedule_wid}),
        .data_out ({schedule_if.uuid, schedule_if.tmask, schedule_if.PC, schedule_if.wid}),
        .valid_out (schedule_if.valid),
        .ready_out (schedule_if.ready)
    );

    reg [7:0] pending_instrs;

    always @(posedge clk) begin
		if (reset) begin
			pending_instrs <= '0;
		end else begin
			pending_instrs <= pending_instrs 
                            + 8'(schedule_fire) 
                            - ({8{commit_sched_if.valid}} & 8'(commit_sched_if.committed));
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
            if (decode_sched_if.valid && ~decode_sched_if.is_wstall) begin
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
