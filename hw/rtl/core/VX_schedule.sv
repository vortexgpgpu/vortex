// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_schedule import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_pipeline_perf_if.schedule perf_schedule_if,
`endif

    // configuration
    input base_dcrs_t       base_dcrs,

    // inputsdecode_if
    VX_warp_ctl_if.slave    warp_ctl_if, 
    VX_branch_ctl_if.slave  branch_ctl_if [`NUM_ALU_BLOCKS],
    VX_decode_sched_if.slave decode_sched_if,
    VX_commit_sched_if.slave commit_sched_if,

    // outputs
    VX_schedule_if.master   schedule_if,
`ifdef GBAR_ENABLE
    VX_gbar_bus_if.master   gbar_bus_if,
`endif
    VX_sched_csr_if.master  sched_csr_if,

    // status
    output wire             busy
);
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_WARPS-1:0] active_warps, active_warps_n; // updated when a warp is activated or disabled
    reg [`NUM_WARPS-1:0] stalled_warps, stalled_warps_n;  // set when branch/gpgpu instructions are issued
    
    reg [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks, thread_masks_n;
    reg [`NUM_WARPS-1:0][`XLEN-1:0] warp_pcs, warp_pcs_n;

    wire [`NW_WIDTH-1:0]    schedule_wid;
    wire [`NUM_THREADS-1:0] schedule_tmask;
    wire [`XLEN-1:0]        schedule_pc;
    wire                    schedule_valid;
    wire                    schedule_ready;

    // split/join
    wire                    join_valid;
    wire                    join_is_dvg;
    wire                    join_is_else;
    wire [`NW_WIDTH-1:0]    join_wid;   
    wire [`NUM_THREADS-1:0] join_tmask;
    wire [`XLEN-1:0]        join_pc;

    reg [`PERF_CTR_BITS-1:0] cycles;

    reg [`NUM_WARPS-1:0][`UUID_WIDTH-1:0] issued_instrs;

    wire schedule_fire = schedule_valid && schedule_ready;
    wire schedule_if_fire = schedule_if.valid && schedule_if.ready;

    // branch
    wire [`NUM_ALU_BLOCKS-1:0]                  branch_valid;    
    wire [`NUM_ALU_BLOCKS-1:0][`NW_WIDTH-1:0]   branch_wid;    
    wire [`NUM_ALU_BLOCKS-1:0]                  branch_taken;
    wire [`NUM_ALU_BLOCKS-1:0][`XLEN-1:0]       branch_dest;
    for (genvar i = 0; i < `NUM_ALU_BLOCKS; ++i) begin
        assign branch_valid[i] = branch_ctl_if[i].valid;
        assign branch_wid[i]   = branch_ctl_if[i].wid;
        assign branch_taken[i] = branch_ctl_if[i].taken;
        assign branch_dest[i]  = branch_ctl_if[i].dest;
    end

    // barriers
    reg [`NUM_BARRIERS-1:0][`NUM_WARPS-1:0] barrier_masks, barrier_masks_n;
    reg [`NUM_WARPS-1:0] barrier_stalls, barrier_stalls_n;
    wire [`CLOG2(`NUM_WARPS+1)-1:0] active_barrier_count;
    wire [`NUM_WARPS-1:0] curr_barrier_mask;    
`ifdef GBAR_ENABLE
    reg [`NUM_WARPS-1:0] curr_barrier_mask_n;
    reg gbar_req_valid;
    reg [`NB_WIDTH-1:0] gbar_req_id;
    reg [`NC_WIDTH-1:0] gbar_req_size_m1;
`endif

    assign curr_barrier_mask = barrier_masks[warp_ctl_if.barrier.id];
    `POP_COUNT(active_barrier_count, curr_barrier_mask);
    `UNUSED_VAR (active_barrier_count)

    always @(*) begin
        active_warps_n  = active_warps;
        stalled_warps_n = stalled_warps;
        thread_masks_n  = thread_masks;
        barrier_masks_n = barrier_masks;
        barrier_stalls_n= barrier_stalls;
        warp_pcs_n      = warp_pcs;

        // wspawn handling
        if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
            active_warps_n |= warp_ctl_if.wspawn.wmask;
            for (integer i = 0; i < `NUM_WARPS; ++i) begin
                if (warp_ctl_if.wspawn.wmask[i]) begin
                    thread_masks_n[i][0] = 1;
                    warp_pcs_n[i] = warp_ctl_if.wspawn.pc;
                end
            end
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end
        
        // TMC handling
        if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
            active_warps_n[warp_ctl_if.wid]  = (warp_ctl_if.tmc.tmask != 0);
            thread_masks_n[warp_ctl_if.wid]  = warp_ctl_if.tmc.tmask;
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end
        
        // split handling
        if (warp_ctl_if.valid && warp_ctl_if.split.valid) begin
            if (warp_ctl_if.split.is_dvg) begin
                thread_masks_n[warp_ctl_if.wid] = warp_ctl_if.split.then_tmask;
            end
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end

        // join handling
        if (join_valid) begin
            if (join_is_dvg) begin
                if (join_is_else) begin
                    warp_pcs_n[join_wid] = join_pc;
                end
                thread_masks_n[join_wid] = join_tmask;
            end
            stalled_warps_n[join_wid] = 0; // unlock warp
        end

        // barrier handling        
    `ifdef GBAR_ENABLE
        curr_barrier_mask_n = curr_barrier_mask;
        curr_barrier_mask_n[warp_ctl_if.wid] = 1;
    `endif
        if (warp_ctl_if.valid && warp_ctl_if.barrier.valid) begin
            if (~warp_ctl_if.barrier.is_global 
             && (active_barrier_count[`NW_WIDTH-1:0] == warp_ctl_if.barrier.size_m1[`NW_WIDTH-1:0])) begin                                
                barrier_masks_n[warp_ctl_if.barrier.id] = '0;
                barrier_stalls_n &= ~barrier_masks[warp_ctl_if.barrier.id];
            end else begin
                barrier_masks_n[warp_ctl_if.barrier.id][warp_ctl_if.wid] = 1;
                barrier_stalls_n[warp_ctl_if.wid] = 1;
            end
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end
    `ifdef GBAR_ENABLE
        if (gbar_bus_if.rsp_valid && (gbar_req_id == gbar_bus_if.rsp_id)) begin
            barrier_masks_n[gbar_bus_if.rsp_id] = '0;
            barrier_stalls_n = '0; // unlock all warps
        end
    `endif

        // Branch handling
        for (integer i = 0; i < `NUM_ALU_BLOCKS; ++i) begin
            if (branch_valid[i]) begin
                if (branch_taken[i]) begin
                    warp_pcs_n[branch_wid[i]] = branch_dest[i];
                end
                stalled_warps_n[branch_wid[i]] = 0; // unlock warp
            end
        end

        // decode unlock
        if (decode_sched_if.valid && ~decode_sched_if.is_wstall) begin
            stalled_warps_n[decode_sched_if.wid] = 0;
        end

        // CSR unlock
        if (sched_csr_if.unlock_warp) begin
            stalled_warps_n[sched_csr_if.unlock_wid] = 0;
        end

        // stall the warp until decode stage
        if (schedule_fire) begin
            stalled_warps_n[schedule_wid] = 1;
        end

        // advance PC
        if (schedule_if_fire) begin
            warp_pcs_n[schedule_if.data.wid] = schedule_if.data.PC + 4;
        end
    end

    `UNUSED_VAR (base_dcrs)

    always @(posedge clk) begin
        if (reset) begin
            barrier_masks   <= '0;
        `ifdef GBAR_ENABLE
            gbar_req_valid  <= 0;
        `endif
            stalled_warps   <= '0;
            warp_pcs        <= '0;
            active_warps    <= '0;
            thread_masks    <= '0;
            barrier_stalls  <= '0;
            issued_instrs   <= '0;
            cycles          <= '0;

            // activate first warp
            warp_pcs[0]     <= base_dcrs.startup_addr;
            active_warps[0] <= 1;
            thread_masks[0][0] <= 1;
        end else begin
            active_warps   <= active_warps_n;
            stalled_warps  <= stalled_warps_n;
            thread_masks   <= thread_masks_n;
            warp_pcs       <= warp_pcs_n;
            barrier_masks  <= barrier_masks_n;
            barrier_stalls <= barrier_stalls_n;

            // global barrier scheduling
        `ifdef GBAR_ENABLE
            if (warp_ctl_if.valid && warp_ctl_if.barrier.valid
             && warp_ctl_if.barrier.is_global 
             && (curr_barrier_mask_n == active_warps)) begin
                gbar_req_valid <= 1;
                gbar_req_id <= warp_ctl_if.barrier.id;
                gbar_req_size_m1 <= warp_ctl_if.barrier.size_m1[`NC_WIDTH-1:0];
            end
            if (gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
                gbar_req_valid <= 0;
            end
        `endif

            if (schedule_if_fire) begin
                issued_instrs[schedule_if.data.wid] <= issued_instrs[schedule_if.data.wid] + `UUID_WIDTH'(1);
            end

            if (busy) begin
                cycles <= cycles + 1;
            end            
        end
    end

    // barrier handling

`ifdef GBAR_ENABLE
    assign gbar_bus_if.req_valid   = gbar_req_valid;
    assign gbar_bus_if.req_id      = gbar_req_id;
    assign gbar_bus_if.req_size_m1 = gbar_req_size_m1;
    assign gbar_bus_if.req_core_id = `NC_WIDTH'(CORE_ID % `NUM_CORES);
`endif

    // split/join handling

    `RESET_RELAY (split_join_reset, reset);

    VX_split_join #(
        .CORE_ID (CORE_ID)
    ) split_join (
        .clk        (clk),
        .reset      (split_join_reset),
        .valid      (warp_ctl_if.valid),
        .wid        (warp_ctl_if.wid),
        .split      (warp_ctl_if.split),
        .sjoin      (warp_ctl_if.sjoin),
        .join_valid (join_valid),
        .join_is_dvg (join_is_dvg),
        .join_is_else (join_is_else),
        .join_wid   (join_wid), 
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
        assign schedule_data[i] = {thread_masks[i], warp_pcs[i]};
    end

    assign {schedule_tmask, schedule_pc} = {
        schedule_data[schedule_wid][(`NUM_THREADS + `XLEN)-1:(`NUM_THREADS + `XLEN)-4], 
        schedule_data[schedule_wid][(`NUM_THREADS + `XLEN)-5:0]
    };

`ifndef NDEBUG
    localparam GNW_WIDTH = `LOG2UP(`NUM_CLUSTERS * `NUM_CORES * `NUM_WARPS);
    reg [`UUID_WIDTH-1:0] instr_uuid;
    wire [GNW_WIDTH-1:0] g_wid = (GNW_WIDTH'(CORE_ID) << `NW_BITS) + GNW_WIDTH'(schedule_wid);
`ifdef SV_DPI
    always @(posedge clk) begin
        if (reset) begin
            instr_uuid <= `UUID_WIDTH'(dpi_uuid_gen(1, 0, 0));
        end else if (schedule_fire) begin
            instr_uuid <= `UUID_WIDTH'(dpi_uuid_gen(0, 32'(g_wid), 64'(schedule_pc)));
        end        
    end
`else
    wire [GNW_WIDTH+16-1:0] w_uuid = {g_wid, 16'(schedule_pc)};
    always @(*) begin
        instr_uuid = `UUID_WIDTH'(w_uuid);
    end
`endif
`else
    wire [`UUID_WIDTH-1:0] instr_uuid = '0;
`endif

    VX_elastic_buffer #( 
        .DATAW (`NUM_THREADS + `XLEN + `NW_WIDTH)
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (schedule_valid),
        .ready_in  (schedule_ready),
        .data_in   ({schedule_tmask, schedule_pc, schedule_wid}),
        .data_out  ({schedule_if.data.tmask, schedule_if.data.PC, schedule_if.data.wid}),
        .valid_out (schedule_if.valid),
        .ready_out (schedule_if.ready)
    );

    assign schedule_if.data.uuid = instr_uuid;

    `RESET_RELAY (pending_instr_reset, reset);

    wire no_pending_instr;
    VX_pending_instr #( 
        .CTR_WIDTH  (12),
        .DECR_COUNT (`ISSUE_WIDTH),
        .ALM_EMPTY  (1)
    ) pending_instr(
        .clk       (clk),
        .reset     (pending_instr_reset),
        .incr      (schedule_if_fire),
        .incr_wid  (schedule_if.data.wid),
        .decr      (commit_sched_if.committed),
        .decr_wid  (commit_sched_if.committed_wid),
        .alm_empty_wid (sched_csr_if.alm_empty_wid),
        .alm_empty (sched_csr_if.alm_empty),
        .empty     (no_pending_instr)
    );

    `BUFFER_EX(busy, (active_warps != 0 || ~no_pending_instr), 1'b1, 1);

    // export CSRs
    assign sched_csr_if.cycles = cycles;
    assign sched_csr_if.active_warps = active_warps;
    assign sched_csr_if.thread_masks = thread_masks;
          
   // timeout handling
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

`ifdef PERF_ENABLE    
    reg [`PERF_CTR_BITS-1:0] perf_sched_idles;
    reg [`PERF_CTR_BITS-1:0] perf_sched_stalls;

    wire schedule_idle = ~schedule_valid;
    wire schedule_stall = schedule_if.valid && ~schedule_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            perf_sched_idles  <= '0;
            perf_sched_stalls <= '0;            
        end else begin
            perf_sched_idles  <= perf_sched_idles + `PERF_CTR_BITS'(schedule_idle);
            perf_sched_stalls <= perf_sched_stalls + `PERF_CTR_BITS'(schedule_stall);
        end
    end

    assign perf_schedule_if.sched_idles = perf_sched_idles;
    assign perf_schedule_if.sched_stalls = perf_sched_stalls;    
`endif

endmodule
