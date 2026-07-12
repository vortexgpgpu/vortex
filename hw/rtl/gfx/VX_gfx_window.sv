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

// VX_gfx_window — the shared per-core graphics window SFU PE. It owns the
// per-(warp, lane) slot register file and the generic window macro-ops
// (CUSTOM1 funct3=6: SETW writes one slot, GETWF/GETW read a slot window to the
// FP/GP file). The window is reused by the FF graphics units (TEX/OM payload and
// result windows) and is present whenever any graphics extension is enabled.
//
// Storage is a synchronous, lane-packed RAM: one word holds all NUM_LANES copies
// of a slot, addressed by {warp, simd-group, slot}. Reads take one cycle.
//
// A read port is a full RAM mirror, so mirrors are the window's dominant area
// term. There is exactly ONE: the core-op port, which the RTU time-shares. The
// RTU is now the window's only tenant — its ray-in / hit-out records genuinely
// exceed any RISC-V encoding, which is what the window is for. TEX used to spill
// its quad operands here and owned two more mirrors; it now takes u/v/lod in
// registers (vx_tex, R4-type) and computes its mip LOD in the shader, so those
// mirrors are gone (proposal P5-B).
//
// All writers share one write port, resolved by fixed priority:
//   RTU > FILL
// The RTU outranks the execute-side fill so a parked warp's hit record always
// drains; SETW/fill is last because it can stall its own warp. Multi-slot
// writers present one word per cycle and advance only on grant.
// (The raster seed port is gone: a fragment's stamp now rides in its launch and
// lands in the core's launch registers, so the window has no raster tenant.)
//
// The RTU ray-tracing engine is the one consumer that is also a bus MASTER here.
// The window holds no ray, no hit record and no traversal state: when a warp
// arms a TRACE the window writes t_min/t_max like any other fill, rings the
// RTU's doorbell, and retires. The RTU then reads the ray out of the slot RAM
// and later writes the hit record back into it, one slot per beat, each beat
// carrying its own address (see VX_rtu_bus_if). All that remains here is three
// bits per warp:
//   response_ready — a record landed; the parked WAIT may complete
//   trace_open     — that record was a candidate; a CONTINUE may resume it
//   unlock_owed    — the wstall'd TRACE is still waiting for its first record
// each set or cleared by the RTU's write to the status slot, which is by
// construction the last write of a record.
//
// The TRACE macro-op arrives pre-expanded from VX_gfxw_uops (one micro-op per
// cycle): CFG unpacks the lane-packed rs1 config; ORIGIN/DIR stream the f0..f7
// ray window into the RAM; ARM writes t_min/t_max and rings the doorbell.
// WAIT reads the status slot. GETWF/GETW/GETWS are windowed reads; SETW writes
// one slot.

`include "VX_define.vh"

module VX_gfx_window import VX_gpu_pkg::*, VX_gfx_window_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter RTU_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if

`ifdef VX_CFG_EXT_RTU_ENABLE
    ,
    // socket-shared RTU bus (the RTU is the master; this window is its slave)
    VX_rtu_bus_if.master    rtu_bus_if,

    // TRACE wstall release (-> scheduler)
    VX_sched_unlock_if.master sched_unlock_if
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

`ifdef VX_CFG_EXT_RTU_ENABLE
    import VX_rtu_pkg::*;
`endif

    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);

    localparam WIN_GROUPS  = `VX_CFG_NUM_THREADS / NUM_LANES;
    localparam RAM_SIZE    = `VX_CFG_NUM_WARPS * WIN_GROUPS * GFXW_SLOT_COUNT;
    localparam RAM_ADDRW   = `CLOG2(RAM_SIZE);
    localparam RAM_DATAW   = 32 * NUM_LANES;

    // The address packs {warp, simd-group, slot}; both strides must be powers of
    // two or the concatenation below degenerates into a multiplier.
    `STATIC_ASSERT(((1 << GFXW_SLOT_BITS) == GFXW_SLOT_COUNT),
        ("window slot count must be a power of two"))
    `STATIC_ASSERT(((1 << LANE_BITS) == NUM_LANES),
        ("window lane count must be a power of two"))

    // Address a slot of one (warp, simd-group). `tbase` is a thread index, so its
    // low LANE_BITS are zero and the simd-group is the remaining high bits.
    function automatic [RAM_ADDRW-1:0] win_addr (
        input [NW_WIDTH-1:0]       w,
        input [THREAD_BITS-1:0]    tb,
        input [GFXW_SLOT_BITS-1:0] s
    );
        reg [RAM_ADDRW-1:0] grp;
        begin
            grp = (RAM_ADDRW'(w) * RAM_ADDRW'(WIN_GROUPS)) + RAM_ADDRW'(tb >> LANE_BITS);
            win_addr = (grp * RAM_ADDRW'(GFXW_SLOT_COUNT)) + RAM_ADDRW'(s);
        end
    endfunction

    wire [GFXW_OP_BITS-1:0]   op    = execute_if.data.op_args.gfxw.op;
    wire [GFXW_SLOT_BITS-1:0] slot  = execute_if.data.op_args.gfxw.slot[GFXW_SLOT_BITS-1:0];
    wire [NW_WIDTH-1:0]       wid   = execute_if.data.header.wid;
    wire [PID_W-1:0]          pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0]    thread_base = THREAD_BITS'(pid) << LANE_BITS;

    // ── the single write port ──────────────────────────────────────────────
    typedef struct packed {
        logic [NW_WIDTH-1:0]        wid;
        logic [THREAD_BITS-1:0]     tbase;
        logic [GFXW_SLOT_BITS-1:0]  slot;
        logic [NUM_LANES-1:0]       mask;
        logic [NUM_LANES-1:0][31:0] data;
    } win_wr_t;

    win_wr_t wr_rtu, wr_fill;
    wire     req_rtu, req_fill;
    wire     gnt_rtu, gnt_fill;

    assign gnt_rtu  = req_rtu;
    assign gnt_fill = req_fill && ~req_rtu;

    win_wr_t wr_sel;
    always @(*) begin
        case (1'b1)
            gnt_rtu:  wr_sel = wr_rtu;
            default:  wr_sel = wr_fill;
        endcase
    end

    wire                 ram_write = req_rtu || req_fill;
    wire [RAM_ADDRW-1:0] ram_waddr = win_addr(wr_sel.wid, wr_sel.tbase, wr_sel.slot);
    wire [NUM_LANES-1:0] ram_wren  = wr_sel.mask;
    wire [RAM_DATAW-1:0] ram_wdata = wr_sel.data;

    // ── the read port: one mirror, shared by the core ops and the RTU ──────
    wire [RAM_ADDRW-1:0] ram_raddr;
    wire                 ram_rden;
    wire [RAM_DATAW-1:0] ram_rdata;

    // VX_dp_ram is 1W1R. The window used to replicate this RAM once per read port
    // (RD_PORTS = 4) because TEX held two of them; with TEX out (P5-B) a single
    // instance serves the only remaining reader.
    VX_dp_ram #(
        .DATAW    (RAM_DATAW),
        .SIZE     (RAM_SIZE),
        .WRENW    (NUM_LANES),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) win_ram (
        .clk   (clk),
        .reset (reset),
        .read  (ram_rden),
        .write (ram_write),
        .wren  (ram_wren),
        .waddr (ram_waddr),
        .wdata (ram_wdata),
        .raddr (ram_raddr),
        .rdata (ram_rdata)
    );

    // The core ops (GETWF/GETW/GETWS/WAIT) and the RTU share it.
    //
    // The RTU wins, and the priority is not symmetric. RTU demand is bounded: at
    // most two reads are in flight (the return credits) and they come in short
    // bursts (a ray pull at ARM), so a core op waits a few cycles at worst. The
    // converse does not hold — a warp issuing GETW every cycle could starve the
    // RTU indefinitely, and the traced warp is wstall'd, so it cannot break the
    // cycle itself. Core-first would therefore have an unbounded-starvation
    // corner; RTU-first has none.
    wire [RAM_ADDRW-1:0]       core_raddr;
    wire                       core_rden;
    wire [RAM_ADDRW-1:0]       rtu_raddr;
    wire                       rtu_rd_gnt;
    wire [NUM_LANES-1:0][31:0] core_rdata = ram_rdata;
    assign ram_raddr = rtu_rd_gnt ? rtu_raddr : core_raddr;
    assign ram_rden  = rtu_rd_gnt || core_rden;

    // ── op classification ──────────────────────────────────────────────────
    wire is_setw  = (op == GFXW_OP_SETW);
    wire is_getwf = (op == GFXW_OP_GETWF);
    wire is_getw  = (op == GFXW_OP_GETW);
    wire is_getws = (op == GFXW_OP_GETWS);

    // ── result stage (the RAM read is synchronous) ─────────────────────────
    // One op is accepted per cycle into s1; its word arrives from the RAM on the
    // next, where result_if presents it. A pending RAM read still owns the read
    // port's output, so every op that issues a read must wait for s1 to drain.
    reg                       s1_valid;
    reg                       s1_from_ram;
    sfu_header_t              s1_header;
    reg [NUM_LANES-1:0][31:0] s1_data;

    wire s1_ready = ~s1_valid || result_if.ready;

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [2:0] uop = execute_if.data.op_args.gfxw.uop;

    // Lane-packed config rides lanes 1..3 of the rs1 register (the implicit
    // vx_wgather layout: lane1=scene, lane2=payload, lane3=flags|cull). The trace
    // ABI requires SIMD_WIDTH >= 4; clamp the indices so narrower builds (which
    // never issue TRACE) still elaborate.
    localparam CFG_L1 = (NUM_LANES > 1) ? 1 : 0;
    localparam CFG_L2 = (NUM_LANES > 2) ? 2 : 0;
    localparam CFG_L3 = (NUM_LANES > 3) ? 3 : 0;

    wire is_trace  = (op == GFXW_OP_TRACE);
    wire is_arm    = is_trace && (uop == GFXW_UOP_ARM);
    wire is_cfg    = is_trace && (uop == GFXW_UOP_CFG);
    wire is_origin = is_trace && (uop == GFXW_UOP_ORIGIN);
    wire is_dir    = is_trace && (uop == GFXW_UOP_DIR);
    wire is_wait   = (op == GFXW_OP_WAIT);
    wire is_cont   = (op == GFXW_OP_CB_RET);  // CONTINUE reuses the CB_RET encoding

    // Ops whose result word comes out of the slot RAM. WAIT is one of them: the
    // status it returns IS a slot, written by the RTU as the last beat of a
    // record.
    wire is_read = is_getwf || is_getw || is_getws || is_wait;

    // Per-warp trace state (see the header comment).
    reg [`VX_CFG_NUM_WARPS-1:0] response_ready;
    reg [`VX_CFG_NUM_WARPS-1:0] trace_open;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_owed;
    // Scene base is warp-uniform (the CFG uop broadcasts one value to every
    // lane), so it is one word per warp, not per lane.
    reg [31:0] scene_base [`VX_CFG_NUM_WARPS];

    // Register the outgoing RTU bus at this module boundary so the core->socket
    // seam launches at a flop (see VX_rtu_bus_slice). We drive the internal
    // working copy; the RTU core registers the direction it sources.
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (RTU_TAG_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (RTU_TAG_WIDTH),
        .MST_OUT_BUF (3),  // register the arm/req channels we source
        .SLV_OUT_BUF (0)   // win registered by the RTU core output
    ) rtu_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (rtu_bus_w),
        .bus_out_if (rtu_bus_if)
    );
`else
    `UNUSED_PARAM (RTU_TAG_WIDTH)
    wire is_read = is_getwf || is_getw || is_getws;
`endif

    // ── execute-side fill writes (SETW / CFG / ORIGIN / DIR / ARM) ─────────
    // One slot per cycle onto the shared write port; the op is held until its
    // last word is granted.
    reg [1:0] fw_cnt;

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [1:0] fw_need = is_setw ? 2'd1
                       : (is_cfg || is_origin || is_dir) ? 2'd3
                       : is_arm ? 2'd2      // t_min, t_max
                       : 2'd0;
`else
    wire [1:0] fw_need = is_setw ? 2'd1 : 2'd0;
`endif

    wire fw_more = execute_if.valid && (fw_cnt < fw_need);
    // Done once every word is placed. The second term retires on the same cycle
    // as the last grant; the first covers a result-stage stall after that grant,
    // where the counter has already advanced past the last word.
    wire fw_done = (fw_cnt == fw_need)
                || ((fw_need != 2'd0) && gnt_fill && (fw_cnt == (fw_need - 2'd1)));

    // Fill word select.
    reg [GFXW_SLOT_BITS-1:0]  fill_slot;
    reg [NUM_LANES-1:0][31:0] fill_data;
    always @(*) begin
        fill_slot = slot;
        for (integer i = 0; i < NUM_LANES; ++i) begin
            fill_data[i] = execute_if.data.rs1_data[i][31:0];
        end
`ifdef VX_CFG_EXT_RTU_ENABLE
        if (is_cfg) begin
            case (fw_cnt)
                2'd0: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_PAYLOAD_PTR_LO);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = execute_if.data.rs1_data[CFG_L2][31:0];
                    end
                end
                2'd1: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_FLAGS);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = {16'd0, execute_if.data.rs1_data[CFG_L3][15:0]};
                    end
                end
                default: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_CULL_MASK);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = {16'd0, execute_if.data.rs1_data[CFG_L3][31:16]};
                    end
                end
            endcase
        end else if (is_origin || is_dir) begin
            fill_slot = GFXW_SLOT_BITS'((is_origin ? `VX_RT_RAY_ORIGIN : `VX_RT_RAY_DIRECTION) + 32'(fw_cnt));
            for (integer i = 0; i < NUM_LANES; ++i) begin
                case (fw_cnt)
                    2'd0:    fill_data[i] = execute_if.data.rs1_data[i][31:0];
                    2'd1:    fill_data[i] = execute_if.data.rs2_data[i][31:0];
                    default: fill_data[i] = execute_if.data.rs3_data[i][31:0];
                endcase
            end
        end else if (is_arm) begin
            fill_slot = (fw_cnt == 2'd0) ? GFXW_SLOT_BITS'(`VX_RT_T_MIN)
                                         : GFXW_SLOT_BITS'(`VX_RT_T_MAX);
            for (integer i = 0; i < NUM_LANES; ++i) begin
                fill_data[i] = (fw_cnt == 2'd0) ? execute_if.data.rs1_data[i][31:0]
                                                : execute_if.data.rs2_data[i][31:0];
            end
        end
`endif
    end

    assign req_fill      = fw_more;
    assign wr_fill.wid   = wid;
    assign wr_fill.tbase = thread_base;
    assign wr_fill.slot  = fill_slot;
    assign wr_fill.mask  = execute_if.data.header.tmask[thread_base +: NUM_LANES];
    assign wr_fill.data  = fill_data;

`ifdef VX_CFG_EXT_RTU_ENABLE

    // ── the RTU's window port ──────────────────────────────────────────────
    // One slot access per beat, addressed by the beat itself. Writes contend for
    // the shared write port; reads contend for the core-op read mirror, where the
    // RTU has priority, so they only ever wait on a return credit.
    wire                      win_v     = rtu_bus_w.win_valid;
    wire                      win_we    = rtu_bus_w.win_data.we;
    wire [NW_WIDTH-1:0]       win_wid   = rtu_bus_w.win_data.wid;
    wire [THREAD_BITS-1:0]    win_tbase = rtu_bus_w.win_data.tbase;
    wire [GFXW_SLOT_BITS-1:0] win_slot  = rtu_bus_w.win_data.slot;
    `UNUSED_VAR (rtu_bus_w.win_data.tag)

    // Read-return credits. A read is accepted only if its word is guaranteed a
    // place to land: at most two may be in flight (one in the RAM's output
    // register, the rest in the return buffer below).
    reg [1:0] rd_credits;
    wire rd_ok = (rd_credits != 2'd0);

    wire win_rd_go = win_v && ~win_we && rd_ok;

    assign req_rtu = win_v && win_we;
    assign rtu_bus_w.win_ready = win_we ? gnt_rtu : rd_ok;

    assign wr_rtu.wid   = win_wid;
    assign wr_rtu.tbase = win_tbase;
    assign wr_rtu.slot  = win_slot;
    assign wr_rtu.mask  = rtu_bus_w.win_data.mask;
    assign wr_rtu.data  = rtu_bus_w.win_data.data;

    // Read requests go to the shared mirror 0 (the RTU has priority there).
    assign rtu_raddr  = win_addr(win_wid, win_tbase, win_slot);
    assign rtu_rd_gnt = win_rd_go;

    // The word lands in the RAM's output register one cycle after the read is
    // taken, and is pushed into the return buffer from there.
    reg rd_pend;

    wire                       rdata_valid;
    wire [NUM_LANES-1:0][31:0] rdata_word;
    wire                       rdata_pop;
    wire                       rdata_space;

    VX_elastic_buffer #(
        .DATAW   (RAM_DATAW),
        .SIZE    (2),
        .OUT_REG (1)
    ) rdata_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rd_pend),
        .ready_in  (rdata_space),   // the credit count keeps this high
        .data_in   (ram_rdata),
        .data_out  (rdata_word),
        .valid_out (rdata_valid),
        .ready_out (rtu_bus_w.req_ready)
    );
    `UNUSED_VAR (rdata_space)

    // ── the req channel: read returns, and the warp's CONTINUE actions ─────
    // A read return is what the RTU is blocked on, so it goes first; a CONTINUE
    // can wait a cycle. They cannot actually collide — a CONTINUE is what makes
    // the RTU issue its next read — but the priority makes that harmless either
    // way.
    wire cont_want = execute_if.valid && is_cont && trace_open[wid] && s1_ready;

    wire [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_act;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cont_act
        assign cont_act[i] = execute_if.data.rs1_data[i][RTU_CB_ACTION_BITS-1:0];
    end

    assign rtu_bus_w.req_valid     = rdata_valid || cont_want;
    assign rtu_bus_w.req_data.kind = rdata_valid ? RTU_REQ_RDATA : RTU_REQ_CONT;
    assign rtu_bus_w.req_data.data = rdata_word;
    // The CONTINUE carries every active lane's action word, including lanes that
    // are still traversing (PENDING) and whose action is meaningless. The
    // scheduler applies an action only to a lane that actually yielded a
    // candidate, so those ride along harmlessly.
    assign rtu_bus_w.req_data.cb_action = cont_act;

    assign rdata_pop = rdata_valid && rtu_bus_w.req_ready;
    wire cont_fire   = cont_want && ~rdata_valid && rtu_bus_w.req_ready;

    // ── the arm channel: a warp armed a TRACE ─────────────────────────────
    // Rung once the ray is completely in the RAM (t_min/t_max are this op's own
    // fill words), because the RTU reads the ray straight back out of it.
    assign rtu_bus_w.arm_valid           = execute_if.valid && is_arm && fw_done && s1_ready;
    assign rtu_bus_w.arm_data.wid        = wid;
    assign rtu_bus_w.arm_data.tbase      = thread_base;
    assign rtu_bus_w.arm_data.mask       = execute_if.data.header.tmask[thread_base +: NUM_LANES];
    assign rtu_bus_w.arm_data.scene_base = scene_base[wid];
    assign rtu_bus_w.arm_data.tag        = ($bits(rtu_bus_w.arm_data.tag))'(execute_if.data.header.uuid);

    wire arm_fire = rtu_bus_w.arm_valid && rtu_bus_w.arm_ready;

    // ── the status write ──────────────────────────────────────────────────
    // The last write of a record, and the only one the window looks inside: it
    // completes the parked WAIT, reopens or closes the trace, and releases the
    // warp that has been wstall'd since TRACE.
    wire status_wr = req_rtu && gnt_rtu && (win_slot == GFXW_SLOT_BITS'(`VX_RT_STATUS));

    assign sched_unlock_if.valid = status_wr && unlock_owed[win_wid];
    assign sched_unlock_if.wid   = win_wid;

    // ── op completion ─────────────────────────────────────────────────────
    // WAIT blocks until a record has landed. A CONTINUE with no open candidate
    // retires as a no-op (defensive — correct kernels only issue one inside the
    // yield loop).
    // A read op also needs the shared mirror, which the RTU may have taken this
    // cycle; without ~rtu_rd_gnt the op would retire and sample the RTU's word.
    wire read_fire = execute_if.valid && is_read && (~is_wait || response_ready[wid])
                  && s1_ready && ~rtu_rd_gnt;
    wire fill_fire = execute_if.valid && (is_setw || is_cfg || is_origin || is_dir)
                  && fw_done && s1_ready;
    wire cont_ret  = cont_fire
                  || (execute_if.valid && is_cont && ~trace_open[wid] && s1_ready);
    wire op_fire   = read_fire || fill_fire || arm_fire || cont_ret;
`else
    wire read_fire = execute_if.valid && is_read && s1_ready;
    wire fill_fire = execute_if.valid && is_setw && fw_done && s1_ready;
    wire op_fire   = read_fire || fill_fire;
    assign req_rtu    = 1'b0;
    assign wr_rtu     = '0;
    assign rtu_rd_gnt = 1'b0;
    assign rtu_raddr  = '0;
`endif

    // ── core read port address select ──────────────────────────────────────
    // GETWS reads a slot-keyed frag record: the block index is warp-uniform, so
    // it is taken from lane 0 rather than decoded per lane.
    wire [NW_WIDTH-1:0] frag_widx = execute_if.data.rs1_data[0][NW_WIDTH-1:0];
    wire [NW_WIDTH-1:0] rd_wid    = is_getws ? frag_widx : wid;

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [GFXW_SLOT_BITS-1:0] rd_slot = is_wait ? GFXW_SLOT_BITS'(`VX_RT_STATUS) : slot;
`else
    wire [GFXW_SLOT_BITS-1:0] rd_slot = slot;
`endif

    assign core_rden  = read_fire;
    assign core_raddr = win_addr(rd_wid, thread_base, rd_slot);

    // ── sequential state ───────────────────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin
            s1_valid <= 1'b0;
            fw_cnt   <= 2'd0;
`ifdef VX_CFG_EXT_RTU_ENABLE
            response_ready <= '0;
            trace_open     <= '0;
            unlock_owed    <= '0;
            rd_credits     <= 2'd2;
            rd_pend        <= 1'b0;
`endif
        end else begin
            // result stage
            if (s1_valid && result_if.ready) begin
                s1_valid <= 1'b0;
            end
            if (op_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= is_read;
                s1_header   <= execute_if.data.header;
                s1_data     <= '0;
            end

            // fill word advance
            if (op_fire) begin
                fw_cnt <= 2'd0;
            end else if (gnt_fill) begin
                fw_cnt <= fw_cnt + 2'd1;
            end

`ifdef VX_CFG_EXT_RTU_ENABLE
            rd_pend <= win_rd_go;

            case ({win_rd_go, rdata_pop})
                2'b10:   rd_credits <= rd_credits - 2'd1;
                2'b01:   rd_credits <= rd_credits + 2'd1;
                default: ;
            endcase

            // Scene base is warp-uniform; the CFG uop broadcasts one value.
            if (execute_if.valid && is_cfg) begin
                scene_base[wid] <= execute_if.data.rs1_data[CFG_L1][31:0];
            end

            // WAIT consumes the record so the next one blocks again.
            if (read_fire && is_wait) begin
                response_ready[wid] <= 1'b0;
            end
            // A CONTINUE resolves the open candidate.
            if (cont_fire) begin
                trace_open[wid] <= 1'b0;
            end
            if (arm_fire) begin
                unlock_owed[wid] <= 1'b1;
            end
            // Last, so a record landing in the same cycle as the op it unblocks
            // wins: the record is the newer event.
            if (status_wr) begin
                response_ready[win_wid] <= 1'b1;
                trace_open[win_wid]     <= rtu_bus_w.win_data.is_cand;
                unlock_owed[win_wid]    <= 1'b0;
            end
`endif
        end
    end

    // ── result path ───────────────────────────────────────────────────────
    sfu_result_t rsp_data_out;
    assign rsp_data_out.header = s1_header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        wire [31:0] rdata = s1_from_ram ? core_rdata[i] : s1_data[i];
        assign rsp_data_out.data[i] = `VX_CFG_XLEN'(rdata);
    end

    assign result_if.valid = s1_valid;
    assign result_if.data  = rsp_data_out;

    assign execute_if.ready = op_fire;

`ifdef DBG_TRACE_GFXW
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%t: %s gfxw-op: wid=%0d, PC=0x%0h, tmask=%b, op=%0d, slot=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, op, slot, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
