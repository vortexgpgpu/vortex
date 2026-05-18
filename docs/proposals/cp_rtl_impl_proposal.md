# CP RTL Implementation Proposal (`rtl/cp/`)

Status: draft proposal
Branch: `feature_cp`
Parent: [command_processor_proposal.md](command_processor_proposal.md)
Companion: [cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md)

## 1. Scope

This proposal specifies the **RTL implementation** of the Command
Processor (CP) block defined in §6 of the parent CP proposal. It
covers the new `hw/rtl/cp/` tree, the DCR-bus extension to true
request/response on `Vortex.sv`, the XRT AFU shim rework, the DCR
address allocations, and the per-module verification strategy. It is
intended to be detailed enough that an RTL engineer can start coding
without further design calls.

It does **not** redesign the CP architecture. Every module name,
every interface, every command opcode in this document is taken from
§6 of the parent proposal verbatim.

### 1.1 In scope

- Full `hw/rtl/cp/` source tree (~14 files).
- `VX_cp_pkg.sv` package: typedefs, opcodes, parameters.
- `VX_cp_if.sv` SV-interface bundles between CP and AFU, CP and
  Vortex, and CPE and shared resources.
- Per-module ports, parameters, state, FSMs, and key combinational
  logic.
- `Vortex.sv` / `Vortex_axi.sv` top-level DCR bus extension (write-only
  → req/rsp).
- `VX_afu_wrap.sv` (XRT) integration with the CP.
- DCR address-space reservations under `VX_types.toml`.
- Per-module verification: unit testbenches, integration tests, lint
  setup, simulation flow.
- Phased task breakdown aligned with parent migration plan
  (phases 1-5).

### 1.2 Out of scope

- The runtime software — see
  [cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md).
- Per-block helper RTL (TEX / RASTER / OM / DXA programming details) —
  owned by their subsystem proposals; the CP only sees DCR writes.
- OPAE AFU shim (deprecated per parent §7.2).
- Multi-context KMU (phase 7 follow-on).
- Interrupt path (phase 6, v1.1).
- Multi-clock-domain CDC between CP and Vortex (assumed single clock
  in v1; see open question §15.4).

## 2. File layout

```
hw/rtl/cp/
├── VX_cp_pkg.sv          package: opcodes, structs, parameters             (~120 LOC)
├── VX_cp_if.sv           SV interface bundles                              (~150 LOC)
├── VX_cp_core.sv         top-level wrapper; generates N engines + helpers  (~250 LOC)
├── VX_cp_engine.sv       one Command Processor Engine per queue            (~450 LOC)
├── VX_cp_fetch.sv        AXI read of next command cache line               (~150 LOC)
├── VX_cp_unpack.sv       cache-line → packed cmd_t stream                  (~140 LOC)
├── VX_cp_arbiter.sv      generic round-robin arbiter (instantiated 3×)     (~80 LOC)
├── VX_cp_launch.sv       KMU start/busy wrapper                            (~80 LOC)
├── VX_cp_dma.sv          AXI ↔ Vortex memory DMA engine                    (~350 LOC)
├── VX_cp_dcr_proxy.sv    DCR req/rsp gateway                               (~120 LOC)
├── VX_cp_event_unit.sv   wait-on-seqnum comparator + signal gen            (~250 LOC)
├── VX_cp_completion.sv   per-queue seqnum + head writeback                 (~180 LOC)
├── VX_cp_profiling.sv    cycle counter + 32 B timestamp writeback          (~150 LOC)
└── VX_cp_axi_xbar.sv     AXI master multiplexer (fetch+DMA+event+cmpl+prof)(~200 LOC)
                                                                     Total: ~2700 LOC
```

Modifications to existing files:

```
hw/rtl/Vortex.sv               +12 lines  add dcr_rsp_{valid,data} top-level ports
hw/rtl/Vortex_axi.sv           +12 lines  same
hw/rtl/afu/xrt/VX_afu_wrap.sv  ~150 lines rework: instantiate VX_cp_core alongside Vortex
hw/rtl/afu/xrt/VX_afu_ctrl.sv  ~80 lines  extend AXI-Lite register decode for CP
VX_types.toml                  +1 block   reserve [dcr_cp] range 0x080–0x0BF
VX_config.toml                 +1 block   add [cp] knobs (parent §11)
```

## 3. Package and interfaces

### 3.1 `VX_cp_pkg.sv`

```systemverilog
package VX_cp_pkg;

  // ---------- Parameters mirrored from VX_config.toml ----------
  localparam int VX_CP_NUM_QUEUES      = `VX_CP_NUM_QUEUES;       // default 4
  localparam int VX_CP_RING_SIZE_LOG2  = `VX_CP_RING_SIZE_LOG2;   // default 16 (64 KiB)
  localparam int VX_CP_MAX_CMDS_PER_CL = `VX_CP_MAX_CMDS_PER_CL;  // default 5
  localparam int VX_CP_AXI_TID_WIDTH   = `VX_CP_AXI_TID_WIDTH;    // default 6
  localparam int CL_BYTES              = 64;
  localparam int CL_BITS               = CL_BYTES * 8;

  // ---------- Opcode encoding (parent §6.5) ----------
  typedef enum logic [7:0] {
    CMD_NOP          = 8'h00,
    CMD_MEM_WRITE    = 8'h01,
    CMD_MEM_READ     = 8'h02,
    CMD_MEM_COPY     = 8'h03,
    CMD_DCR_WRITE    = 8'h04,
    CMD_DCR_READ     = 8'h05,
    CMD_LAUNCH       = 8'h06,
    CMD_FENCE        = 8'h07,
    CMD_EVENT_SIGNAL = 8'h08,
    CMD_EVENT_WAIT   = 8'h09
  } cp_opcode_e;

  // ---------- Header flags (parent §6.5) ----------
  localparam int F_PROFILE   = 0;
  localparam int F_FENCE_PRE = 1;

  typedef struct packed {
    logic [7:0]  opcode;       // cp_opcode_e
    logic [7:0]  flags;
    logic [15:0] reserved;
  } cmd_header_t;

  // ---------- Decoded command record (output of unpacker) ----------
  typedef struct packed {
    cmd_header_t hdr;
    logic [63:0] arg0;
    logic [63:0] arg1;
    logic [63:0] arg2;
    logic [63:0] profile_slot;  // present iff hdr.flags[F_PROFILE]
  } cmd_t;

  // ---------- EVENT_WAIT comparison ops (in arg2[1:0]) ----------
  typedef enum logic [1:0] {
    WAIT_OP_EQ = 2'd0,
    WAIT_OP_GE = 2'd1,
    WAIT_OP_GT = 2'd2,
    WAIT_OP_NE = 2'd3
  } wait_op_e;

  // ---------- Per-CPE state (parent §6.3) ----------
  typedef struct packed {
    logic [63:0]                       ring_base;      // host IO addr
    logic [VX_CP_RING_SIZE_LOG2:0]     ring_size_mask; // size_bytes - 1
    logic [63:0]                       head_addr;
    logic [63:0]                       cmpl_addr;
    logic [63:0]                       tail;
    logic [63:0]                       head;
    logic [63:0]                       seqnum;
    logic [1:0]                        priority;
    logic                              enabled;
    logic                              profile_en;
  } cpe_state_t;

  // ---------- Resource-bid record (CPE → arbiter) ----------
  typedef enum logic [1:0] {
    RES_KMU = 2'd0,
    RES_DMA = 2'd1,
    RES_DCR = 2'd2
  } cp_resource_e;

  typedef struct packed {
    logic        valid;
    logic [1:0]  priority;
    cmd_t        cmd;
  } cpe_bid_t;

endpackage : VX_cp_pkg
```

### 3.2 `VX_cp_if.sv`

```systemverilog
// AXI4 master bundle for the CP (one per CP block, multiplexed by VX_cp_axi_xbar)
interface VX_cp_axi_m_if #(parameter ADDR_W=64, DATA_W=512, TID_W=6) ();
  // Write address
  logic              awvalid; logic awready;
  logic [ADDR_W-1:0] awaddr;  logic [TID_W-1:0] awid;
  logic [7:0]        awlen;   logic [2:0]       awsize; logic [1:0] awburst;
  // Write data
  logic              wvalid;  logic wready;
  logic [DATA_W-1:0] wdata;   logic [DATA_W/8-1:0] wstrb; logic wlast;
  // Write response
  logic              bvalid;  logic bready;
  logic [TID_W-1:0]  bid;     logic [1:0] bresp;
  // Read address
  logic              arvalid; logic arready;
  logic [ADDR_W-1:0] araddr;  logic [TID_W-1:0] arid;
  logic [7:0]        arlen;   logic [2:0]       arsize; logic [1:0] arburst;
  // Read data
  logic              rvalid;  logic rready;
  logic [DATA_W-1:0] rdata;   logic [TID_W-1:0] rid;
  logic              rlast;   logic [1:0]       rresp;

  modport master (output awvalid, awaddr, awid, awlen, awsize, awburst,
                          wvalid, wdata, wstrb, wlast, bready,
                          arvalid, araddr, arid, arlen, arsize, arburst, rready,
                  input  awready, wready, bvalid, bid, bresp,
                          arready, rvalid, rdata, rid, rlast, rresp);
endinterface

// AXI4-Lite slave bundle for the CP's host-facing control surface
interface VX_cp_axil_s_if ();
  // Write
  logic        awvalid; logic awready;
  logic [11:0] awaddr;
  logic        wvalid;  logic wready;
  logic [31:0] wdata;   logic [3:0] wstrb;
  logic        bvalid;  logic bready; logic [1:0] bresp;
  // Read
  logic        arvalid; logic arready;
  logic [11:0] araddr;
  logic        rvalid;  logic rready;  logic [31:0] rdata; logic [1:0] rresp;
endinterface

// CP → Vortex GPU bundle
interface VX_cp_gpu_if;
  // DCR request (CP master)
  logic                         dcr_req_valid;
  logic                         dcr_req_rw;
  logic [`VX_DCR_ADDR_WIDTH-1:0] dcr_req_addr;
  logic [`VX_DCR_DATA_WIDTH-1:0] dcr_req_data;
  logic                         dcr_req_ready;

  // DCR response (Vortex master)  — NEW in this proposal (§10)
  logic                         dcr_rsp_valid;
  logic [`VX_DCR_DATA_WIDTH-1:0] dcr_rsp_data;

  // KMU launch handshake
  logic                         start;
  logic                         busy;
endinterface

// CPE → resource arbiter (instantiated once per CPE per resource)
interface VX_cp_engine_bid_if;
  logic                         valid;
  VX_cp_pkg::cmd_t              cmd;
  logic [1:0]                   priority;
  logic                         grant;
endinterface
```

## 4. `VX_cp_core.sv`

Top-level wrapper. Instantiates the parameterized number of CPEs,
the three resource arbiters, the shared helpers, and the AXI xbar.

```systemverilog
module VX_cp_core
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = VX_CP_NUM_QUEUES
)(
  input  wire             clk,
  input  wire             reset,

  // Platform-facing interfaces
  VX_cp_axi_m_if.master   axi_m,        // for fetch/DMA/event/cmpl/profile writebacks
  VX_cp_axil_s_if         axil_s,       // host-side control + doorbells

  // GPU-facing
  VX_cp_gpu_if            gpu_if,

  // Vortex memory port (when CP_DMA_DEV_PORT == DEDICATED)
  // omitted when SHARED — DMA traffic goes through axi_m instead
  output wire             interrupt     // tied to 0 in v1 (phase 6 enables)
);
  // Per-CPE state and bidding
  cpe_state_t                       q_state    [NUM_QUEUES];
  VX_cp_engine_bid_if                     bid_kmu    [NUM_QUEUES] ();
  VX_cp_engine_bid_if                     bid_dma    [NUM_QUEUES] ();
  VX_cp_engine_bid_if                     bid_dcr    [NUM_QUEUES] ();

  // AXI sub-master sources (one per requester, fanned in by xbar)
  VX_cp_axi_m_if #(.TID_W(VX_CP_AXI_TID_WIDTH))  axi_cpe_fetch [NUM_QUEUES] ();
  VX_cp_axi_m_if #(.TID_W(VX_CP_AXI_TID_WIDTH))  axi_dma      ();
  VX_cp_axi_m_if #(.TID_W(VX_CP_AXI_TID_WIDTH))  axi_event    ();
  VX_cp_axi_m_if #(.TID_W(VX_CP_AXI_TID_WIDTH))  axi_cmpl     ();
  VX_cp_axi_m_if #(.TID_W(VX_CP_AXI_TID_WIDTH))  axi_prof     ();

  // 1) Per-queue CPEs
  genvar i;
  generate for (i = 0; i < NUM_QUEUES; ++i) begin : g_cpe
    VX_cp_engine #(.QID(i)) u_cpe (
      .clk, .reset,
      .state_o     (q_state[i]),
      .axil_s      (axil_s),         // each CPE decodes its own register block
      .axi_fetch   (axi_cpe_fetch[i].master),
      .bid_kmu     (bid_kmu[i]),
      .bid_dma     (bid_dma[i]),
      .bid_dcr     (bid_dcr[i])
    );
  end endgenerate

  // 2) Resource arbiters (round-robin)
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_kmu (.clk, .reset, .bid(bid_kmu));
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_dma (.clk, .reset, .bid(bid_dma));
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_dcr (.clk, .reset, .bid(bid_dcr));

  // 3) Shared resources
  VX_cp_launch       u_launch    (.clk, .reset, .bid(bid_kmu), .gpu_if);
  VX_cp_dma          u_dma       (.clk, .reset, .bid(bid_dma), .axi(axi_dma.master));
  VX_cp_dcr_proxy    u_dcr_proxy (.clk, .reset, .bid(bid_dcr), .gpu_if, .axi(axi_event.master));

  // 4) Helpers
  VX_cp_event_unit   u_evt   (.clk, .reset, /* bid + axi */);
  VX_cp_completion   u_cmpl  (.clk, .reset, .q_state, /* retire pulses */, .axi(axi_cmpl.master));
  VX_cp_profiling    u_prof  (.clk, .reset, /* sample pulses */, .axi(axi_prof.master));

  // 5) AXI master xbar — fan N+M sources into one master
  VX_cp_axi_xbar #(.N_FETCH(NUM_QUEUES), .N_HELPERS(4)) u_xbar (
    .clk, .reset,
    .in_fetch(axi_cpe_fetch),
    .in_dma(axi_dma), .in_event(axi_event),
    .in_cmpl(axi_cmpl), .in_prof(axi_prof),
    .out(axi_m)
  );

  // 6) AXI-Lite register decode (parent §6.10)
  //    Handles CP_CTRL, CP_STATUS, CP_DEV_CAPS_*, CP_CYCLE_*, plus
  //    per-queue Q_RING_BASE / HEAD_ADDR / CMPL_ADDR / RING_SIZE_LOG2 /
  //    Q_CONTROL / Q_TAIL doorbells / Q_SEQNUM read / Q_ERROR.
  //    Doorbell writes update q_state[qid].tail.
  //    See cp_axil_regfile.sv (instantiated here; not a separate top file).

  assign interrupt = 1'b0;   // v1.1 wires this up

endmodule : VX_cp_core
```

## 5. `VX_cp_engine.sv` — per-queue Command Processor Engine

The core per-queue state machine. There are `NUM_QUEUES` of these.

### 5.1 Ports

```systemverilog
module VX_cp_engine
  import VX_cp_pkg::*;
#(parameter int QID = 0)
(
  input  wire                  clk,
  input  wire                  reset,
  output cpe_state_t           state_o,           // for top to expose via AXI-Lite RO regs
  VX_cp_axil_s_if              axil_s,            // per-queue register block decoded here
  VX_cp_axi_m_if.master        axi_fetch,         // dedicated fetch master (merged by xbar)
  VX_cp_engine_bid_if.bidder         bid_kmu,
  VX_cp_engine_bid_if.bidder         bid_dma,
  VX_cp_engine_bid_if.bidder         bid_dcr
);
```

### 5.2 FSM

```
                    ┌───────────┐
                    │   IDLE    │◄────────────────────────────────────────┐
                    └────┬──────┘                                         │
            (tail != head, enabled)                                       │
                         ▼                                                │
                    ┌───────────┐                                         │
                    │ FETCH_REQ │  issue AXI ar for next CL               │
                    └────┬──────┘                                         │
                         ▼                                                │
                    ┌───────────┐                                         │
                    │ FETCH_RSP │  wait for rvalid; latch 64 B            │
                    └────┬──────┘                                         │
                         ▼                                                │
                    ┌───────────┐                                         │
                    │  UNPACK   │  combinational: VX_cp_unpack            │
                    └────┬──────┘                                         │
                         ▼                                                │
                    ┌───────────┐  per command i ∈ [0, n_cmds):           │
                    │  DECODE   │ ─┬─► CMD_NOP        : retire            │
                    └────┬──────┘  ├─► CMD_FENCE      : wait drain ─►retire│
                         │         ├─► CMD_LAUNCH     : bid KMU            │
                         │         ├─► CMD_DCR_*      : bid DCR            │
                         │         ├─► CMD_MEM_*      : bid DMA            │
                         │         ├─► CMD_EVENT_WAIT : bid EVENT          │
                         │         └─► CMD_EVENT_SIGNAL: enqueue to cmpl   │
                         ▼                                                 │
                    ┌───────────┐                                          │
                    │ WAIT_GRANT│  hold bid asserted until granted         │
                    └────┬──────┘                                          │
                         ▼                                                 │
                    ┌───────────┐                                          │
                    │  COMMIT   │  fire retire pulse to VX_cp_completion   │
                    └────┬──────┘  (also fires SUBMIT/START/END pulses     │
                         │          to VX_cp_profiling if F_PROFILE)       │
                         ▼                                                 │
                    (more cmds in this CL?) ── yes ──► DECODE ─────────────┘
                         │                                                 │
                         no                                                │
                         ▼                                                 │
                  advance head by CL_BYTES; goto IDLE                      │
```

### 5.3 Key state

```systemverilog
typedef enum logic [3:0] {
  S_IDLE, S_FETCH_REQ, S_FETCH_RSP, S_UNPACK, S_DECODE,
  S_WAIT_GRANT, S_COMMIT, S_FENCE_WAIT, S_EVENT_WAIT
} cpe_fsm_e;

cpe_fsm_e                                fsm;
cpe_state_t                              state;
logic [CL_BITS-1:0]                      cl_buf;
cmd_t                                    cl_cmds [VX_CP_MAX_CMDS_PER_CL];
logic [$clog2(VX_CP_MAX_CMDS_PER_CL)-1:0] cl_n_cmds;
logic [$clog2(VX_CP_MAX_CMDS_PER_CL)-1:0] cl_idx;
cp_resource_e                            pending_res;
logic                                    waiting_on_event;
logic [63:0]                             event_addr_r;
logic [63:0]                             event_value_r;
wait_op_e                                event_op_r;
```

### 5.4 Bid-and-hold semantics

A CPE bids by asserting `bid.valid` with its decoded `cmd`. The
arbiter grants by asserting `bid.grant`. The CPE then waits for the
*resource* to signal completion (e.g. KMU's `busy` falling, DMA's
`done` pulse, DCR proxy's `ack`). KMU bid is held for the entire
launch duration; DMA and DCR bids are released as soon as the
resource accepts the command.

`S_EVENT_WAIT` is special — the CPE issues an AXI read to the event
slot through `VX_cp_event_unit`, blocks until the comparison
succeeds, then retires the `CMD_EVENT_WAIT` and returns to `DECODE`
for the next command in the current line.

### 5.5 Profiling hooks

When `cl_cmds[cl_idx].hdr.flags[F_PROFILE]` is set, the CPE fires
three single-cycle pulses to `VX_cp_profiling`:

- `submit_evt` at entry to `S_DECODE` for this command.
- `start_evt` at the grant edge in `S_WAIT_GRANT`.
- `end_evt` at entry to `S_COMMIT`.

Each pulse carries `cl_cmds[cl_idx].profile_slot` so profiling can
issue the 32 B writeback to the right host address.

## 6. `VX_cp_fetch.sv`

Per-CPE AXI read of the next 64 B cache line at
`state.ring_base + (state.head & state.ring_size_mask)`. Issues one
outstanding request; pipelining is a phase-5 optimization.

```systemverilog
module VX_cp_fetch (
  input  wire           clk, reset,
  input  wire           req_valid,
  input  wire [63:0]    req_addr,
  output logic          req_ready,
  output logic          rsp_valid,
  output logic [511:0]  rsp_data,
  VX_cp_axi_m_if.master axi
);
```

Internal state is a 2-state FSM (IDLE → AR_WAIT → R_WAIT → IDLE)
plus a tag (the CPE's QID, encoded in `arid[VX_CP_AXI_TID_WIDTH-1:0]`)
used by the xbar to route the response back.

## 7. `VX_cp_unpack.sv`

Same as the prototype's `cacheline_cmd_unpacker` but extended for the
new opcodes and the `F_PROFILE` `profile_slot` field. Pure
combinational walk of the 64 B line, sizing each command from
`cmd_size_bytes(opcode, flags[F_PROFILE])`:

| Opcode             | Base bytes | +profile_slot (F_PROFILE) | Total |
|--------------------|-----------|--------------------------|-------|
| `CMD_NOP`          | 4         | n/a                      | 4     |
| `CMD_LAUNCH`       | 12        | +8                       | 12/20 |
| `CMD_FENCE`        | 8         | +8                       | 8/16  |
| `CMD_DCR_WRITE`    | 20        | +8                       | 20/28 |
| `CMD_DCR_READ`     | 20        | +8                       | 20/28 |
| `CMD_EVENT_SIGNAL` | 20        | +8                       | 20/28 |
| `CMD_EVENT_WAIT`   | 28        | +8                       | 28/36 |
| `CMD_MEM_WRITE`    | 28        | +8                       | 28/36 |
| `CMD_MEM_READ`     | 28        | +8                       | 28/36 |
| `CMD_MEM_COPY`     | 28        | +8                       | 28/36 |

Stops emitting when `offset + next_cmd_size > CL_BYTES` or when the
next header is `CMD_NOP` (treated as padding). Outputs `cmd_count` ∈
`[0, VX_CP_MAX_CMDS_PER_CL]`.

Synthesis note: this unpacker is combinational with up to 5 nested
size-based offsets, so its critical path can be long. If timing
closure fails on this module, split it into a 2-cycle pipelined
version (decode first 3 cmds in cycle 0, next 2 in cycle 1).

## 8. `VX_cp_arbiter.sv` — generic round-robin

```systemverilog
module VX_cp_arbiter
  import VX_cp_pkg::*;
#(parameter int N = 4)
(
  input  wire           clk, reset,
  VX_cp_engine_bid_if.arbiter bid [N]            // valid in, grant out
);
  logic [$clog2(N)-1:0] last_grant;
  // Combinational: scan bidders starting at (last_grant+1) % N;
  // first valid bidder gets the grant. Priority field can promote
  // a bidder by one slot when VX_CP_PRIORITY_ENABLE is set.
  // On grant fire, update last_grant.
endmodule
```

Instantiated three times in `VX_cp_core` (KMU, DMA, DCR). Priority
support is a compile-time flag; v1 default is plain round-robin per
parent §6.4.

## 9. `VX_cp_launch.sv`

Tiny wrapper over `gpu_if.start` / `gpu_if.busy`:

- On grant from KMU arbiter, pulse `gpu_if.start` for 1 cycle.
- Hold KMU arbiter grant until `gpu_if.busy` falls low (drained).
- Fire `start_evt` / `end_evt` pulses to profiling.

```systemverilog
module VX_cp_launch (
  input  wire        clk, reset,
  VX_cp_engine_bid_if.arbiter bid [VX_CP_NUM_QUEUES],
  VX_cp_gpu_if       gpu_if
);
```

## 10. `VX_cp_dma.sv`

Generic DMA engine. Source and destination each addressable as
either host (AXI master) or device (Vortex memory port). The
`CP_DMA_DEV_PORT_MODE` build-time parameter selects whether device
accesses borrow a dedicated Vortex memory port or share the AXI
fabric (parent §6.6).

**v1 default: `SHARED`** (per parent §6.6 resolution). The DMA engine
issues device-side accesses through the same AXI master that handles
host-memory traffic; the AFU's existing AXI fabric arbitrates between
CP DMA and Vortex memory traffic. Works on every XRT shell, no
shell-dependent surprises. `DEDICATED` is opt-in via
`--cp-dma-port=dedicated` for multi-bank shells where contention
measurably hurts; phase 5 perf decides whether to promote it.

In `DEDICATED` mode, the DMA engine connects to a separate Vortex
memory port via the `dev_mem` interface (commented out below);
`VX_cp_core` instantiates the connection only when the build mode is
`DEDICATED`.

Internally:

- Read source in `MAX_BURST` bursts; tag with `cmd_id`.
- Forward read data into a small streaming FIFO.
- Write to destination as data arrives, draining the FIFO.
- Done when last burst's write response returns.
- Single command in flight at a time (v1); pipelining is phase-5.

```systemverilog
module VX_cp_dma (
  input  wire              clk, reset,
  VX_cp_engine_bid_if.arbiter    bid [VX_CP_NUM_QUEUES],
  VX_cp_axi_m_if.master    axi,
  // device memory port (only when DEDICATED mode):
  // VX_mem_bus_if.master  dev_mem
  output logic             done
);
```

## 11. `VX_cp_dcr_proxy.sv`

Drives Vortex's DCR request port and captures DCR responses (the
top-level wire added in §13). For `CMD_DCR_WRITE`, fires `dcr_req`
with `rw=1` and acks immediately. For `CMD_DCR_READ`, fires with
`rw=0`, captures `dcr_rsp_data` when it arrives, and pushes a
writeback request to `axi` so the value lands at the user-supplied
host address.

State machine: IDLE → REQ → WAIT_RSP → WRITEBACK → IDLE. One
outstanding DCR transaction at a time (DCR bus is not pipelined in
Vortex).

## 12. `VX_cp_event_unit.sv`

Implements `CMD_EVENT_WAIT`. Logic:

1. Receive `event_addr`, `expected_value`, `op` from a CPE.
2. AXI-read 8 B from `event_addr` (or hit the local LRU cache of
   recent reads).
3. Compare `read_value` to `expected_value` under `op`:
   - `EQ`:   match if equal
   - `GE`:   match if `read >= expected` (common case)
   - `GT`:   match if `read >  expected`
   - `NE`:   match if not equal
4. On match, signal the CPE; on miss, re-read after a backoff
   counter (default 256 cycles, parametric).

```systemverilog
module VX_cp_event_unit
  import VX_cp_pkg::*;
#(parameter int CACHE_ENTRIES = 4)
(
  input  wire                 clk, reset,
  // per-CPE request port (bundled)
  input  wire                 req_valid [VX_CP_NUM_QUEUES],
  input  wire [63:0]          req_addr  [VX_CP_NUM_QUEUES],
  input  wire [63:0]          req_value [VX_CP_NUM_QUEUES],
  input  wait_op_e            req_op    [VX_CP_NUM_QUEUES],
  output logic                rsp_match [VX_CP_NUM_QUEUES],
  // AXI master for the slot reads
  VX_cp_axi_m_if.master       axi
);
```

A small LRU cache reduces AXI traffic when many CPEs spin on the
same completion slot. Cache lines are invalidated when an
`EVENT_SIGNAL` writes a matching address (snooping the completion
writes through `VX_cp_completion`).

## 13. `VX_cp_completion.sv`

Triggered by per-CPE retire pulses. For each retired command:

1. Increment that CPE's `seqnum` (skipped for `CMD_NOP`).
2. Issue an AXI write of the new seqnum to `q_state[qid].cmpl_addr`.
3. Issue an AXI write of the updated `q_state[qid].head` to
   `q_state[qid].head_addr` so the host can reclaim ring-buffer
   space.

Both writes can be coalesced when several retirements happen
back-to-back on the same queue: only the *last* seqnum and head
values for a queue need to be visible, so the unit collapses
in-flight updates and only issues new AXI writes when no
acknowledgment is pending or the value has actually changed.

(v1.1) Also pulses `interrupt` when a queue retires a command whose
`F_INTERRUPT` flag is set — placeholder hook, not implemented in v1.

## 14. `VX_cp_profiling.sv`

```systemverilog
module VX_cp_profiling (
  input  wire                  clk, reset,
  // free-running cycle counter, exposed via CP_CYCLE_LO/HI (RO AXI-Lite regs)
  output logic [63:0]          cp_cycle,
  // per-event samples
  input  wire                  submit_evt [VX_CP_NUM_QUEUES],
  input  wire                  start_evt  [VX_CP_NUM_QUEUES],
  input  wire                  end_evt    [VX_CP_NUM_QUEUES],
  input  wire [63:0]           slot_addr  [VX_CP_NUM_QUEUES],
  // AXI master for the 32 B writebacks
  VX_cp_axi_m_if.master        axi
);
  // Counter
  always_ff @(posedge clk) cp_cycle <= reset ? 64'd0 : cp_cycle + 64'd1;

  // Per-CPE small FIFO of {slot_addr, submit_ts, start_ts, end_ts}.
  // On end_evt, pop FIFO entry, write 32 B record to slot_addr via axi.
  // Read host-supplied QUEUED ns is left to runtime; CP writes 0 there.
endmodule
```

## 15. `VX_cp_axi_xbar.sv`

Multiplexes the N+4 internal AXI requesters into the single
upstream master:

| Requester              | Read | Write | Notes                                        |
|------------------------|------|-------|----------------------------------------------|
| Per-CPE fetch (N)      | ✓    |       | One outstanding read per CPE.                |
| `VX_cp_dma`            | ✓    | ✓     | DMA engine.                                  |
| `VX_cp_event_unit`     | ✓    |       | Slot reads.                                  |
| `VX_cp_completion`     |      | ✓     | Seqnum + head writes.                        |
| `VX_cp_profiling`      |      | ✓     | 32 B records.                                |

Strategy:

- Independent read and write arbiters, both round-robin.
- Each requester gets a distinct tag prefix in `arid`/`awid`; the
  xbar de-multiplexes responses by tag prefix. Tag-width budget:
  `ceil(log2(N+5))` bits of prefix + the remaining bits free for
  the requester to encode its own transaction id. With the default
  `VX_CP_AXI_TID_WIDTH=6` and `NUM_QUEUES=4`, prefix is 4 bits, 2
  bits free per requester (sufficient for one outstanding per
  requester in v1; phase-5 pipelining may need to bump the width).
- W-channel arbitration follows AW grant (Xilinx-style); no
  interleaving in v1.

## 16. `Vortex.sv` / `Vortex_axi.sv` DCR req/rsp extension

Vortex's internal `VX_dcr_bus_if` already carries both req and rsp.
Today's top-level only exposes the req side. Add to `Vortex.sv`'s
port list:

```systemverilog
  // DCR read response — NEW
  output wire                          dcr_rsp_valid,
  output wire [VX_DCR_DATA_WIDTH-1:0]  dcr_rsp_data,
```

Wire to the existing internal:

```systemverilog
  assign dcr_rsp_valid = dcr_bus_if.rsp_valid;
  assign dcr_rsp_data  = dcr_bus_if.rsp_data;
```

Same change in `Vortex_axi.sv`. This is a **non-breaking** change:
existing consumers (legacy XRT AFU) can simply ignore the new
outputs.

## 17. `VX_afu_wrap.sv` (XRT) integration

The XRT AFU wrapper is reworked to instantiate the CP alongside
Vortex. Conceptually:

```
                ┌─────── VX_afu_wrap.sv ───────┐
   AXI4-Lite ─►│  axi-lite register decode    │── existing legacy
   (kernel)    │   (legacy + new CP map)      │   AP_CTRL/DEV_CAPS/...
               │                              │
               │   ┌─────────────────────┐    │── CP doorbells +
               │   │   VX_cp_core         │◄───┤   queue config regs
               │   │   (rtl/cp/)         │    │
               │   │                     │    │
               │   │   axi_m  axi_l   gpu│    │
               │   └──┬───────┬─────────┬┘    │
               │      │       │         │     │
               │      │       │         ▼     │
               │      │       │     ┌───────┐ │
               │      │       │     │Vortex │ │── existing AXI master(s)
               │      │       └────►│  (.sv)│ │   to HBM/DDR banks
               │      ▼             │       │ │
               │   AXI-mux ────────►│       │ │
               │   (host+CP)        └───────┘ │
               └──────────────────────────────┘
```

Changes:

1. Instantiate `VX_cp_core` with `axi_m` connected to the kernel's
   host-AXI4 master and `axil_s` connected to the kernel's
   AXI4-Lite slave (de-muxed by an address range so legacy AP_CTRL
   registers stay at their current offsets and CP registers occupy
   `0x100..0x3FF`).
2. Wire `gpu_if.dcr_req_*` and `gpu_if.dcr_rsp_*` to Vortex's DCR
   bus.
3. Wire `gpu_if.start` and `gpu_if.busy` to Vortex's `start` and
   `busy` ports.
4. **Per-queue `Q_TAIL` doorbell** is committed atomically via the
   high-half write (parent §6.10 resolution): the AXI-Lite slave
   inside `VX_cp_core` decodes `+0x20` (Q_TAIL_LO) as a *staging*
   register that latches the host's value into a per-queue
   `tail_lo_staging[QID]` register without advancing the queue, and
   decodes `+0x24` (Q_TAIL_HI) as both a staging write to
   `tail_hi_staging[QID]` *and* a 1-cycle `tail_commit_pulse[QID]`.
   On `tail_commit_pulse`, the CPE's `tail` register atomically
   loads `{tail_hi_staging, tail_lo_staging}`. A host that writes
   only Q_TAIL_LO does not advance the queue; partial writes are
   inert. The implementation is a small always_ff block in the CP's
   AXI-Lite register decode block (see §4 / §15) — no protocol
   dependence on AXI-Lite interconnect ordering.
5. **Compatibility mode**: keep the legacy AP_CTRL FSM intact so
   that callers using `vortex.h` continue to drive single-launch
   semantics. When AP_CTRL `ap_start` fires, the legacy FSM holds
   `start` independently of the CP (mutually exclusive: legacy mode
   is engaged only when no queue is enabled). This compat mode is
   removed in phase 8.

## 18. DCR address allocations

Per parent §6.12, reserve `0x080..0x0BF` in `VX_types.toml` for
CP-internal DCRs. v1 does not actually use any of these — the
reservation is forward-compatibility for future CP↔GPU coordination
(e.g. in-flight kernel barriers when multi-context KMU lands).

```toml
[dcr_cp]
VX_DCR_CP_BEGIN   = 0x080
VX_DCR_CP_END     = 0x0BF    # inclusive sentinel
```

Verify no overlap with the existing `[dcr_kmu]` (0x010-0x01F),
`[dcr_tex]` (0x020-0x03F), `[dcr_raster]` (0x040-0x045),
`[dcr_om]` (0x060-0x071), `[dcr_dxa]` (0x100-0x27F) blocks.

## 19. Verification strategy

### 19.1 Per-module unit testbenches

Each module under `hw/rtl/cp/` gets a peer testbench in
`hw/unittest/cp/`:

```
hw/unittest/cp/
├── tb_VX_cp_unpack.sv          parameterized random CLs; check cmd_count and decoded fields
├── tb_VX_cp_arbiter.sv         random valid patterns; verify round-robin fairness
├── tb_VX_cp_fetch.sv           AXI BFM as slave; verify single outstanding
├── tb_VX_cp_dma.sv             AXI BFM both ends; verify byte-accurate copy
├── tb_VX_cp_event_unit.sv      script slot values; verify match latency and op semantics
├── tb_VX_cp_completion.sv      retire pulses; verify seqnum + head writeback ordering
├── tb_VX_cp_profiling.sv       inject submit/start/end; verify 32 B record content
├── tb_VX_cp_dcr_proxy.sv       mock DCR bus; verify req/rsp ordering + writeback
├── tb_VX_cp_engine.sv                full CPE FSM exercise; pre-loaded ring image
└── tb_VX_cp_core.sv             integration: 2 CPEs + 1 launch + 1 DCR; smoke flow
```

Framework: Verilator + SV testbench wrappers, integrated into the
existing `hw/unittest/Makefile` test-harness pattern. Each TB
includes a self-check (`assert` on golden output) and is run under
the project's standard 120 s timeout
([feedback-test-timeout-120s]).

### 19.2 Lint

`verilator --lint-only -Wall -Wno-fatal` over the entire `rtl/cp/`
tree. CI fails on any new warning. Run as a github action via the
self-hosted runner ([project-ci-machine]).

### 19.3 Integration tests

Hardware-in-the-loop on the XRT FPGA:

- Phase-2 smoke: `tests/kernel/vecadd` ported to `vortex2.h` runs
  end-to-end through the CP.
- Phase-3 stress: 4-queue concurrent enqueue with cross-queue
  events; assert no deadlock under 10 k iterations.
- Phase-4 conformance: POCL backend (when ready) exercises the
  OpenCL 1.2 conformance subset.

### 19.4 Coverage targets (v1.1)

- Functional coverage on FSM transitions in `VX_cp_engine` (every
  state×opcode combination hit).
- Cross coverage: KMU arbiter wins × source CPE (every CPE wins KMU
  at least once).
- Branch coverage in `VX_cp_unpack` for the size table.

## 20. Phased implementation tasks

Aligned with parent migration plan (§13).

### Phase 1 — DCR req/rsp extension (1 PR, ~3 days)

- [ ] Add `dcr_rsp_valid` / `dcr_rsp_data` outputs to `Vortex.sv`
      and `Vortex_axi.sv` (§16).
- [ ] Forward through `VX_afu_wrap.sv` to the AXI-Lite DCR-rsp
      register (replaces the prototype's software shadow).
- [ ] No CP yet; verifies the DCR-rsp wire change in isolation.
- [ ] Existing legacy tests must still pass unchanged.

### Phase 2 — single-CPE CP skeleton (3 PRs, ~3 weeks)

- [ ] `VX_cp_pkg.sv` complete.
- [ ] `VX_cp_if.sv` complete.
- [ ] `VX_cp_core.sv` with `NUM_QUEUES=1` and only `CMD_LAUNCH`,
      `CMD_DCR_WRITE`, `CMD_MEM_*` opcodes implemented.
- [ ] `VX_cp_engine.sv` FSM minus `EVENT_*` and `FENCE` support.
- [ ] `VX_cp_fetch`, `VX_cp_unpack`, single-bidder `VX_cp_arbiter`,
      `VX_cp_launch`, `VX_cp_dma`, `VX_cp_dcr_proxy`,
      `VX_cp_completion` (seqnum-only, no head writeback),
      `VX_cp_axi_xbar`.
- [ ] AFU shim rework to instantiate `VX_cp_core` alongside Vortex,
      with legacy AP_CTRL kept as compat mode.
- [ ] Unit TBs for `unpack`, `fetch`, `arbiter`, `dma`,
      `completion`, `cpe`.
- [ ] Hardware smoke test: vecadd via `vortex2.h` queue passes.

### Phase 3 — N CPEs + arbiters + full completion (2 PRs, ~2 weeks)

- [ ] Lift to `NUM_QUEUES=4`.
- [ ] Three resource arbiters with round-robin.
- [ ] Full `VX_cp_completion` (seqnum + head writeback,
      coalescing).
- [ ] Per-queue AXI-Lite register block.
- [ ] Doorbell update logic in `VX_cp_engine` (latches new tail on Q_TAIL
      hi-half write).
- [ ] Integration test: 4-queue cross-queue overlap on hardware.

### Phase 4 — events + barriers + profiling + DCR read (3 PRs, ~3 weeks)

- [ ] `VX_cp_engine` FSM gains `EVENT_WAIT` and `FENCE` states.
- [ ] `CMD_EVENT_SIGNAL` retire path through `VX_cp_completion`.
- [ ] `VX_cp_event_unit` with cache + AXI slot reads.
- [ ] `VX_cp_dcr_proxy` extended for `CMD_DCR_READ` writeback.
- [ ] `VX_cp_profiling` with cycle counter, sample points, 32 B
      writeback.
- [ ] Header flag decoding (`F_PROFILE`, `F_FENCE_PRE`) in unpacker
      and CPE.
- [ ] Hardware test: 3-queue DAG with cross-queue events on
      hardware passes 10 k iterations without hang.

### Phase 5 — perf pass (1-2 PRs, timing-driven)

- [ ] Pipelined `VX_cp_unpack` if critical-path closure fails.
- [ ] Pipelined `VX_cp_dma` (multiple outstanding bursts).
- [ ] Intra-CPE pipelining (DMA-while-launch on same queue).
- [ ] AXI tag-width bump if needed.
- [ ] Driven by post-phase-4 perf measurements on hardware.

## 21. Open implementation questions

1. ~~**DMA dedicated vs shared port default.**~~ **Resolved**: v1
   default = `SHARED` (parent §6.6, this proposal §10). `DEDICATED`
   opt-in via `--cp-dma-port=dedicated`; phase 5 measurements decide
   whether to promote on multi-bank shells.
2. **`VX_cp_unpack` critical path.** May need pipelining (§7).
   Decide based on phase-2 timing reports.
3. **Event-unit cache size.** `CACHE_ENTRIES=4` (one per CPE) is
   the default. If multiple CPEs commonly spin on the same external
   event (e.g. host-signaled fan-out), a larger shared cache helps.
   Decide based on phase-4 stress test traces.
4. **Single clock vs CP/GPU split.** v1 assumes one clock for the
   whole CP+Vortex+AFU domain. If timing forces a CDC between CP
   and Vortex (FPGA shell PLLs often do), add an `async_fifo` on
   the DCR bus and on the start/busy handshake. Decide based on
   place-and-route reports.
5. ~~**AXI-Lite write atomicity for 64 B `Q_TAIL`.**~~ **Resolved**:
   the high-half write (Q_TAIL_HI at +0x24) fires an explicit
   1-cycle commit pulse that atomically latches
   `{tail_hi_staging, tail_lo_staging}` into the CPE's `tail`
   register. Q_TAIL_LO (+0x20) only stages; no dependency on
   AXI-Lite interconnect ordering. See parent §6.10 and §17 of this
   proposal.
6. **Coverage tooling.** Verilator's coverage support is limited;
   consider adding QuestaSim or Xcelium integration for the
   coverage targets in §19.4. Out of scope for v1 but worth
   tracking.

## 22. References

- [docs/proposals/command_processor_proposal.md](command_processor_proposal.md)
  — parent architecture proposal; this document implements §6, §7.1, §9, §10 from there.
- [cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md)
  — companion runtime implementation proposal.
- [hw/rtl/VX_kmu.sv](../../hw/rtl/VX_kmu.sv)
  — KMU module the CP drives via DCR + start/busy.
- [hw/rtl/Vortex.sv](../../hw/rtl/Vortex.sv)
  — GPU top; §16 extends DCR bus to req/rsp.
- [hw/rtl/Vortex_axi.sv](../../hw/rtl/Vortex_axi.sv)
  — XRT-targeted Vortex wrapper; same DCR change.
- [hw/rtl/afu/xrt/VX_afu_wrap.sv](../../hw/rtl/afu/xrt/VX_afu_wrap.sv)
  — XRT AFU shim; §17 reworks for CP integration.
- [VX_types.toml](../../VX_types.toml)
  — DCR address map; §18 reserves `[dcr_cp]` range 0x080-0x0BF.
- [VX_config.toml](../../VX_config.toml)
  — per parent §11, gains the `[cp]` knobs (`VX_CP_NUM_QUEUES`,
  `VX_CP_RING_SIZE_LOG2`, `VX_CP_AXI_TID_WIDTH`,
  `VX_CP_DMA_DEV_PORT`, `VX_CP_PROFILE_DEFAULT`).
