# Command Processor Prototype — Review of `~/dev/vortex_cp`

## 1. Purpose of this document

The active `feature_cp` branch will introduce a *portable* command-processor
(CP) architecture for Vortex that works across OPAE, XRT, and future
back-ends. Before designing the new CP, we are reviewing an earlier student
prototype that added a deferred-rendering command buffer to Vortex on Intel
OPAE only. That prototype lives in `~/dev/vortex_cp` and is the subject of
this report.

The goals of this report are:

1. Describe how the prototype runtime + RTL implement deferred commands.
2. Document the hardware FSM, command format, ring-buffer protocol, and the
   software-side `CommandBuffer` class as they actually exist in that tree.
3. Call out the concrete limitations that the next-generation portable CP
   must address.

This report intentionally avoids prescribing the new design — that belongs
in a separate proposal under [docs/proposals/](../proposals/). Here we only
describe what exists today.

## 2. High-level model

In the stock Vortex runtime, every host-visible API call (`vx_copy_to_dev`,
`vx_copy_from_dev`, `vx_start`, `vx_dcr_write`, …) is a **lock-step MMIO
transaction**: the runtime drives a small command FSM in the AFU one
command at a time and polls `MMIO_STATUS` between commands. The AFU only
holds a single in-flight operation, the GPU sits idle while the host
walks through MMIO writes, and there is no way for the host to *queue
ahead*.

The prototype replaces that with a deferred model:

```
Host code           (record)               (submit)              (consume)
─────────────       ─────────────          ─────────────         ─────────────
vx_copy_to_dev ──┐                                              ┌─ DMA host→dev
vx_dcr_write   ──┤  push into pinned   ── MMIO doorbell ──►    ├─ DCR write to GPU
vx_dcr_write   ──┤  CommandBuffer in                            ├─ DCR write to GPU
vx_start       ──┤  host memory                                 ├─ DCR write to GPU
                 └─                                             └─ assert vx_reset, run, wait !busy
                                                              (CP FSM in AFU walks ring buffer)
vx_flush_commands ──── one MMIO write that arms the consumer ──┘
vx_ready_wait      ──── polls MMIO_STATUS for state == IDLE
```

Three things are new:

* A **pinned 1 MB host buffer** ("CommandBuffer") laid out as a sequence of
  64-byte cache lines, each line containing up to 5 packed commands.
* A **hardware ring-buffer consumer** in the AFU that DMAs cache lines from
  that buffer over CCI-P, unpacks them with a small parser, and feeds them
  into the existing per-command FSM.
* A new public entry point `vx_flush_commands()` plus a `CMD_DCR_WRITE`
  opcode so DCR programming (e.g. KMU startup-PC / argument-pointer
  registers) can be queued rather than executed inline.

The lock-step MMIO command path (`MMIO_CMD_TYPE` / `MMIO_CMD_ARG0..2`)
still exists in the RTL but is muxed behind the ring-buffer path and is
**not used by the prototype's runtime** — every API call goes through the
ring buffer.

## 3. Source layout

### Hardware (`~/dev/vortex_cp/hw/rtl/`)

```
afu/
├── opae/
│   ├── vortex_afu.sv              top-level AFU; CCI-P pipes, ring-buffer reader, mux, FSM glue
│   ├── vortex_afu.vh              AFU UUID + MMIO register-index defines (see §4.1)
│   ├── cmd_dispatch.sv            5-state FSM: IDLE → {MEM_READ, MEM_WRITE, DCR_WRITE, RUN}
│   ├── ccip_read_req.sv           CCI-P read-side controller (pending-tag table)
│   ├── ccip_write_req.sv          CCI-P write-side controller
│   ├── ccip_interface_reg.sv      pipeline-stage register for CCI-P signals
│   ├── local_mem_cfg_pkg.sv       Avalon local-memory parameters
│   └── ccip/ccip_if_pkg.sv        upstream CCI-P interface package
└── xrt/                            stub only — XRT AFU is NOT CP-enabled
```

The XRT AFU files in this tree (`VX_afu_wrap.sv`, `VX_afu_ctrl.sv`) are
the baseline lock-step XRT shell — none of the ring-buffer or
`cmd_dispatch` logic has been ported to them.

### Runtime (`~/dev/vortex_cp/runtime/`)

```
include/vortex.h          public C API; adds vx_flush_commands() and two test entry points
common/                   DeviceConfig (DCR shadow), MemoryAllocator, callbacks
opae/
├── driver.{h,cpp}        dynamic loader for libopae-c.so
└── vortex.cpp            CP-aware OPAE driver: CommandBuffer, StagingBuffer, enqueue_command()
xrt/vortex.cpp            stub; no CP support
rtlsim/, simx/, stub/     unchanged back-ends; no CP awareness
```

## 4. Hardware architecture

### 4.1 MMIO register map

From [hw/rtl/afu/opae/vortex_afu.vh](../../../vortex_cp/hw/rtl/afu/opae/vortex_afu.vh):

| Index | Byte offset | Name | Direction | Purpose |
|-------|-------------|------|-----------|---------|
| 10 | 0x28 | `MMIO_CMD_TYPE`           | W | Legacy MMIO command opcode (unused by CP runtime) |
| 12 | 0x30 | `MMIO_CMD_ARG0`           | W | Legacy MMIO arg0 |
| 14 | 0x38 | `MMIO_CMD_ARG1`           | W | Legacy MMIO arg1 |
| 16 | 0x40 | `MMIO_CMD_ARG2`           | W | Legacy MMIO arg2 |
| 18 | 0x48 | `MMIO_STATUS`             | R | `[7:0]` = FSM state, `[63:8]` = packed console-out stream |
| 20 | 0x50 | `MMIO_SCOPE_READ`         | R | logic-analyzer read |
| 22 | 0x58 | `MMIO_SCOPE_WRITE`        | W | logic-analyzer write |
| 24 | 0x60 | `MMIO_DEV_CAPS`           | R | device capability word |
| 26 | 0x68 | `MMIO_ISA_CAPS`           | R | ISA capability word |
| 28 | 0x70 | `MMIO_FLUSH`              | W | doorbell — `1` arms the ring-buffer consumer |
| 30 | 0x78 | `MMIO_HOST_RING_BUFFER_BASE_ADDR` | W | physical (IO-mapped) address of the pinned host buffer |
| 32 | 0x80 | `MMIO_RING_BUFFER_WPTR`   | W | declared write pointer (not currently consumed by HW — see §6) |
| 34 | 0x88 | `MMIO_RING_BUFFER_RPTR`   | R | read pointer (declared, not driven) |
| 36 | 0x90 | `MMIO_RING_BUFFER_NUM_CMD_REMAINING` | W | number of 64-byte cache lines the host has just made available |

The opcode encoding (also in `vortex_afu.vh`):

```verilog
`define AFU_IMAGE_CMD_MEM_READ   1
`define AFU_IMAGE_CMD_MEM_WRITE  2
`define AFU_IMAGE_CMD_RUN        3
`define AFU_IMAGE_CMD_DCR_WRITE  4
`define AFU_IMAGE_CMD_MAX_VALUE  4
```

### 4.2 Command word format

Each command in the ring buffer is a 4-byte header plus 0–3 8-byte
arguments. The packed `cmd_t` type defined in `cmd_pkg` inside
`vortex_afu.sv` is:

```systemverilog
typedef enum logic [31:0] {
    CMD_MEM_READ_e  = 1,
    CMD_MEM_WRITE_e = 2,
    CMD_RUN_e       = 3,
    CMD_DCR_WRITE_e = 4
} cmd_opcode_e;

typedef struct packed {
    cmd_opcode_e opcode;   // 4  bytes
    logic [63:0] arg0;     // 8
    logic [63:0] arg1;     // 8
    logic [63:0] arg2;     // 8
} cmd_t;                   // 28 bytes worst case
```

| Opcode          | Bytes | arg0                 | arg1                 | arg2            |
|-----------------|-------|----------------------|----------------------|-----------------|
| `CMD_MEM_READ`  | 28    | dst host addr (CL)   | src device addr (CL) | size (CL)       |
| `CMD_MEM_WRITE` | 28    | src host addr (CL)   | dst device addr (CL) | size (CL)       |
| `CMD_DCR_WRITE` | 20    | DCR address          | DCR value            | —               |
| `CMD_RUN`       | 12    | —                    | —                    | —               |

`CL` = 64-byte cache line. All host/device addresses are cache-line
indices; the AFU shifts by 6 internally.

### 4.3 Cache-line layout and the unpacker

The runtime treats every 64-byte cache line as a self-contained "frame"
that holds **up to 5 commands**. If a new command would cross a
cache-line boundary, the rest of the current line is zero-padded and the
next command starts at the next line. This is enforced both by
[`CommandBuffer::push_command`](../../../vortex_cp/runtime/opae/vortex.cpp)
on the host side and by the
[`cacheline_cmd_unpacker`](../../../vortex_cp/hw/rtl/afu/opae/vortex_afu.sv)
module on the FPGA side:

```systemverilog
module cacheline_cmd_unpacker #(
    parameter int CL_BYTES = 64,
    parameter int MAX_CMDS = 5
)(
    input  logic [CL_BYTES*8-1:0]            cl_data,
    output logic [$clog2(MAX_CMDS+1)-1:0]    cmd_count,
    output cmd_pkg::cmd_t                    cmds [MAX_CMDS]
);
```

It walks the line byte-wise, reads the next 4-byte header, sizes the
payload from `cmd_size_bytes(opcode)`, emits one `cmd_t`, and stops when
the next header would exceed `CL_BYTES` or when an unknown opcode is
seen (treated as end-of-line padding).

### 4.4 Ring-buffer consumer

State held in `vortex_afu.sv`:

```systemverilog
reg [63:0]                                host_ring_buffer_base_addr;
reg [MAX_RING_BUFFER_CMDS_WIDTH-1:0]      ring_buffer_num_cmds_remaining;
reg [MAX_RING_BUFFER_CMDS_WIDTH-1:0]      ring_buffer_num_cmds_consumed;
```

* `host_ring_buffer_base_addr` is loaded once at device init from
  `MMIO_HOST_RING_BUFFER_BASE_ADDR`.
* `ring_buffer_num_cmds_remaining` is set by the host every time it
  rings the `MMIO_FLUSH` doorbell, and is **decremented** by hardware as
  each cache line is fetched.
* `ring_buffer_num_cmds_consumed` is a monotonic counter the hardware
  uses to compute the next CCI-P read address:

```systemverilog
wire ring_buffer_has_data  = ring_buffer_num_cmds_remaining > 0;
wire [63:0] ring_buffer_byte_addr =
        host_ring_buffer_base_addr + (64'(ring_buffer_num_cmds_consumed) * 64'd64);
```

Cache-line responses are tagged with `mdata[15:8] = 8'hAB` so the AFU
can distinguish them from ordinary GPU memory traffic. A small SystemVerilog
FIFO (`VX_fifo_queue`, "kernel_fifo") buffers raw cache lines between
the CCI-P read pipeline and the unpacker, after which individual
`cmd_t` records are popped one-per-cycle and presented to the
`cmd_dispatch` FSM (§4.5).

The "all done" signal that re-arms the host wait loop is:

```systemverilog
wire all_done = !line_active
              & cmd_fifo_empty
              & (ring_buffer_num_cmds_remaining == 0)
              & (ring_buffer_num_cmds_consumed != 0)
              & flush;
```

i.e. the host's previously-declared batch has been fully fetched,
unpacked, and dispatched.

### 4.5 `cmd_dispatch` FSM

[hw/rtl/afu/opae/cmd_dispatch.sv](../../../vortex_cp/hw/rtl/afu/opae/cmd_dispatch.sv)
implements the per-command FSM:

| State           | Entry condition                | Exit condition                                                |
|-----------------|--------------------------------|--------------------------------------------------------------|
| `STATE_IDLE`    | reset, or previous state done  | sees a valid opcode in `cmd_type` from the mux               |
| `STATE_MEM_READ`| `cmd_type == CMD_MEM_READ`     | `cmd_mem_rd_done` from `ccip_read_req`                       |
| `STATE_MEM_WRITE`| `cmd_type == CMD_MEM_WRITE`   | `cmd_mem_wr_done` from `ccip_write_req`                      |
| `STATE_DCR_WRITE`| `cmd_type == CMD_DCR_WRITE`   | one cycle (combinational drive of `vx_dcr_wr_*`)             |
| `STATE_RUN`     | `cmd_type == CMD_RUN`          | reset hold (`RESET_DELAY` cycles) → wait `vx_busy==1` → wait `vx_busy==0` |

The state-encoded `output_state` value is exactly what the host reads
out of `MMIO_STATUS[7:0]`, so `state == 0` (IDLE) **and** `all_done`
together signal completion. There is no per-command completion fence
visible to the host.

`STATE_RUN` always reasserts `vx_reset` for `RESET_DELAY` cycles before
releasing the GPU. That means **every** `CMD_RUN` from the queue
performs a full reset; consecutive launches do not carry warp / cache /
register state. This is a deliberate consequence of the legacy lock-step
launch model that the CP did not re-architect.

### 4.6 Mux of ring-buffer vs. legacy MMIO command source

The AFU keeps the old MMIO command path alive but selects the
ring-buffer source whenever it has data:

```systemverilog
wire use_unpacked = line_active
                  & (unpack_cmd_count != 0)
                  & (num_cmds_finished_from_cl < unpack_cmd_count);

assign cmd_header   = use_unpacked ? unpack_cmds[num_cmds_finished_from_cl].opcode : ...;
assign fifo_cmd_args[0] = use_unpacked ? unpack_cmds[idx].arg0 : ...;
...
assign cmd_args = use_unpacked ? fifo_cmd_args : mmio_cmd_args;
```

A consequence: the legacy MMIO path is not a true fallback — it shares
the same downstream FSM and `vx_reset` logic. There is no compile-time
toggle to fully disable the CP and rebuild a stock Vortex AFU; the
prototype is a one-way change.

### 4.7 Vortex GPU integration

Vortex itself is instantiated essentially unchanged. The AFU drives:

```systemverilog
Vortex vortex (
    .clk(clk),
    .reset(vx_reset),               // driven by the FSM, asserted around every CMD_RUN
    .mem_req_*, .mem_rsp_*,         // unchanged
    .dcr_wr_valid (vx_dcr_wr_valid),// driven by STATE_DCR_WRITE
    .dcr_wr_addr  (vx_dcr_wr_addr),
    .dcr_wr_data  (vx_dcr_wr_data),
    .busy         (vx_busy)
);
```

There is **no DCR read response path** in this top-level wrapper —
`CMD_DCR_WRITE` is fire-and-forget, and the runtime keeps a software
shadow (see §5.4) for reads.

## 5. Runtime architecture

### 5.1 Public API surface

The CP-aware API from
[runtime/include/vortex.h](../../../vortex_cp/runtime/include/vortex.h)
adds one new public entry point and two test entry points:

```c
// COMMAND BUFFER: initial testing
int vx_send_ring_buffer_dummy(vx_device_h hdevice);
int vx_test_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr,
                        uint64_t dst_offset, uint64_t size);

int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr,
                   uint64_t dst_offset, uint64_t size);
int vx_flush_commands(vx_device_h hdevice);    // NEW
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer,
                     uint64_t src_offset, uint64_t size);

int vx_start(vx_device_h hdevice,
             vx_buffer_h hkernel, vx_buffer_h harguments);
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

int vx_dcr_read (vx_device_h hdevice, uint32_t addr, uint32_t* value);
int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value);
```

The signatures of the existing calls are **identical** to the stock
runtime — the change in semantics (deferred vs. blocking) is silent.
Callers must know to insert `vx_flush_commands()` followed by
`vx_ready_wait()` at the points where they actually need the work to
complete.

### 5.2 `CommandBuffer` — host-side record buffer

[runtime/opae/vortex.cpp:98-173](../../../vortex_cp/runtime/opae/vortex.cpp):

```cpp
class CommandBuffer {
public:
  struct CmdHeader { uint32_t cmd_type; };

  CommandBuffer(uint8_t* base, size_t capacity, size_t cache_block_size);

  bool push_command(uint32_t cmd_type, const void* payload, size_t payload_size) {
    CmdHeader hdr = { cmd_type };
    size_t total = sizeof(CmdHeader) + payload_size;

    // enforce "one command per cache block" rule
    if (curr_offset_ + total > cache_block_size_) {
      size_t pad = cache_block_size_ - curr_offset_;
      if (!write_bytes(nullptr, pad))   // zero pad to end of CL
        return false;
      curr_offset_ = 0;
    }
    if (!write_bytes(&hdr, sizeof(CmdHeader))) return false;
    if (!write_bytes(payload, payload_size)) return false;
    curr_offset_ += total;
    return true;
  }

  size_t   used_space() const { return size_; }
  uint8_t* data()             { return base_addr_; }

private:
  bool   write_bytes(const void* src, size_t len) {
    if (len > free_space()) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    for (size_t i = 0; i < len; ++i) {
      uint8_t v = p ? p[i] : 0;
      base_addr_[(tail_ + i) % capacity_] = v;
    }
    tail_ = (tail_ + len) % capacity_;
    size_ += len;
    return true;
  }
  size_t free_space() const { return capacity_ - size_; }

  uint8_t* base_addr_;
  size_t   capacity_;
  size_t   cache_block_size_;
  size_t   head_, tail_;
  size_t   curr_offset_;
  size_t   size_;
};
```

Two observations that matter for the next design:

1. The class **is named** "ring buffer" but in practice it is a
   one-shot linear buffer. `size_` only ever grows and `head_` is never
   advanced — `free_space()` returns `capacity_ - size_`. There is no
   API to release space after the hardware has consumed a region. Once
   the 1 MB buffer fills, `push_command()` returns `false` and the
   driver has no way to recover. (The wrap-around modulo arithmetic
   inside `write_bytes` therefore never actually wraps under normal
   use.)
2. The "one command per cache block" rule means a 12-byte `CMD_RUN`
   wastes the remaining 52 bytes if it is the last command pushed
   before a `vx_flush_commands()`. The host has no batching API to pack
   multiple commands explicitly — packing happens implicitly via the
   `curr_offset_` bookkeeping in `push_command`.

Allocation of the pinned buffer happens in `vx_device::init()`:

```cpp
static constexpr size_t CMD_BUFFER_CAPACITY = 1024 * 1024;   // 1 MB

api_.fpgaPrepareBuffer(fpga_, CMD_BUFFER_CAPACITY,
                       &cmd_buffer_ptr_, &cmd_buffer_wsid_, 0);
api_.fpgaGetIOAddress (fpga_, cmd_buffer_wsid_, &cmd_buffer_ioaddr_);
api_.fpgaWriteMMIO64  (fpga_, 0, MMIO_HOST_RING_BUFFER_BASE_ADDR,
                       cmd_buffer_ioaddr_);
cmd_buffer_ = CommandBuffer(reinterpret_cast<uint8_t*>(cmd_buffer_ptr_),
                            CMD_BUFFER_CAPACITY, CACHE_BLOCK_SIZE);
```

### 5.3 Per-transfer `StagingBuffer`s

```cpp
struct StagingBuffer {
  uint64_t wsid;        // OPAE workspace id
  uint64_t ioaddr;      // FPGA-visible IO address
  uint8_t* ptr;         // host VA
  uint64_t size;
};
std::vector<StagingBuffer> staging_buffers_;
```

`upload()` (a.k.a. `vx_copy_to_dev`) allocates a fresh OPAE-pinned
staging buffer for **every** transfer, `memcpy`s the user payload into
it, and enqueues a `CMD_MEM_WRITE` whose `arg0` is the staging buffer's
IO address. The driver remembers every staging buffer in
`staging_buffers_` and only releases them in `~vx_device()`.

The implication: a long-running session that streams many small
transfers leaks pinned-memory descriptors at OPAE level until the
device is closed.

### 5.4 Deferred call shapes

Each user-visible call becomes a record-then-return:

| API call           | Hardware commands enqueued        | Blocking step          |
|--------------------|-----------------------------------|------------------------|
| `vx_copy_to_dev`   | `CMD_MEM_WRITE`                   | none                   |
| `vx_dcr_write`     | `CMD_DCR_WRITE` + shadow update   | none                   |
| `vx_start`         | 4× `CMD_DCR_WRITE` (KMU PC / args) + `CMD_RUN` | none      |
| `vx_flush_commands`| —                                 | 2× MMIO writes (arm)   |
| `vx_copy_from_dev` | `CMD_MEM_READ`                    | calls `ready_wait()`   |
| `vx_ready_wait`    | —                                 | polls `MMIO_STATUS`    |
| `vx_dcr_read`      | —                                 | reads software shadow  |

`vx_dcr_read` is interesting: the prototype keeps a `DeviceConfig dcrs_`
mirror in the driver and `dcr_read()` returns from that mirror without
touching the FPGA. This works for kernel-launch parameters that the
host wrote itself, but cannot observe any value the GPU produced
(perf counters, status). The legacy MMIO `CMD_DCR_READ` path was not
re-introduced.

### 5.5 `vx_flush_commands` and the arming protocol

```cpp
int flush_commands() {
  size_t bytes_written = cmd_buffer_.used_space();
  uint64_t num_cls = (bytes_written % 64 > 0)
                    ? bytes_written/64 + 1
                    : bytes_written/64;
  api_.fpgaWriteMMIO64(fpga_, 0,
                       MMIO_RING_BUFFER_NUM_CMD_REMAINING, num_cls);
  api_.fpgaWriteMMIO64(fpga_, 0,
                       MMIO_FLUSH, 1);
  return 0;
}
```

Two MMIO writes — one publishes the number of cache lines to consume,
one rings the doorbell. Because `MMIO_RING_BUFFER_WPTR` is unused
hardware-side, the host re-uses `NUM_CMD_REMAINING` as the de facto
producer pointer.

`ready_wait()` polls `MMIO_STATUS` every ms, checks the low 8 bits for
`state == 0`, and along the way drains the GPU's `vx_printf` console
stream that is multiplexed into the upper bits of the same register.

### 5.6 Notable gap: kernel launch grid/block setup

`vx_start()` in the prototype only writes the four legacy startup DCRs
(`VX_DCR_BASE_STARTUP_ADDR0/1`, `VX_DCR_BASE_STARTUP_ARG0/1`) before
the `CMD_RUN`. The new KMU on `feature_cp` expects an additional
~11 DCRs (grid_dim, block_dim, lmem_size, warp_step, block_size — see
[VX_kmu.sv](../../hw/rtl/VX_kmu.sv) and the `[dcr_kmu]` section of
[VX_types.toml](../../VX_types.toml)). The prototype was written
against the pre-KMU lock-step launch model and would need extension
before it could drive the current GPU at all.

## 6. Known limitations

The items below are taken from in-tree `TODO`s, dead-code comments, and
behavioral analysis of the prototype.

### 6.1 Hardware

* **No ring-buffer wrap-around.** `vortex_afu.sv` line 1027 carries an
  explicit `TODO: figure out wrap-around if ring buffer size is
  limited`. `ring_buffer_num_cmds_consumed` is a monotonic counter; if
  the host ever submits enough cache lines to overflow its width, the
  address computation goes off the end of the pinned buffer.
* **No per-command completion event.** `cmd_done` in the AFU is wired
  to `is_kernel_finished` only; `STATE_DCR_WRITE` and `STATE_MEM_*`
  completions are inferred from the next-state transition rather than
  pulsed back. A `TODO: include RUN/DCR completion pulses` comment marks
  this. Consequence: the host cannot tell which command in a batch
  failed or even how far the AFU has gotten.
* **Hardcoded routing signals.** `switch_hardcode = 0` and similar
  notes (`TODO_: Find all instance of switch_hardcode and replace with
  actual switch controller`, `TODO_: Need a proper "start state and end
  state"`) indicate that several muxes were left tied off for the
  prototype and need to be promoted to real control logic.
* **Hard reset on every `CMD_RUN`.** Each launch reasserts `vx_reset`
  for `RESET_DELAY` cycles. The CP cannot dispatch back-to-back
  kernels without flushing the GPU pipeline.
* **No interrupt path.** The AFU never raises an interrupt; the host
  must spin on `MMIO_STATUS`. (The XRT baseline already exposes an
  `interrupt` pin that the new design should use.)
* **No CCI-P/Avalon decoupling.** The CP-side DMA modules
  (`ccip_read_req`, `ccip_write_req`) are written directly against
  CCI-P and `t_ccip_clAddr`; there is no abstraction layer that could
  be retargeted to AXI for XRT.
* **OPAE only.** The XRT AFU files in this tree do not contain any of
  the ring-buffer logic. Porting the prototype to XRT would mean
  rewriting `cmd_dispatch.sv` plus all of the CCI-P front-end against
  the AXI4 master / AXI4-Lite slave interfaces from
  `VX_afu_wrap.sv` / `VX_afu_ctrl.sv`.

### 6.2 Software

* **CommandBuffer is one-shot, not a ring.** `head_` is never advanced;
  once 1 MB has been pushed, `push_command()` returns false and the
  driver has no recovery path. Long sessions will eventually fail.
* **`MMIO_RING_BUFFER_WPTR` is dead.** A `// TODO: change from 1 to
  wptr` comment in `enqueue_command()` shows the intent was to update
  a hardware-visible write pointer per push, but the driver only ever
  writes the `NUM_CMD_REMAINING` counter at flush time. There is no
  producer/consumer cursor pair; everything is implicit in the doorbell.
* **Pinned-buffer leak per transfer.** Every `vx_copy_to_dev` /
  `vx_copy_from_dev` calls `fpgaPrepareBuffer` and stashes the result
  in `staging_buffers_`. The list is only walked at device close.
* **Blocking downloads.** `download()` enqueues `CMD_MEM_READ`, calls
  `ready_wait()`, then `memcpy`s out of the staging buffer. Uploads
  are deferred but downloads serialize the host on every read.
* **No fences / ordering primitives.** The host has to flush the
  entire queue and wait for `STATE_IDLE` to enforce ordering between
  any two operations. There is no `vx_event` / `vx_fence` /
  `vx_wait(handle)` API.
* **DCR shadow only.** `vx_dcr_read` cannot observe GPU-written DCR
  values; it only returns what the host previously wrote.
* **No error reporting back to host.** If a `CMD_DCR_WRITE` targets a
  bad address or a `CMD_MEM_*` overflows device memory, the AFU has no
  channel to report it. The host only sees a stuck `MMIO_STATUS` and
  a `ready_wait` timeout.
* **No bypass / lock-step fallback.** The legacy MMIO command path
  exists in RTL but the runtime never uses it, and there is no build
  flag to disable the CP entirely.
* **No test/example exercising the CP path.** The `tests/` tree
  contains kernel-side programs only. The two new test hooks
  (`vx_send_ring_buffer_dummy`, `vx_test_copy_to_dev`) are not wired
  into any harness, and no public test demonstrates the
  `record / flush / wait` pattern end-to-end.
* **No CP-aware KMU programming.** As noted in §5.6, the prototype
  predates the current KMU and only programs the four legacy startup
  DCRs.

## 7. Implications for the next design

The above is descriptive, not prescriptive — the portable-CP design
will be drafted separately under [docs/proposals/](../proposals/). For
that work, the key takeaways from this review are:

* The functional pattern (host pushes packed cache-line frames into
  pinned memory, hardware DMAs them, an in-AFU FSM dispatches them
  one at a time) is sound and worth keeping.
* The CCI-P/Avalon-specific code is the largest portability hazard.
  The new CP block should live under a new `hw/rtl/cp/` tree with a
  thin technology-specific DMA/PIO shim under `hw/rtl/afu/{opae,xrt}/`
  that only adapts read/write request channels to the platform.
* The CP must talk to the GPU via the **DCR bus into KMU**, not via
  the legacy startup-DCRs and `vx_reset`-on-launch path. Eliminating
  the reset-per-`CMD_RUN` is a prerequisite for true command-stream
  throughput.
* The host-side `CommandBuffer` needs to become a real ring (with a
  consumer-driven head pointer, possibly exposed via a hardware-written
  `RPTR` MMIO or via a memory write the host can poll), per-command
  completion events, and a fence primitive in the public API.
* The runtime API should grow explicit asynchronous semantics
  (`vx_event`, `vx_fence`, `vx_wait(event)`) rather than overloading the
  semantics of existing calls silently.
* DCR reads must round-trip through the GPU again so the host can
  observe GPU-written values (perf counters, status registers).
