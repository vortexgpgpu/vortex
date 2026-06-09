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

// ============================================================================
// cmd_processor.h — functional C++ model of the Command Processor.
// Shared by simx and rtlsim; presents the same cp_mmio_* MMIO surface to
// the runtime.
//
// Implements a `tick()`-per-cycle state machine that reads commands from a
// host-pinned ring in DRAM, dispatches them to the right resource (DCR proxy,
// launch, DMA), and publishes the retired sequence number back to a
// host-pinned completion slot.
//
// Address map (CP-internal, 0-based):
//   Globals (CP-internal offsets 0x000..0x0FF)
//     0x000 CP_CTRL       bit0=enable_global, bit1=reset_all
//     0x004 CP_STATUS     bit0=busy, bit1=error
//     0x008 CP_DEV_CAPS   {AXI_TID_W:8 | RING_LOG2:8 | NUM_QUEUES:8}
//     0x010 CP_CYCLE_LO
//     0x014 CP_CYCLE_HI
//     0x018 GPU_DEV_CAPS_LO  device-config caps, low 32 bits   (RO)
//     0x01C GPU_DEV_CAPS_HI  device-config caps, high 32 bits  (RO)
//     0x020 GPU_ISA_CAPS_LO  ISA caps, low 32 bits             (RO)
//     0x024 GPU_ISA_CAPS_HI  ISA caps, high 32 bits            (RO)
//   Per queue 0 (CP-internal offsets 0x100..0x13F)
//     0x100/04 Q_RING_BASE_LO/HI
//     0x108/0C Q_HEAD_ADDR_LO/HI   (where the CP publishes head)
//     0x110/14 Q_CMPL_ADDR_LO/HI   (where the CP publishes seqnum)
//     0x118    Q_RING_SIZE_LOG2
//     0x11C    Q_CONTROL          bit0=enable, bit1=reset
//     0x120    Q_TAIL_LO          (staging)
//     0x124    Q_TAIL_HI          (atomic commit)
//     0x128    Q_SEQNUM           (RO mirror)
//     0x12C    Q_ERROR
//     0x130    Q_LAST_DCR_RSP     (RO — latest CMD_DCR_READ response)
// ============================================================================

#ifndef VORTEX_COMMAND_PROCESSOR_H
#define VORTEX_COMMAND_PROCESSOR_H

#include <cstdint>
#include <functional>
#include <array>
#include <vm_types.h>   // SATP_t / PTE_t / vAddr_t (VM page-table walk)

namespace vortex {

class CommandProcessor {
public:
    struct Hooks {
        // Read `bytes` bytes from device DRAM at `addr` into `dst`.
        // Used for ring-buffer fetches (one cache line at a time).
        std::function<void(uint64_t addr, void* dst, std::size_t bytes)> dram_read;

        // Write `bytes` bytes from `src` into device DRAM at `addr`.
        // Used for completion-slot writebacks (8 B seqnum).
        std::function<void(uint64_t addr, const void* src, std::size_t bytes)> dram_write;

        // Issue a single DCR write to Vortex (for CMD_DCR_WRITE).
        std::function<void(uint32_t addr, uint32_t value)> vortex_dcr_write;

        // Issue a single DCR read to Vortex (for CMD_DCR_READ). `tag` is
        // placed on the DCR data bus and addresses things like per-core
        // CACHE_FLUSH. The backend must block until the response is
        // available before returning.
        std::function<uint32_t(uint32_t addr, uint32_t tag)> vortex_dcr_read;

        // Pulse Vortex's start signal (for CMD_LAUNCH). The launch FSM
        // calls this once when transitioning into the "started" state.
        std::function<void()> vortex_start;

        // Query Vortex's busy state. The launch FSM waits for this to
        // rise (kernel actually executing) then fall (kernel done)
        // before retiring the CMD_LAUNCH.
        std::function<bool()> vortex_busy;
    };

    explicit CommandProcessor(const Hooks& hooks);

    // ----- Host-facing MMIO surface -----
    // Offsets match VX_cp_axil_regfile (CP-internal, 0-based).
    // Backends doing MMIO at byte offset 0x1000+ should subtract 0x1000
    // on their side before calling these.
    void     mmio_write(uint32_t off, uint32_t value);
    uint32_t mmio_read (uint32_t off) const;

    // ----- Sim integration -----
    // Advance the CP one functional cycle. Called by the simulator's
    // per-cycle loop. Cheap: a small FSM step (single-digit branches).
    void tick();

    // True iff CP_CTRL.enable_global && Q_CONTROL.enable. The simulator
    // can use this to skip tick() when the host hasn't enabled the CP.
    bool enabled() const;

    // True iff the engine has commands in flight OR ring has pending
    // entries. Lets the host's wait loop break early when the CP is idle.
    bool busy() const;

private:
    // Engine FSM states.
    enum class EngState { Idle, Decode, Bid, WaitDone, Retire };

    // KMU launch sub-FSM.
    enum class LaunchState { Idle, PulseStart, WaitBusy, WaitDrain };

    // Command opcodes: low 8 bits of the on-wire command header.
    enum : uint8_t {
        OP_NOP        = 0x00,
        OP_MEM_WRITE  = 0x01,
        OP_MEM_READ   = 0x02,
        OP_MEM_COPY   = 0x03,
        OP_DCR_WRITE  = 0x04,
        OP_DCR_READ   = 0x05,
        OP_LAUNCH     = 0x06,
        OP_FENCE      = 0x07,
        OP_EVENT_SIG  = 0x08,
        OP_EVENT_WAIT = 0x09,
        OP_CACHE_FLUSH = 0x0A,
    };

    // CMD_MEM_* header flag (cmd_t.flags bit2 = F_MEM_PHYSICAL): the device
    // operand is a physical address — the MMU-aware CP DMA skips translation.
    // Used for page-table bootstrap writes and the PT region itself. A
    // dedicated bit, distinct from flags bit0 (F_PROFILE) — the addressing
    // mode is its own descriptor field (modern-GPU convention).
    static constexpr uint8_t MEM_FLAG_PHYSICAL = 0x04;

    // Decoded cmd record (matches cmd_t struct layout on-wire).
    struct Cmd {
        uint8_t  opcode;
        uint8_t  flags;
        uint16_t reserved;
        uint64_t arg0;
        uint64_t arg1;
        uint64_t arg2;
    };

    // ----- Per-queue programmable state (q_state_t mirror) -----
    struct Queue {
        uint64_t ring_base   = 0;
        uint64_t head_addr   = 0;
        uint64_t cmpl_addr   = 0;
        uint8_t  ring_log2   = 16;     // 64 KiB default
        uint32_t control     = 0;      // bit0=enable, bits3:2=prio
        uint64_t tail        = 0;
        uint32_t tail_lo_staging = 0;
        // CP-tracked state (not host-writable):
        uint64_t head        = 0;      // bytes consumed
        uint64_t seqnum      = 0;      // commands retired
        uint32_t error       = 0;
    };

    // ----- Globals -----
    uint32_t cp_ctrl_ = 0;           // bit0=enable_global
    uint64_t satp_ = 0;              // CP_SATP — page-table root for the CP DMA's MMU
    uint64_t cycle_counter_ = 0;
    Queue    q0_;                    // single-queue model
    Hooks    hooks_;
    uint32_t last_dcr_rsp_ = 0;     // Q_LAST_DCR_RSP slot (0x130)

    // ----- Engine/launch state machines -----
    EngState    eng_state_ = EngState::Idle;
    LaunchState launch_state_ = LaunchState::Idle;
    Cmd         cur_cmd_{};
    bool        cur_is_launch_ = false;
    bool        cur_is_no_resource_ = false;

    // ----- Fetch state -----
    // The simulator fetches one cache line at a time when head < tail,
    // then walks the CL extracting decoded cmds before fetching the next.
    static constexpr std::size_t CL_BYTES = 64;
    static constexpr int MAX_CMDS_PER_CL = 5;
    std::array<uint8_t, CL_BYTES> cl_buf_{};
    int  cl_cmd_count_ = 0;
    int  cl_cmd_slot_ = 0;
    bool cl_loaded_   = false;

    // Walk `cl_buf_` and populate `decoded_cmds_` / `cl_cmd_count_`.
    void unpack_cl();
    // Decode a single header at byte offset `off` into a Cmd record;
    // returns the size in bytes of the command (so caller can advance).
    int  decode_cmd(int off, Cmd& out);
    // Inverse of decoded helpers: write seqnum to cmpl_addr.
    void publish_completion();
    // CMD_EVENT_WAIT compare helper — reads cur_cmd_.arg0 from DRAM and
    // compares to cur_cmd_.arg1 under the wait_op encoded in arg2[1:0].
    bool event_wait_satisfied_();
    // Advance the launch FSM one step using cur_cmd_.
    void tick_launch();
    // Advance the engine FSM one step.
    void tick_engine();
    // Fetch one CL from ring into cl_buf_ if needed.
    void fetch_if_needed();
    // VM: translate a device virtual address to physical via a page-table
    // walk. A no-op when VM is disabled, SATP is unset/BARE, or physical.
    uint64_t cp_translate(uint64_t vaddr, bool physical) const;
};

} // namespace vortex

#endif // VORTEX_COMMAND_PROCESSOR_H
