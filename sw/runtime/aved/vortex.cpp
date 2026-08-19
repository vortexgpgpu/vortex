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
// V80 backend transport HAL. Exposes:
//   * device lifecycle       — init() / ~vx_device()
//   * CP register channel    — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation, DMA and capability decoding all live in the
// common core; the Command Processor is the sole memory engine. Host memory
// is the command ring + DMA staging; it must stay coherent with the device's
// view of it without an explicit sync.
// ============================================================================

#include <common.h>

#ifdef SCOPE
#include "scope.h"
#endif

#ifdef AVEDSIM
#include <vrt_c.h>
#else
#include <vrt/device.hpp>
#include <vrt/kernel.hpp>
// For detail::reserveFakePhysAddr: the simulation host-memory path has to draw
// its CP-visible addresses out of one of the linker's simulated memory windows,
// and VRT owns those bases.
#include <vrt/buffer.hpp>
#endif

#include <cstdlib>
#include <cstring>
#include <exception>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <util.h>

using namespace vortex;

#ifndef AVEDSIM
#define CPP_API
#endif

#define MMIO_CTL_ADDR 0x00
#define MMIO_SCP_ADDR 0x28

#define CTL_AP_IDLE  (1 << 2)
#define CTL_AP_RESET (1 << 4)

// ----- Command Processor regfile -----
// Host addresses 0x1000..0x1FFF reach the CP regfile, which sees them as its
// native 0x000-based 12-bit space. Callers pass the CP-internal offset;
// cp_reg_* add this base.
#define CP_BASE 0x1000

// CP-internal offsets this backend has to recognise for the simulation sync
// below. They are part of the regfile ABI callers already speak (see
// hw/rtl/cp/VX_cp_axil_regfile.sv); the backend only observes them, it never
// originates a transaction of its own.
#define CP_Q_RING_BASE_LO 0x100
#define CP_Q_HEAD_ADDR_LO 0x108
#define CP_Q_CMPL_ADDR_LO 0x110
#define CP_Q_CONTROL      0x11C
#define CP_Q_TAIL_LO      0x120
#define CP_Q_SEQNUM       0x128

// Free-running cycle counter, incremented every clock unconditionally by
// VX_cp_axil_regfile. Used by the transport gate in init(): it is the only
// register whose value is guaranteed to change, so it can distinguish "reads
// return real data" from "reads return a bus artifact".
#define CP_REG_CYCLE_LO   0x010

#define DEFAULT_DEVICE_BDF "0000:11:00"
#define DEFAULT_VBIN_PATH  "vortex_afu.vbin"
#define KERNEL_NAME        "vortex_afu_0"
// AXI master the CP uses for its command ring. config.cfg.tmpl binds it
// via @HOST_TAG@, so the vbin records whether it lands on the QDMA slave
// bridge (HOST) or a memory bank; staged_probe() reads that back.
#define HOST_PORT_NAME     "m_axi_host"

// Smallest allocation VRT's device allocator will serve: its
// MediumBlockSuperblock is BuddySuperblockBase<12, 21>, so 2^12 bytes.
// Only relevant when CP memory is staged in device memory.
#define STAGED_MIN_ALLOC   4096u

#ifdef CPP_API
typedef vrt::Device vrt_device_t;
typedef vrt::Kernel vrt_kernel_t;
#else
typedef vrtDeviceHandle vrt_device_t;
typedef vrtKernelHandle vrt_kernel_t;
#endif

#define CHECK_HANDLE(handle, _expr, _cleanup)                                  \
  auto handle = _expr;                                                         \
  if (handle == nullptr) {                                                     \
    printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr);                    \
    _cleanup                                                                   \
  }

// VRT throws on error; propagating across the extern "C" callbacks_t
// boundary is UB, so every VRT-touching member wraps in a try/catch and
// returns -1 on exception. Errors are logged once so a missing vbin or an
// AXI-Lite timeout produces a diagnostic, not a silent SEGV from
// libstdc++'s unhandled-exception terminate path.
#define VRT_TRY() try {
#define VRT_CATCH(_ret) }                                                      \
  catch (const std::exception& _e) {                                           \
    fprintf(stderr, "[VXDRV] VRT exception: %s\n", _e.what());                 \
    return _ret;                                                               \
  } catch (...) {                                                              \
    fprintf(stderr, "[VXDRV] VRT exception (unknown)\n");                      \
    return _ret;                                                               \
  }

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
#ifndef CPP_API
    : vrtDevice_(nullptr), vrtKernel_(nullptr)
#endif
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
  #ifndef CPP_API
    if (vrtKernel_) {
      vrtKernelClose(vrtKernel_);
    }
    if (vrtDevice_) {
      vrtDeviceClose(vrtDevice_);
    }
  #endif
  }

  int init() {
    // An empty value is treated as unset: the test harness exports these
    // unconditionally, so they arrive empty rather than absent.
    const char* bdf = getenv("VRT_DEVICE_BDF");
    if (bdf == nullptr || bdf[0] == '\0') {
      bdf = DEFAULT_DEVICE_BDF;
    }

    const char* vbin_path = getenv("VRT_VBIN_PATH");
    if (vbin_path == nullptr || vbin_path[0] == '\0') {
      vbin_path = DEFAULT_VBIN_PATH;
    }

  #ifdef CPP_API

    // Each vrt::Device open reprograms the PL, and a design write only
    // succeeds on a freshly reset device: vrtd runs its reset_with_ami
    // sequence only when the requested shell differs from the current one, so
    // once the shell reads "compute" no reset happens and the load fails with
    // "Input/output error", taking the AMC to NO_AMC and costing a JTAG
    // recovery. Measured 2026-08-19 (see ~/dev/v80/v80_oneshot.sh).
    //
    // That budgets one test per recovery cycle, which makes a regression ladder
    // impractical. When the design is already loaded, skip the reprogram and
    // reuse it: set VORTEX_AVED_NO_PROGRAM=1 for every run after the first.
    const char* noprog = getenv("VORTEX_AVED_NO_PROGRAM");
    const bool program = (noprog == nullptr || noprog[0] == '\0'
                          || noprog[0] == '0');
    VRT_TRY()
      vrtDevice_ = vrt::Device(bdf, vbin_path, program);
      // Only the hardware platform has a slave bridge, so anything else needs
      // the explicit host-memory sync in cp_reg_write/cp_reg_read.
      sim_mode_  = (vrtDevice_.getPlatform() != vrt::Platform::HARDWARE);
      vrtKernel_ = vrt::Kernel(vrtDevice_, KERNEL_NAME);
    VRT_CATCH(-1)

    // Must follow vrtKernel_: the connection map comes from the vbin's
    // system_map.xml, which the kernel handle parses.
    staged_probe();

  #else

    CHECK_HANDLE(vrtDevice, vrtDeviceOpen(bdf, vbin_path), {
      return -1;
    });

    CHECK_HANDLE(vrtKernel, vrtKernelOpen(vrtDevice, KERNEL_NAME), {
      vrtDeviceClose(vrtDevice);
      return -1;
    });

    vrtDevice_ = vrtDevice;
    vrtKernel_ = vrtKernel;

  #endif

    // Transport gate. This must come BEFORE the reset write, because the
    // reset handshake below cannot detect a dead bus: it breaks out on
    // `ctl & CTL_AP_IDLE`, and an AXI DECERR / PCIe completion timeout
    // substitutes 0xFFFFFFFF on the way back -- which has that bit set. The
    // poll therefore self-certifies against a bus that is answering nothing,
    // and the first real symptom appears much later and much further away
    // (historically: device.cpp decoding all-ones CP_DEV_CAPS as VM_ENABLED
    // and spinning on 65,536 PTEs, which presents as a hang, not a bus error).
    //
    // CP_CYCLE_LO free-runs -- VX_cp_axil_regfile increments it every clock
    // unconditionally -- so two reads that differ prove three things at once:
    // reads reach the register file, they return real data rather than a bus
    // artifact, and the AFU clock is live. All-ones or all-zeros means the
    // read never got there.
    {
      uint32_t c0 = 0, c1 = 0;
      CHECK_ERR(this->read_register(CP_BASE + CP_REG_CYCLE_LO, &c0), {
        return err;
      });
      CHECK_ERR(this->read_register(CP_BASE + CP_REG_CYCLE_LO, &c1), {
        return err;
      });
      if (c0 == 0xFFFFFFFFu || c1 == 0xFFFFFFFFu) {
        printf("[VXDRV] Error: AXI-Lite transport is not responding "
               "(CP_CYCLE_LO reads 0x%08x). The read did not reach the AFU: "
               "DECERR or PCIe completion timeout. Check that the vbin loaded "
               "and that the BAR window covers the AFU aperture.\n", c0);
        return -1;
      }
      // All-zeros is a real fault on silicon (clock gated, or reset held
      // asserted) but is a plausible answer from a model that does not
      // implement the counter, so only hardware treats it as fatal.
      bool on_hardware = true;
    #ifdef CPP_API
      on_hardware = !sim_mode_;
    #endif
      if (c0 == 0 && c1 == 0 && on_hardware) {
        printf("[VXDRV] Error: AXI-Lite responds but CP_CYCLE_LO is stuck at "
               "zero -- the AFU clock is gated or reset is held asserted.\n");
        return -1;
      }
      if (c0 == c1 && on_hardware) {
        printf("[VXDRV] Warning: CP_CYCLE_LO did not advance between two "
               "reads (0x%08x). The AFU clock may be stopped.\n", c0);
      }
    }

    CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_RESET), {
      return err;
    });

    // wait for the reset sequence to complete (ap_idle deasserts while the
    // device reset is in flight)
    {
      uint32_t ctl = 0;
      for (int retry = 0; retry < 1000; ++retry) {
        CHECK_ERR(this->read_register(MMIO_CTL_ADDR, &ctl), {
          return err;
        });
        // Reject the all-ones no-completion signature before testing any bit;
        // see the transport gate above for why bit-testing it is unsafe.
        if (ctl != 0xFFFFFFFFu && (ctl & CTL_AP_IDLE)) {
          break;
        }
      }
      if (ctl == 0xFFFFFFFFu) {
        printf("[VXDRV] Error: control register reads all-ones during reset; "
               "the AXI-Lite path died after the transport gate passed.\n");
        return -1;
      }
      if ((ctl & CTL_AP_IDLE) == 0) {
        printf("[VXDRV] Error: device reset timeout!\n");
        return -1;
      }
    }

  #ifdef SCOPE
    {
      scope_callback_t callback;
      callback.registerWrite = [](vx_device_h hdevice, uint64_t value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo = (uint32_t)(value);
        uint32_t value_hi = (uint32_t)(value >> 32);
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR, value_lo), {
          return err;
        });
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR + 4, value_hi), {
          return err;
        });
        return 0;
      };
      callback.registerRead = [](vx_device_h hdevice, uint64_t *value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo, value_hi;
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR, &value_lo), {
          return err;
        });
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR + 4, &value_hi), {
          return err;
        });
        *value = (((uint64_t)value_hi) << 32) | value_lo;
        return 0;
      };
      CHECK_ERR(vx_scope_start(&callback, this, -1, -1), {
        return err;
      });
    }
  #endif

    return 0;
  }

  // ----- CP register channel -----
  // Under the VRT simulation platform the CP's view of "host" memory is a
  // model living in the xsim process, so it agrees with ours only where we
  // make it. The CP protocol already supplies the two moments that matter and
  // both pass through here: the doorbell (Q_TAIL_LO) is the last write before
  // the CP reads the ring, and Q_SEQNUM is polled after it has written its
  // results back. Syncing on those keeps this confined to the backend, so the
  // common core keeps the coherent-memory contract it documents.
  //
  // All of this is CPP_API-only. `sim_mode_` and the whole sim_* apparatus are
  // declared under `#ifdef CPP_API`, so referencing them unguarded here broke
  // the TARGET=avedsim build outright ("'sim_mode_' was not declared in this
  // scope"). Under avedsim there is nothing to sync: the Verilator model runs
  // in this process and shares our memory directly, so the correct behaviour
  // is exactly the `sim_mode_ == false` path.
  int cp_reg_write(uint32_t off, uint32_t value) {
  #ifdef CPP_API
    if (sim_mode_ && !staged_cfg_) {
      sim_note_region_addr(off, value);
      // Seed on the first doorbell rather than at queue enable. The harness
      // only advances the clock once the design has been started, and a
      // populate issued before that blocks its ZMQ worker mid-request, which
      // wedges every command behind it. The CP cannot have fetched anything
      // before the first doorbell, so this is the earliest safe point -- and
      // the seed must include the head and completion lines, since simulated
      // memory starts undefined and a CP that reads X there would compute a
      // bogus pending count and run away on garbage descriptors.
      if (off == CP_Q_TAIL_LO) {
        sim_trace("doorbell", value, sim_ring_tail_, 0);
        const bool seed = !sim_seeded_;
        sim_seeded_ = true;
        if (sim_publish(seed, value) != 0) {
          return -1;
        }
      }
    }
    // Same two moments, same reason, different transport: when the ring lives
    // in device memory it is no more coherent with us than the simulator's
    // model is. Seed on the first doorbell so the head and completion lines
    // are defined before the CP can read them -- a CP that reads garbage there
    // computes a bogus pending count and runs away on junk descriptors.
    if (staged_cfg_) {
      staged_note_region_addr(off, value);
    }
    if (staged_cfg_ && off == CP_Q_TAIL_LO) {
      const bool seed = !staged_seeded_;
      staged_seeded_ = true;
      if (staged_publish(seed) != 0) {
        return -1;
      }
    }
  #endif
    return this->write_register(CP_BASE + off, value);
  }

  int cp_reg_read(uint32_t off, uint32_t *value) {
    CHECK_ERR(this->read_register(CP_BASE + off, value), { return err; });
  #ifdef CPP_API
    if (sim_mode_ && !staged_cfg_ && off == CP_Q_SEQNUM) {
      // Refresh only when the CP actually advanced. This register is read in
      // a spin loop, and a fetch per iteration would dominate the run.
      if (!sim_seqnum_valid_ || *value != sim_last_seqnum_) {
        sim_last_seqnum_  = *value;
        sim_seqnum_valid_ = true;
        return sim_refresh();
      }
    }
    if (staged_cfg_ && off == CP_Q_SEQNUM) {
      // Refresh only when the CP actually advanced. This register is read in a
      // spin loop and a QDMA sync per iteration would dominate the run.
      if (!staged_seqnum_valid_ || *value != staged_last_seqnum_) {
        staged_last_seqnum_  = *value;
        staged_seqnum_valid_ = true;
        return staged_refresh();
      }
    }
  #endif
    return 0;
  }

  // ----- CP-visible host memory (command ring + DMA staging) -----
  // On hardware the device masters into this region by bus address, so the
  // allocation must come from the driver rather than malloc. Under the
  // in-process model the two views coincide and plain memory suffices.
  int host_mem_alloc(uint64_t size, void **host_ptr, uint64_t *cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
  #ifdef CPP_API
    if (sim_mode_ && !staged_cfg_) {
      // There is no slave bridge to allocate from, so the CP reaches this
      // region at an address out of one of the linker's simulated memory
      // windows. Take the DDR window (0x600_0000_0000), not the HBM one: HBM
      // is where m_axi_mem_0 lands once PLATFORM_MEMORY_OFFSET rebases device
      // memory there, and the two allocators know nothing about each other --
      // VRT bump-allocates up from the base while Vortex's device allocator
      // starts at base + VX_MEM_USER_BASE_ADDR, so a DMA staging buffer over
      // 64 KB would run straight into the application's buffers. The DDR
      // window is backed by a separate BRAM model and is mapped into
      // m_axi_host's address space, which also matches the hardware topology
      // where the two masters target different slaves.
      void *ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
      if (ptr == nullptr) {
        return -1;
      }
      std::memset(ptr, 0, asize);
      uint64_t addr = vrt::detail::reserveFakePhysAddr(asize,
                                                       vrt::MemoryRangeType::DDR);
      {
        std::lock_guard<std::mutex> g(sim_mu_);
        sim_regions_.emplace(addr, sim_region_t{ptr, asize});
      }
      sim_trace("alloc", addr, 0, asize);
      *host_ptr = ptr;
      *cp_addr  = addr;
      return 0;
    }
    if (staged_cfg_) {
      // m_axi_host reaches a memory bank, so the CP addresses these bytes by
      // their device address and we hold the host-side shadow. getPhysAddr()
      // is absolute, which is what the CP needs: VX_afu_wrap.sv deliberately
      // does NOT apply PLATFORM_MEMORY_OFFSET to the host port ("that offset
      // is device-memory specific"), so no rebasing belongs here.
      // VRT's device allocator refuses anything below its smallest buddy
      // block: MediumBlockSuperblock is BuddySuperblockBase<12, 21>, so 2^12.
      // The head and completion regions are one cacheline each (CP_CL_BYTES),
      // which lands two orders of magnitude under that and throws
      //   "Size too small for MediumBlockSuperblock"
      // out of vx_device_open. Round up rather than special-case those two:
      // the CP only ever touches the leading bytes, the rest is slack, and a
      // DMA staging buffer of any size then allocates by the same rule.
      const uint64_t dsize = std::max<uint64_t>(asize, STAGED_MIN_ALLOC);
      VRT_TRY()
        auto buf = std::make_shared<vrt::Buffer<uint8_t>>(
            vrtDevice_, dsize, *staged_cfg_);
        void* hp = buf->get();
        const uint64_t dev = buf->getPhysAddr();
        std::memset(hp, 0, asize);
        {
          std::lock_guard<std::mutex> g(staged_mu_);
          staged_regions_.emplace(dev, staged_region_t{buf, asize});
        }
        *host_ptr = hp;
        *cp_addr  = dev;
      VRT_CATCH(-1)
      return 0;
    }
    VRT_TRY()
      auto buffer = vrtDevice_.allocHostBuffer(asize);
      {
        std::lock_guard<std::mutex> g(host_bufs_mu_);
        host_bufs_.emplace(buffer.dmaAddress, buffer);
      }
      *host_ptr = buffer.address;
      *cp_addr  = buffer.dmaAddress;
    VRT_CATCH(-1)
  #else
    void *ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
    if (ptr == nullptr) {
      return -1;
    }
    *host_ptr = ptr;
    *cp_addr  = reinterpret_cast<uint64_t>(ptr);
  #endif
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
  #ifdef CPP_API
    if (sim_mode_ && !staged_cfg_) {
      std::lock_guard<std::mutex> g(sim_mu_);
      auto sit = sim_regions_.find(cp_addr);
      if (sit == sim_regions_.end()) {
        return -1;
      }
      // Publish before dropping the region. DMA staging is filled, submitted
      // and freed back-to-back, and inside a batch the submit only appends to
      // the ring -- the doorbell comes later, from cp_batch_end, when this
      // region no longer exists. Waiting for the doorbell would therefore lose
      // the bytes entirely and the CP would DMA whatever the model happened to
      // hold. Publishing here is correct in both modes: the CP cannot have
      // read the region yet, because the doorbell has not been rung.
      sim_trace("free", cp_addr, 0, sit->second.size);
      const int rc = sim_xfer(cp_addr, sit->second.host_ptr, 0,
                              sit->second.size, true);
      free(sit->second.host_ptr);
      sim_regions_.erase(sit);
      return rc;
    }
    if (staged_cfg_) {
      // Publish before dropping the region, for the same reason the sim path
      // does: DMA staging is filled, submitted and freed back-to-back, and
      // inside a batch the submit only appends to the ring -- the doorbell
      // comes later, from cp_batch_end, when this region is gone. Waiting for
      // the doorbell would lose the bytes entirely.
      //
      // But unlike the sim path, publishing is NOT enough to make the free
      // safe. There, sim_xfer copies the bytes *into the model*, which keeps
      // them after the host free(). Here the bytes are pushed into a
      // vrt::Buffer whose destructor hands the device memory straight back to
      // VRT's buddy allocator -- while a descriptor in the ring still names
      // that device address. Inside a batch the CP has not read it yet, so it
      // then DMAs from a block the allocator has already re-served to the next
      // same-sized staging buffer. The non-batch path is safe only by
      // accident: cp_submit_cl_ rings the doorbell and polls to completion
      // before host_free is reached, which is why `minimal -l` passes and
      // anything that launches a kernel would not.
      //
      // So: publish now, but keep the buffer alive until the CP has provably
      // moved past the commands that reference it. staged_refresh() runs on a
      // Q_SEQNUM advance -- i.e. after the CP has retired work -- and releases
      // the deferred set there.
      std::lock_guard<std::mutex> g(staged_mu_);
      auto sit = staged_regions_.find(cp_addr);
      if (sit == staged_regions_.end()) {
        return -1;
      }
      const int rc = staged_xfer(sit->second, true);
      staged_pending_free_.push_back(std::move(sit->second));
      staged_regions_.erase(sit);
      return rc;
    }
    std::lock_guard<std::mutex> g(host_bufs_mu_);
    auto it = host_bufs_.find(cp_addr);
    if (it == host_bufs_.end()) {
      return -1;
    }
    if (it->second.address != nullptr) {
      munmap(it->second.address, it->second.length);
    }
    close(it->second.fd);
    host_bufs_.erase(it);
  #else
    free(reinterpret_cast<void *>(cp_addr));
  #endif
    return 0;
  }

private:

#ifdef CPP_API
  // ----- Staged CP memory (m_axi_host wired to device memory) -----
  //
  // WHY THIS EXISTS
  // ---------------
  // The CP's command ring, head/completion cachelines and DMA staging buffers
  // all live in memory the CP masters into over m_axi_host. When that port is
  // tagged HOST it reaches host DRAM through the QDMA slave bridge, which is
  // what allocHostBuffer serves and what the code below is bypassed for.
  //
  // That path does not work on this V80 compute shell. Measured 2026-08-19
  // with a ten-line HLS kernel whose only distinguishing feature is
  // sp=<kernel>.m_axi_gmem0:HOST (~/dev/v80/hostprobe): its AXI reads never
  // complete. AP_CTRL sat at 0x1 (ap_start, not idle, not done) for ~15
  // minutes with the argument registers verified correct (src=0xfea9a000,
  // size=0x100) -- a 256-iteration II=1 loop that should retire in about a
  // microsecond. The byte-identical control build with the single line changed
  // to :HBM0 (~/dev/v80/hostprobe_hbm) completed in under 0.1 s and returned
  // the right sum, which isolates the fault to the HOST path rather than to
  // mastering, the AFU, or the build flow.
  //
  // Neither simulator can catch this: avedsim shares process memory outright
  // and sim copies host memory into the model via the sim_* path above, so
  // both bypass the very thing that fails.
  //
  // So the ring is staged in device memory instead. The bytes then need the
  // same explicit publish/refresh the simulation path needs, for the same
  // reason -- device memory is not coherent with us -- and at exactly the same
  // two moments: the doorbell before the CP reads, and a seqnum change after
  // it has written back.
  //
  // Self-configuring rather than a build flag: portMemoryConfig() reads the
  // connection map out of the vbin's system_map.xml and throws when the port
  // has no memory target. A vbin built with HOST_TAG=HOST therefore keeps the
  // slave-bridge path untouched, and one built with HOST_TAG=HBM1 stages,
  // with no way for the two to disagree.
  struct staged_region_t {
    std::shared_ptr<vrt::Buffer<uint8_t>> buf;
    uint64_t size;
  };

  // Resolve once, at init. Throwing means target="HOST" -- the slave bridge --
  // so staging stays off and nothing else in this file changes behaviour.
  void staged_probe() {
    staged_cfg_.reset();
    // VORTEX_AVED_FORCE_STAGE exists because this path is otherwise
    // hardware-only and therefore untestable without a board.
    //
    // sim_mode_ normally routes to the sim_* apparatus, which models CP-visible
    // memory by copying it into the simulator. That is the right default -- the
    // two mechanisms must not stack -- but it also means the staged path, which
    // is the one carrying the publish/refresh ordering and the deferred-free
    // lifetime rules, never executes anywhere except on silicon. A board run
    // here costs a device reset and can cost a host reset, so "first execution
    // is on hardware" is a bad trade for the riskiest code in the file.
    //
    // With this set, staging runs against the simulated device memory instead:
    // vrt::Buffer and sync() carry the same API on the simulation platform (the
    // addresses are drawn from the linker's simulated windows), so the
    // ordering, the ring exclusion and the pending-free lifetime all get
    // exercised. Never set it for a hardware run -- there it is already on by
    // virtue of the vbin, and forcing it changes nothing but the log line.
    const char* force = getenv("VORTEX_AVED_FORCE_STAGE");
    const bool force_stage = (force != nullptr && force[0] != '\0'
                              && force[0] != '0');
    if (sim_mode_ && !force_stage) {
      return;  // the sim_* path already models this; do not stack the two.
    }
    if (sim_mode_ && force_stage) {
      fprintf(stderr, "[VXDRV] VORTEX_AVED_FORCE_STAGE set: exercising the "
                      "staged CP path against the simulator\n");
    }
    try {
      staged_cfg_ = vrtKernel_.portMemoryConfig(HOST_PORT_NAME);
      fprintf(stderr,
              "[VXDRV] m_axi_host targets device memory; staging CP memory "
              "there (HBM port %d)\n",
              staged_cfg_->hbmPort ? int(*staged_cfg_->hbmPort) : -1);
    } catch (const std::exception&) {
      // target="HOST": the QDMA slave bridge. Nothing to stage.
    }
  }

  // Moves a whole region in one direction. vrt::Buffer::sync has no offset or
  // length, so unlike sim_xfer this cannot ship just the cachelines appended
  // since the last doorbell. The ring is CP_RING_SIZE and a sync is a QDMA
  // descriptor rather than a simulated AXI burst, so paying for the whole
  // region is cheap enough here; if it ever shows up in a profile, the fix is
  // a partial-sync API in VRT, not a smarter loop.
  int staged_xfer(const staged_region_t& r, bool to_device) {
    VRT_TRY()
      r.buf->sync(to_device ? vrt::SyncType::HOST_TO_DEVICE
                            : vrt::SyncType::DEVICE_TO_HOST);
    VRT_CATCH(-1)
    return 0;
  }

  // Publish what the CP is about to read. Once the queue is live the head and
  // completion cachelines belong to the CP, so pushing them would clobber its
  // writes; include_cp_owned is set only for the one-shot seeding on the first
  // doorbell, before the CP can have fetched anything.
  int staged_publish(bool include_cp_owned) {
    std::lock_guard<std::mutex> g(staged_mu_);
    for (const auto& kv : staged_regions_) {
      const uint32_t lo = uint32_t(kv.first & 0xFFFFFFFFu);
      if (!include_cp_owned
          && (lo == staged_head_addr_ || lo == staged_cmpl_addr_)) {
        continue;
      }
      if (staged_xfer(kv.second, true) != 0) {
        return -1;
      }
    }
    return 0;
  }

  // Pull back what the CP wrote: MEM_READ staging buffers, and the head and
  // completion lines it owns.
  //
  // The ring is EXCLUDED, and must be. It is ours to write: cp_ring_append_
  // fills the host shadow and the bytes only reach the device at the next
  // doorbell. This runs from the Q_SEQNUM poll, which can land between an
  // append and that doorbell, so pulling the ring back would overwrite
  // freshly appended descriptors with the stale device copy and silently drop
  // commands. sim_refresh() skips it for the same reason.
  int staged_refresh() {
    std::lock_guard<std::mutex> g(staged_mu_);
    for (const auto& kv : staged_regions_) {
      if (uint32_t(kv.first & 0xFFFFFFFFu) == staged_ring_addr_) {
        continue;
      }
      if (staged_xfer(kv.second, false) != 0) {
        return -1;
      }
    }
    // The CP has retired work (this only runs on a Q_SEQNUM advance), so every
    // descriptor appended before that advance has been read. Buffers freed by
    // host_free while a batch was still open can now go back to the allocator.
    // Holding them one seqnum longer than strictly necessary costs a little
    // device memory and nothing else; releasing them early costs correctness.
    staged_pending_free_.clear();
    return 0;
  }

  // Mirrors sim_note_region_addr: record which base is which so publish can
  // tell the CP-owned lines from the ones we write.
  void staged_note_region_addr(uint32_t off, uint32_t value) {
    switch (off) {
    case CP_Q_RING_BASE_LO: staged_ring_addr_ = value; break;
    case CP_Q_HEAD_ADDR_LO: staged_head_addr_ = value; break;
    case CP_Q_CMPL_ADDR_LO: staged_cmpl_addr_ = value; break;
    default: break;
    }
  }

  // ----- Simulation host-memory sync -----
  struct sim_region_t { void *host_ptr; uint64_t size; };

  // Set VORTEX_AVED_TRACE=1 to log every host-memory sync. The harness's own
  // SIM_EXEC_VERBOSE shows what reached the model; this shows what this
  // backend intended, and the two together localise a lost transfer to one
  // side or the other. Off by default: this is on the doorbell path.
  static bool sim_trace_enabled() {
    static const bool on = []() {
      const char *v = getenv("VORTEX_AVED_TRACE");
      return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
  }

  void sim_trace(const char *what, uint64_t addr, uint64_t off, uint64_t len) {
    if (!sim_trace_enabled()) {
      return;
    }
    fprintf(stderr, "[aved-sim] %-8s addr=0x%llx off=0x%llx len=0x%llx\n", what,
            (unsigned long long)addr, (unsigned long long)off,
            (unsigned long long)len);
  }

  // Records the region bases the common core programs into the CP so the two
  // directions below can tell who owns each one.
  void sim_note_region_addr(uint32_t off, uint32_t value) {
    switch (off) {
    case CP_Q_RING_BASE_LO: sim_ring_addr_ = value; break;
    case CP_Q_HEAD_ADDR_LO: sim_head_addr_ = value; break;
    case CP_Q_CMPL_ADDR_LO: sim_cmpl_addr_ = value; break;
    default: break;
    }
  }

  // Moves [off, off+len) of one region. Sub-ranges matter: the ring is 64 KiB
  // but a doorbell only ever publishes the handful of cachelines appended
  // since the last one, and every byte transferred here is a full AXI burst
  // driven through the simulated fabric.
  int sim_xfer(uint64_t addr, void *host_ptr, uint64_t off, uint64_t len,
               bool to_device) {
    if (len == 0) {
      return 0;
    }
    auto *base = static_cast<uint8_t *>(host_ptr) + off;
    sim_trace(to_device ? "push" : "pull", addr, off, len);
    VRT_TRY()
      auto server = vrtDevice_.getHandle()->getZmqServer();
      if (!server) {
        return -1;
      }
      if (to_device) {
        std::vector<uint8_t> buf(base, base + len);
        server->sendBufferSim(addr + off, buf);
      } else {
        std::vector<uint8_t> buf;
        server->fetchBufferSim(addr + off, len, buf);
        if (buf.size() < len) {
          return -1;
        }
        std::memcpy(base, buf.data(), len);
      }
    VRT_CATCH(-1)
    return 0;
  }

  // Publish what the CP is about to read. Once the queue is live the head and
  // completion cachelines belong to the CP, so pushing them would race its
  // writes; `include_cp_owned` is set only for the one-shot seeding done
  // before the queue is enabled.
  int sim_publish(bool include_cp_owned, uint32_t new_tail) {
    std::lock_guard<std::mutex> g(sim_mu_);
    for (const auto &kv : sim_regions_) {
      uint32_t lo = uint32_t(kv.first & 0xFFFFFFFFu);
      if (!include_cp_owned && (lo == sim_head_addr_ || lo == sim_cmpl_addr_)) {
        continue;
      }
      uint64_t off = 0;
      uint64_t len = kv.second.size;
      // For the ring, ship only what was appended since the last doorbell.
      // The doorbell value is the new tail, and the common core refuses to
      // wrap a cacheline across the end of the ring, so the delta is always
      // one contiguous span. Anything unexpected falls back to the whole
      // region, which is slow but never wrong.
      if (lo == sim_ring_addr_ && !include_cp_owned) {
        uint32_t delta = new_tail - sim_ring_tail_;
        uint64_t start = sim_ring_tail_ % kv.second.size;
        if (delta != 0 && delta <= kv.second.size &&
            start + delta <= kv.second.size) {
          off = start;
          len = delta;
        }
      }
      if (sim_xfer(kv.first, kv.second.host_ptr, off, len, true) != 0) {
        return -1;
      }
    }
    sim_ring_tail_ = new_tail;
    return 0;
  }

  // Refresh what the CP writes. The ring is ours alone, so pulling it back
  // would clobber commands already staged for a later doorbell.
  int sim_refresh() {
    std::lock_guard<std::mutex> g(sim_mu_);
    for (const auto &kv : sim_regions_) {
      uint32_t lo = uint32_t(kv.first & 0xFFFFFFFFu);
      if (lo == sim_ring_addr_) {
        continue;
      }
      if (sim_xfer(kv.first, kv.second.host_ptr, 0, kv.second.size, false) != 0) {
        return -1;
      }
    }
    return 0;
  }
#endif

  int write_register(uint32_t addr, uint32_t value) {
  #ifdef CPP_API
    VRT_TRY()
      vrtKernel_.write(addr, value);
    VRT_CATCH(-1)
  #else
    CHECK_ERR(vrtKernelWriteRegister(vrtKernel_, addr, value), {
      return err;
    });
  #endif
    return 0;
  }

  int read_register(uint32_t addr, uint32_t *value) {
  #ifdef CPP_API
    VRT_TRY()
      *value = vrtKernel_.read(addr);
    VRT_CATCH(-1)
  #else
    CHECK_ERR(vrtKernelReadRegister(vrtKernel_, addr, value), {
      return err;
    });
  #endif
    return 0;
  }

  vrt_device_t vrtDevice_;
  vrt_kernel_t vrtKernel_;

#ifdef CPP_API
  // Keyed by the address the device masters to, which is what host_mem_free
  // is given. std::map is not thread-safe even on disjoint keys and queue
  // workers allocate from arbitrary threads, so the mutex covers every site.
  std::map<uint64_t, vrtd::Session::HostBuffer> host_bufs_;
  std::mutex                                    host_bufs_mu_;

  // Set when the vbin is not a hardware platform, i.e. when there is no slave
  // bridge and the CP's host memory has to be synced explicitly.
  bool sim_mode_ = false;

  std::map<uint64_t, sim_region_t> sim_regions_;
  std::mutex                       sim_mu_;

  // Low halves of the region bases; enough to tell the three CP regions apart
  // within one allocation window.
  uint32_t sim_ring_addr_    = 0;
  uint32_t sim_head_addr_    = 0;
  uint32_t sim_cmpl_addr_    = 0;
  uint32_t sim_last_seqnum_  = 0;
  bool     sim_seqnum_valid_ = false;
  bool     sim_seeded_       = false;
  uint32_t sim_ring_tail_    = 0;

  // ----- Staged CP memory (m_axi_host wired to device memory) -----
  // Set when m_axi_host resolves to a memory bank instead of the QDMA slave
  // bridge. Unset means target="HOST" and allocHostBuffer is correct.
  std::optional<vrt::MemoryConfig> staged_cfg_;

  std::map<uint64_t, staged_region_t> staged_regions_;
  // Regions host_free'd while their descriptors may still be unread by the CP.
  // Released in staged_refresh(), i.e. once Q_SEQNUM has advanced. Guarded by
  // staged_mu_ like staged_regions_.
  std::vector<staged_region_t>        staged_pending_free_;
  std::mutex                          staged_mu_;

  uint32_t staged_ring_addr_    = 0;
  uint32_t staged_head_addr_    = 0;
  uint32_t staged_cmpl_addr_    = 0;
  uint32_t staged_last_seqnum_  = 0;
  bool     staged_seqnum_valid_ = false;
  bool     staged_seeded_       = false;
#endif
};

#include <callbacks.inc>
