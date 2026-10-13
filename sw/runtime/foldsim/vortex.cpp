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
// foldsim backend — a pure transport HAL (see callbacks.h): the target
// carries no Command Processor, so the regfile surface comes from the
// functional CommandProcessor model, and its hooks reach the foldsim
// simulator. Device memory is the simulator's, in device memory at
// FOLDSIM_REG_MEM_BASE; the target's DCR, start and busy are its command queue
// and status (sw/common/foldsim_regs.h).
//
// TARGET=hw runs the image on a V80 through VRT: registers over AXI4-Lite,
// device memory over a QDMA queue. TARGET=avedsim simulates the same kernel
// in-process (sim/foldsim).
// ============================================================================

#include <VX_types.h>
#include <common.h>

#include <cmd_processor.h>
#include <foldsim_regs.h>
#include <util.h>

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef FOLDSIM_EMUL
#include <foldsim_emul.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <slash/uapi/slash_interface.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <v80_pcie.h>
#include <vrt/device.hpp>
#include <vrt/kernel.hpp>
#include <vrtd/device.hpp>
#include <vrtd/session.hpp>
#endif

using namespace vortex;

namespace {

// Where target address 0 is in device memory: the start of HBM, which the
// kernel's memory port reaches through the MEM tag (hw/syn/xilinx/platforms/
// v80.toml). The target's 4 GB lie in HBM0..HBM7, which nothing else uses.
constexpr uint64_t kMemBase = 0x4000000000ull;

#ifdef FOLDSIM_EMUL

class transport_t {
public:
  // The kernel advances while it is accessed.
  static constexpr bool kRunsOnItsOwn = false;
  int init() { return sim_.init(); }
  int reg_write(uint32_t off, uint32_t v) { return sim_.reg_write(off, v); }
  int reg_read(uint32_t off, uint32_t* v) { return sim_.reg_read(off, v); }
  int mem_write(uint64_t addr, const void* src, uint64_t n) {
    return sim_.mem_write(kMemBase + addr, src, n);
  }
  int mem_read(uint64_t addr, void* dst, uint64_t n) {
    return sim_.mem_read(kMemBase + addr, dst, n);
  }

private:
  foldsim_emul sim_;
};

#else

#define KERNEL_NAME "foldsim_afu_0"

// VRT throws; nothing may cross the callbacks' C boundary.
#define VRT_TRY() try {
#define VRT_CATCH(_ret)                                                        \
  }                                                                            \
  catch (const std::exception& _e) {                                           \
    fprintf(stderr, "[VXDRV] VRT exception: %s\n", _e.what());                 \
    return _ret;                                                               \
  }

class transport_t {
public:
  // The kernel runs on the card whether or not the host looks.
  static constexpr bool kRunsOnItsOwn = true;
  int init() {
    const char* vbin = getenv("VRT_VBIN_PATH");
    if (vbin == nullptr || vbin[0] == '\0') {
      fprintf(stderr, "[VXDRV] VRT_VBIN_PATH must name the foldsim .vbin\n");
      return -1;
    }
    const char* bdf_env = getenv("VRT_DEVICE_BDF");
    std::string bdf = (bdf_env != nullptr) ? bdf_env : "";
    if (bdf.empty()) {
      bdf = discover_v80_bdf();
      if (bdf.empty()) {
        fprintf(stderr, "[VXDRV] no V80 found on the bus; set VRT_DEVICE_BDF\n");
        return -1;
      }
    }
    VRT_TRY()
      device_ = std::make_unique<vrt::Device>(bdf, vbin, /*program=*/true);
      // Clock the kernel at the rate the image was timed for.
      const uint64_t timed_hz = device_->getMaxFrequency();
      device_->setFrequency(timed_hz);
      const uint64_t clk_hz = device_->getFrequency();
      if (clk_hz == 0 || clk_hz > timed_hz + timed_hz / 200) {
        fprintf(stderr, "[VXDRV] user clock reads %lu Hz, image is timed for %lu Hz\n",
                (unsigned long)clk_hz, (unsigned long)timed_hz);
        return -1;
      }
      printf("[VXDRV] foldsim kernel clock %lu Hz\n", (unsigned long)clk_hz);
      kernel_ = std::make_unique<vrt::Kernel>(*device_, KERNEL_NAME);
      // A QDMA queue for device memory: the buffer's own memory is the
      // smallest allocation, outside the target's 4 GB, and only its queue is
      // used, at the target's addresses.
      session_ = std::make_unique<vrtd::Session>();
      auto vdev = session_->getDeviceByBdf(bdf);
      dma_ = std::make_unique<vrtd::Buffer>(vdev.openHbmBuffer(/*region=*/8, kDmaChunk));
      fd_ = dma_->getFd();
    VRT_CATCH(-1)
    // The host side of the transfers: a kernel DMA buffer, mapped.
    struct slash_qdma_buf_create create = {};
    create.size = sizeof(create);
    create.flags = O_CLOEXEC;
    create.length = kStaging;
    staging_fd_ = ioctl(fd_, SLASH_QDMA_IOCTL_BUF_CREATE, &create);
    if (staging_fd_ < 0) {
      fprintf(stderr, "[VXDRV] cannot create a QDMA buffer: %s\n", strerror(errno));
      return -1;
    }
    void* p = mmap(nullptr, kStaging, PROT_READ | PROT_WRITE, MAP_SHARED, staging_fd_, 0);
    if (p == MAP_FAILED) {
      fprintf(stderr, "[VXDRV] cannot map the QDMA buffer: %s\n", strerror(errno));
      return -1;
    }
    staging_ = static_cast<uint8_t*>(p);
    return 0;
  }

  ~transport_t() {
    if (staging_ != nullptr)
      munmap(staging_, kStaging);
    if (staging_fd_ >= 0)
      close(staging_fd_);
  }

  int reg_write(uint32_t off, uint32_t v) {
    VRT_TRY()
      kernel_->write(off, v);
    VRT_CATCH(-1)
    return 0;
  }

  int reg_read(uint32_t off, uint32_t* v) {
    VRT_TRY()
      *v = kernel_->read(off);
    VRT_CATCH(-1)
    return 0;
  }

  // QDMA moves any byte range: only the offset into the staging buffer must
  // be page aligned, not the device address or the length. A write is the
  // bytes it writes, with no read of the pages around them.
  int mem_write(uint64_t addr, const void* src, uint64_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* bytes = static_cast<const uint8_t*>(src);
    while (n != 0) {
      const uint64_t span = std::min<uint64_t>(kDmaChunk, n);
      std::memcpy(staging_, bytes, span);
      if (dma(addr, staging_, span, true) != 0)
        return -1;
      addr += span;
      bytes += span;
      n -= span;
    }
    return 0;
  }

  int mem_read(uint64_t addr, void* dst, uint64_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* bytes = static_cast<uint8_t*>(dst);
    while (n != 0) {
      const uint64_t span = std::min<uint64_t>(kDmaChunk, n);
      if (dma(addr, staging_, span, false) != 0)
        return -1;
      std::memcpy(bytes, staging_, span);
      addr += span;
      bytes += span;
      n -= span;
    }
    return 0;
  }

private:
  static constexpr uint64_t kDmaChunk = 64ull << 20;
  static constexpr uint64_t kStaging = kDmaChunk;

  // `len` bytes at target address `addr`, from or to `host` in the staging
  // buffer, page aligned.
  int dma(uint64_t addr, uint8_t* host, uint64_t len, bool to_device) {
    struct slash_qdma_transfer xfer = {};
    xfer.size = sizeof(xfer);
    xfer.count = 1;
    xfer.xfers[0].qpair_index = 0;
    xfer.xfers[0].direction = to_device ? SLASH_QDMA_XFER_H2C : SLASH_QDMA_XFER_C2H;
    xfer.xfers[0].buf_fd = staging_fd_;
    xfer.xfers[0].buf_offset = static_cast<uint64_t>(host - staging_);
    xfer.xfers[0].dev_addr = kMemBase + addr;
    xfer.xfers[0].length = len;
    const int n = ioctl(fd_, SLASH_QDMA_QPAIR_IOCTL_TRANSFER, &xfer);
    if (n < 0 || static_cast<uint64_t>(n) != len) {
      fprintf(stderr, "[VXDRV] QDMA %s of %lu bytes at 0x%lx failed: %s\n",
              to_device ? "write" : "read", (unsigned long)len,
              (unsigned long)(kMemBase + addr), n < 0 ? strerror(errno) : "short");
      return -1;
    }
    return 0;
  }

  std::unique_ptr<vrt::Device> device_;
  std::unique_ptr<vrt::Kernel> kernel_;
  std::unique_ptr<vrtd::Session> session_;
  std::unique_ptr<vrtd::Buffer> dma_;
  int fd_ = -1;
  int staging_fd_ = -1;
  uint8_t* staging_ = nullptr;
  std::mutex mutex_;
};

#endif

uint64_t env_or(const char* name, uint64_t value) {
  const char* s = getenv(name);
  return (s != nullptr && s[0] != '\0') ? strtoull(s, nullptr, 0) : value;
}

// What the host spent on the transport: register accesses, and device-memory
// transfers, with their wall time.
struct transport_stats_t {
  uint64_t reads = 0, writes = 0, dma_bytes = 0;
  double reg_s = 0, dma_s = 0;
};

class timer_t {
public:
  explicit timer_t(double& total) : total_(total), start_(std::chrono::steady_clock::now()) {}
  ~timer_t() {
    total_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

private:
  double& total_;
  std::chrono::steady_clock::time_point start_;
};

// The transport, counted.
class counted_transport_t : public transport_t {
public:
  int reg_write(uint32_t off, uint32_t v) {
    timer_t t(stats.reg_s);
    ++stats.writes;
    return transport_t::reg_write(off, v);
  }
  int reg_read(uint32_t off, uint32_t* v) {
    timer_t t(stats.reg_s);
    ++stats.reads;
    return transport_t::reg_read(off, v);
  }
  int mem_write(uint64_t addr, const void* src, uint64_t n) {
    timer_t t(stats.dma_s);
    stats.dma_bytes += n;
    return transport_t::mem_write(addr, src, n);
  }
  int mem_read(uint64_t addr, void* dst, uint64_t n) {
    timer_t t(stats.dma_s);
    stats.dma_bytes += n;
    return transport_t::mem_read(addr, dst, n);
  }
  transport_stats_t stats;
};

} // namespace

class vx_device {
public:
  vx_device() : cp_(make_cp_hooks()) {}

  ~vx_device() {
    if (open_)
      report();
    for (auto& kv : host_regions_)
      free(reinterpret_cast<void*>(kv.first));
    host_regions_.clear();
  }

  int init() {
    if (transport_.init() != 0)
      return -1;
    uint32_t id = 0, version = 0;
    if (transport_.reg_read(FOLDSIM_REG_ID, &id) != 0 || id != FOLDSIM_ID) {
      fprintf(stderr, "[VXDRV] the kernel is not a foldsim simulator (id 0x%08x)\n", id);
      return -1;
    }
    if (transport_.reg_read(FOLDSIM_REG_VERSION, &version) != 0 || version != FOLDSIM_VERSION) {
      fprintf(stderr, "[VXDRV] the kernel's register map is version %u, this runtime's %u\n",
              version, FOLDSIM_VERSION);
      return -1;
    }
    // The memory model: a latency-bandwidth pipe, 30-cycle latency and 16
    // transactions in flight each way unless set.
    const uint32_t rlat = env_or("FOLDSIM_MEM_READ_LATENCY", 30);
    const uint32_t wlat = env_or("FOLDSIM_MEM_WRITE_LATENCY", 30);
    const uint32_t maxr = env_or("FOLDSIM_MEM_MAX_READS", 16);
    const uint32_t maxw = env_or("FOLDSIM_MEM_MAX_WRITES", 16);
    if (reg_write(FOLDSIM_REG_MEM_BASE_LO, uint32_t(kMemBase)) != 0 ||
        reg_write(FOLDSIM_REG_MEM_BASE_HI, uint32_t(kMemBase >> 32)) != 0 ||
        reg_write(FOLDSIM_REG_MEM_RLAT, rlat) != 0 ||
        reg_write(FOLDSIM_REG_MEM_WLAT, wlat) != 0 ||
        reg_write(FOLDSIM_REG_MEM_MAXR, maxr) != 0 ||
        reg_write(FOLDSIM_REG_MEM_MAXW, maxw) != 0 ||
        reg_write(FOLDSIM_REG_CTRL, 1) != 0)
      return -1;
    // A target left running by an earlier process is reset, as a card's
    // native driver resets its device on open: Vortex counts VX_CFG_RESET_DELAY
    // cycles of reset before it takes a request. The reset also clears a
    // memory error that process left, so only the state after it counts.
    uint32_t s = 0;
    if (command(FOLDSIM_CMD_RESET, 0, 64) != 0 || wait_idle(false) != 0 || status(&s) != 0)
      return -1;
    if (mem_error(s))
      return -1;
    open_ = true;
    return 0;
  }

  // ----- CP register channel -----
  // No hardware CP: the functional model answers, ticked around each access.
  // The model's hooks return no status: a device error they hit fails every
  // access from then on, so the command that hit it does not complete as a
  // success.
  int cp_reg_write(uint32_t off, uint32_t value) {
    cp_.mmio_write(off, value);
    for (int i = 0; i < 256 && cp_.busy(); ++i)
      cp_.tick();
    return failed_ ? -1 : 0;
  }

  int cp_reg_read(uint32_t off, uint32_t* value) {
    for (int i = 0; i < 256 && cp_.busy(); ++i)
      cp_.tick();
    *value = cp_.mmio_read(off);
    return failed_ ? -1 : 0;
  }

  // ----- CP-visible host memory -----
  // The CP model reads host memory directly; only device memory crosses the
  // transport.
  int host_mem_alloc(uint64_t size, void** host_ptr, uint64_t* cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    void* ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
    if (ptr == nullptr)
      return -1;
    std::lock_guard<std::mutex> g(host_mu_);
    host_regions_[reinterpret_cast<uint64_t>(ptr)] = asize;
    *host_ptr = ptr;
    *cp_addr = reinterpret_cast<uint64_t>(ptr);
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
    {
      std::lock_guard<std::mutex> g(host_mu_);
      auto it = host_regions_.find(cp_addr);
      if (it == host_regions_.end())
        return -1;
      host_regions_.erase(it);
    }
    free(reinterpret_cast<void*>(cp_addr));
    return 0;
  }

  int host_mem_pull(uint64_t /*cp_addr*/) { return 0; }
  int host_mem_push(uint64_t /*cp_addr*/) { return 0; }

private:
  int reg_write(uint32_t off, uint32_t v) { return transport_.reg_write(off, v); }

  uint64_t read64(uint32_t lo) {
    uint32_t l = 0, h = 0, h2 = 0;
    do {
      transport_.reg_read(lo + 4, &h);
      transport_.reg_read(lo, &l);
      transport_.reg_read(lo + 4, &h2);
    } while (h != h2);
    return (uint64_t(h) << 32) | l;
  }

  // How fast the simulator ran since the kernel's reset: target cycles, the
  // kernel clock's cycles, and those a target cycle waited for device memory.
  void report() {
    const uint64_t target = read64(FOLDSIM_REG_TCYCLES_LO);
    const uint64_t host = read64(FOLDSIM_REG_HCYCLES_LO);
    const uint64_t stall = read64(FOLDSIM_REG_MSTALL_LO);
    fprintf(stderr,
            "[foldsim] target cycles=%lu host cycles=%lu host/target=%.2f memory waits=%lu\n",
            (unsigned long)target, (unsigned long)host,
            target ? double(host) / double(target) : 0.0, (unsigned long)stall);
    // Each class finishes each of its threads once a target cycle, so its
    // finishes per host cycle is how busy its one physical copy is.
    uint32_t classes = 0;
    transport_.reg_read(FOLDSIM_REG_NUM_CLASSES, &classes);
    std::string line;
    for (uint32_t c = 0; c < classes && host != 0; ++c) {
      const uint64_t finishes = read64(FOLDSIM_REG_FINISH(c));
      char buf[64];
      snprintf(buf, sizeof(buf), " %u:%.1f%%", c, 100.0 * double(finishes) / double(host));
      line += buf;
    }
    if (!line.empty())
      fprintf(stderr, "[foldsim] class busy:%s\n", line.c_str());
    const auto& st = transport_.stats;
    fprintf(stderr,
            "[foldsim] host: %lu register reads, %lu writes in %.3f s; %lu bytes of device memory in %.3f s\n",
            (unsigned long)st.reads, (unsigned long)st.writes, st.reg_s,
            (unsigned long)st.dma_bytes, st.dma_s);
  }

  // Latches a device error: reported once, and every later access fails.
  int fail(const char* what) {
    if (!failed_) {
      fprintf(stderr, "[VXDRV] %s\n", what);
      failed_ = true;
    }
    return -1;
  }

  int status(uint32_t* value) {
    if (transport_.reg_read(FOLDSIM_REG_STATUS, value) != 0)
      return fail("cannot read the simulator's status");
    return 0;
  }

  // Polls the status until `done` holds of it. A memory error fails the wait
  // (unless `error_fails` is false), and so does a simulator whose target
  // cycles stop advancing for the control timeout. Progress, not wall time,
  // is the measure: in-process, a large target may legitimately take minutes
  // of wall time to answer a DCR read that flushes its caches.
  template <typename Done>
  int poll(Done done, bool error_fails, const char* what) {
    auto last_progress = std::chrono::steady_clock::now();
    uint32_t last_tcycles = 0;
    if (transport_.reg_read(FOLDSIM_REG_TCYCLES_LO, &last_tcycles) != 0)
      return fail("cannot read the simulator's target cycles");
    uint32_t s = 0;
    for (uint64_t n = 0;; ++n) {
      if (failed_ || status(&s) != 0 || (error_fails && mem_error(s)))
        return -1;
      if (done(s))
        break;
      // Checked only every so often: in-process, a status read is the cheap
      // part of a poll.
      if ((n & 0xff) == 0xff) {
        uint32_t tcycles = 0;
        if (transport_.reg_read(FOLDSIM_REG_TCYCLES_LO, &tcycles) != 0)
          return fail("cannot read the simulator's target cycles");
        if (tcycles != last_tcycles) {
          last_tcycles = tcycles;
          last_progress = std::chrono::steady_clock::now();
        } else if (elapsed_s(last_progress) > control_timeout_s_) {
          char msg[192];
          snprintf(msg, sizeof(msg), "the target has not %s, and has not advanced a cycle in "
                   "%.0f s (status 0x%x); the simulation is stopped", what, control_timeout_s_, s);
          return fail(msg);
        }
      }
    }
    return 0;
  }

  // Queues a command once the queue has room.
  int command(uint32_t op, uint32_t addr, uint32_t data) {
    if (poll([](uint32_t s) { return !(s & FOLDSIM_STATUS_FULL); }, true,
             "taken a command") != 0)
      return -1;
    if (reg_write(FOLDSIM_REG_CMD_ADDR, addr) != 0 || reg_write(FOLDSIM_REG_CMD_DATA, data) != 0 ||
        reg_write(FOLDSIM_REG_CMD, op) != 0)
      return fail("cannot queue a command");
    return 0;
  }

  int wait_idle(bool error_fails) {
    return poll([](uint32_t s) { return !(s & FOLDSIM_STATUS_BUSY); }, error_fails,
                "gone idle");
  }

  // The memory endpoint takes single-beat transactions; a target that makes a
  // burst stops the simulation there, busy for good.
  bool mem_error(uint32_t s) {
    if (!(s & FOLDSIM_STATUS_MEM_ERR))
      return false;
    fail("the target made a memory burst, which the memory model does not take; "
         "the simulation is stopped");
    return true;
  }

  static double elapsed_s(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
  }

  // The target runs on its own: once it has been busy for a while, polling it
  // back to back only spins a core, so the poll sleeps, longer the longer the
  // target stays busy (up to kMaxPollSleepUs).
  void poll_backoff(bool busy) {
    if (!transport_t::kRunsOnItsOwn)
      return;
    if (!busy) {
      busy_polls_ = 0;
      return;
    }
    if (++busy_polls_ <= kSpinPolls)
      return;
    const uint64_t us = std::min<uint64_t>(1ull << std::min<uint64_t>(busy_polls_ - kSpinPolls, 7),
                                           kMaxPollSleepUs);
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }

  void* host_region_ptr(uint64_t addr) {
    std::lock_guard<std::mutex> g(host_mu_);
    if (host_regions_.empty())
      return nullptr;
    auto it = host_regions_.upper_bound(addr);
    if (it == host_regions_.begin())
      return nullptr;
    --it;
    if (addr >= it->first && addr < it->first + it->second)
      return reinterpret_cast<void*>(addr);
    return nullptr;
  }

  vortex::CommandProcessor::Hooks make_cp_hooks() {
    vortex::CommandProcessor::Hooks h;
    h.dram_read = [this](uint64_t addr, void* dst, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(dst, hp, bytes);
        return;
      }
      if (failed_ || transport_.mem_read(addr, dst, bytes) != 0) {
        fail("cannot read device memory");
        std::memset(dst, 0, bytes);
      }
    };
    h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(hp, src, bytes);
        return;
      }
      if (failed_ || transport_.mem_write(addr, src, bytes) != 0)
        fail("cannot write device memory");
    };
    h.vortex_dcr_write = [this](uint32_t addr, uint32_t value) {
      if (!failed_)
        command(FOLDSIM_CMD_DCR_WRITE, addr, value);
    };
    h.vortex_dcr_read = [this](uint32_t addr, uint32_t tag) -> uint32_t {
      uint32_t value = 0;
      if (failed_ || command(FOLDSIM_CMD_DCR_READ, addr, tag) != 0 ||
          poll([](uint32_t s) { return (s & FOLDSIM_STATUS_RSP) != 0; }, true,
               "answered a DCR read") != 0)
        return 0;
      if (transport_.reg_read(FOLDSIM_REG_DCR_RSP, &value) != 0) {
        fail("cannot read a DCR response");
        return 0;
      }
      return value;
    };
    h.vortex_start = [this]() {
      if (!failed_ && command(FOLDSIM_CMD_START, 0, 0) == 0)
        launch_start_ = std::chrono::steady_clock::now();
    };
    // A failed device reads idle, so the launch ends; the access after it
    // reports the failure.
    h.vortex_busy = [this]() -> bool {
      uint32_t s = 0;
      if (failed_ || status(&s) != 0 || mem_error(s))
        return false;
      const bool busy = (s & FOLDSIM_STATUS_BUSY) != 0;
      if (busy && launch_timeout_s_ > 0 && elapsed_s(launch_start_) > launch_timeout_s_) {
        char msg[128];
        snprintf(msg, sizeof(msg), "launch exceeded VORTEX_FOLDSIM_TIMEOUT_S=%.0f s; "
                 "the simulation is stopped", launch_timeout_s_);
        fail(msg);
        return false;
      }
      poll_backoff(busy);
      return busy;
    };
#ifdef VX_CFG_VM_ENABLE
    h.mmu_fault_report = true;
#endif
    return h;
  }

  // Polls back to back before the first sleep: about a millisecond at a PCIe
  // round trip each.
  static constexpr uint64_t kSpinPolls = 1024;
  static constexpr uint64_t kMaxPollSleepUs = 100;

  // Wall-clock seconds a launch may run before the device is failed; 0, the
  // default, never. A command, DCR read or reset the target leaves unanswered
  // while it advances no cycle for that long fails it too, after 60 s when it
  // is unset.
  static double timeout_env() {
    const char* s = getenv("VORTEX_FOLDSIM_TIMEOUT_S");
    return (s != nullptr && s[0] != '\0') ? atof(s) : 0.0;
  }
  static constexpr double kControlTimeoutS = 60;

  counted_transport_t transport_;
  uint64_t busy_polls_ = 0;
  const double launch_timeout_s_ = timeout_env();
  const double control_timeout_s_ = (launch_timeout_s_ > 0) ? launch_timeout_s_ : kControlTimeoutS;
  std::chrono::steady_clock::time_point launch_start_;
  bool failed_ = false;
  bool open_ = false;
  vortex::CommandProcessor cp_;
  std::mutex host_mu_;
  std::map<uint64_t, uint64_t> host_regions_;
};

#include <callbacks.inc>
