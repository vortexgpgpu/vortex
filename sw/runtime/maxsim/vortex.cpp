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
// maxsim backend — a pure transport HAL (see callbacks.h), like the firesim
// backend: the target carries no Command Processor, so the regfile surface
// comes from the functional CommandProcessor model, and its hooks reach the
// maxsim simulator. Device memory is the simulator's, in device memory at
// MAXSIM_REG_MEM_BASE; the target's DCR, start and busy are its command queue
// and status (sw/common/maxsim_regs.h).
//
// TARGET=hw runs the image on a V80 through VRT: registers over AXI4-Lite,
// device memory over a QDMA queue. TARGET=avedsim simulates the same kernel
// in-process (sim/maxsim).
// ============================================================================

#include <VX_types.h>
#include <common.h>

#include <cmd_processor.h>
#include <maxsim_regs.h>
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

#ifdef MAXSIM_EMUL
#include <maxsim_emul.h>
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

#ifdef MAXSIM_EMUL

class transport_t {
public:
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
  maxsim_emul sim_;
};

#else

#define KERNEL_NAME "maxsim_afu_0"

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
  int init() {
    const char* vbin = getenv("VRT_VBIN_PATH");
    if (vbin == nullptr || vbin[0] == '\0') {
      fprintf(stderr, "[VXDRV] VRT_VBIN_PATH must name the maxsim .vbin\n");
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
      printf("[VXDRV] maxsim kernel clock %lu Hz\n", (unsigned long)clk_hz);
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

  // QDMA moves whole 4 KB pages; a partial one is read, merged and written.
  int mem_write(uint64_t addr, const void* src, uint64_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* bytes = static_cast<const uint8_t*>(src);
    while (n != 0) {
      const uint64_t page = addr & ~(kPage - 1);
      const uint64_t off = addr - page;
      const uint64_t span = std::min<uint64_t>(kDmaChunk - off, n);
      const uint64_t len = (off + span + kPage - 1) & ~(kPage - 1);
      if (off != 0 || (span % kPage) != 0) {
        // Only the first and last pages can be partial.
        if (dma(page, staging_, kPage, false) != 0 ||
            (len > kPage &&
             dma(page + len - kPage, staging_ + len - kPage, kPage, false) != 0))
          return -1;
      }
      std::memcpy(staging_ + off, bytes, span);
      if (dma(page, staging_, len, true) != 0)
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
      const uint64_t page = addr & ~(kPage - 1);
      const uint64_t off = addr - page;
      const uint64_t span = std::min<uint64_t>(kDmaChunk - off, n);
      const uint64_t len = (off + span + kPage - 1) & ~(kPage - 1);
      if (dma(page, staging_, len, false) != 0)
        return -1;
      std::memcpy(bytes, staging_ + off, span);
      addr += span;
      bytes += span;
      n -= span;
    }
    return 0;
  }

private:
  static constexpr uint64_t kPage = 4096;
  static constexpr uint64_t kDmaChunk = 64ull << 20;
  static constexpr uint64_t kStaging = kDmaChunk + kPage;

  // `len` bytes at target address `addr`, from or to `host` in the staging
  // buffer, both page aligned.
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
    if (transport_.reg_read(MAXSIM_REG_ID, &id) != 0 || id != MAXSIM_ID) {
      fprintf(stderr, "[VXDRV] the kernel is not a maxsim simulator (id 0x%08x)\n", id);
      return -1;
    }
    if (transport_.reg_read(MAXSIM_REG_VERSION, &version) != 0 || version != MAXSIM_VERSION) {
      fprintf(stderr, "[VXDRV] the kernel's register map is version %u, this runtime's %u\n",
              version, MAXSIM_VERSION);
      return -1;
    }
    // The memory model: a latency-bandwidth pipe, FASED's defaults unless set.
    const uint32_t rlat = env_or("MAXSIM_MEM_READ_LATENCY", 30);
    const uint32_t wlat = env_or("MAXSIM_MEM_WRITE_LATENCY", 30);
    const uint32_t maxr = env_or("MAXSIM_MEM_MAX_READS", 16);
    const uint32_t maxw = env_or("MAXSIM_MEM_MAX_WRITES", 16);
    if (reg_write(MAXSIM_REG_MEM_BASE_LO, uint32_t(kMemBase)) != 0 ||
        reg_write(MAXSIM_REG_MEM_BASE_HI, uint32_t(kMemBase >> 32)) != 0 ||
        reg_write(MAXSIM_REG_MEM_RLAT, rlat) != 0 ||
        reg_write(MAXSIM_REG_MEM_WLAT, wlat) != 0 ||
        reg_write(MAXSIM_REG_MEM_MAXR, maxr) != 0 ||
        reg_write(MAXSIM_REG_MEM_MAXW, maxw) != 0 ||
        reg_write(MAXSIM_REG_CTRL, 1) != 0)
      return -1;
    // A target left running by an earlier process is reset, as a card's
    // native driver resets its device on open: Vortex counts VX_CFG_RESET_DELAY
    // cycles of reset before it takes a request.
    if (command(MAXSIM_CMD_RESET, 0, 64) != 0 || wait_idle() != 0)
      return -1;
    open_ = true;
    return 0;
  }

  // ----- CP register channel -----
  // No hardware CP: the functional model answers, ticked around each access.
  int cp_reg_write(uint32_t off, uint32_t value) {
    cp_.mmio_write(off, value);
    for (int i = 0; i < 256 && cp_.busy(); ++i)
      cp_.tick();
    return 0;
  }

  int cp_reg_read(uint32_t off, uint32_t* value) {
    for (int i = 0; i < 256 && cp_.busy(); ++i)
      cp_.tick();
    *value = cp_.mmio_read(off);
    return 0;
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
    const uint64_t target = read64(MAXSIM_REG_TCYCLES_LO);
    const uint64_t host = read64(MAXSIM_REG_HCYCLES_LO);
    const uint64_t stall = read64(MAXSIM_REG_MSTALL_LO);
    fprintf(stderr,
            "[maxsim] target cycles=%lu host cycles=%lu host/target=%.2f memory waits=%lu\n",
            (unsigned long)target, (unsigned long)host,
            target ? double(host) / double(target) : 0.0, (unsigned long)stall);
    // Each class finishes each of its threads once a target cycle, so its
    // finishes per host cycle is how busy its one physical copy is.
    uint32_t classes = 0;
    transport_.reg_read(MAXSIM_REG_NUM_CLASSES, &classes);
    std::string line;
    for (uint32_t c = 0; c < classes && host != 0; ++c) {
      const uint64_t finishes = read64(MAXSIM_REG_FINISH(c));
      char buf[64];
      snprintf(buf, sizeof(buf), " %u:%.1f%%", c, 100.0 * double(finishes) / double(host));
      line += buf;
    }
    if (!line.empty())
      fprintf(stderr, "[maxsim] class busy:%s\n", line.c_str());
    const auto& st = transport_.stats;
    fprintf(stderr,
            "[maxsim] host: %lu register reads, %lu writes in %.3f s; %lu bytes of device memory in %.3f s\n",
            (unsigned long)st.reads, (unsigned long)st.writes, st.reg_s,
            (unsigned long)st.dma_bytes, st.dma_s);
  }

  int status(uint32_t* value) { return transport_.reg_read(MAXSIM_REG_STATUS, value); }

  // Queues a command once the queue has room.
  int command(uint32_t op, uint32_t addr, uint32_t data) {
    uint32_t s = 0;
    do {
      if (status(&s) != 0)
        return -1;
    } while (s & MAXSIM_STATUS_FULL);
    if (reg_write(MAXSIM_REG_CMD_ADDR, addr) != 0 || reg_write(MAXSIM_REG_CMD_DATA, data) != 0 ||
        reg_write(MAXSIM_REG_CMD, op) != 0)
      return -1;
    return 0;
  }

  int wait_idle() {
    uint32_t s = 0;
    do {
      if (status(&s) != 0)
        return -1;
    } while (s & MAXSIM_STATUS_BUSY);
    return 0;
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
      transport_.mem_read(addr, dst, bytes);
    };
    h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(hp, src, bytes);
        return;
      }
      transport_.mem_write(addr, src, bytes);
    };
    h.vortex_dcr_write = [this](uint32_t addr, uint32_t value) {
      command(MAXSIM_CMD_DCR_WRITE, addr, value);
    };
    h.vortex_dcr_read = [this](uint32_t addr, uint32_t tag) -> uint32_t {
      if (command(MAXSIM_CMD_DCR_READ, addr, tag) != 0)
        return 0;
      uint32_t s = 0, value = 0;
      do {
        if (status(&s) != 0)
          return 0;
      } while (!(s & MAXSIM_STATUS_RSP));
      transport_.reg_read(MAXSIM_REG_DCR_RSP, &value);
      return value;
    };
    h.vortex_start = [this]() { command(MAXSIM_CMD_START, 0, 0); };
    h.vortex_busy = [this]() -> bool {
      uint32_t s = 0;
      if (status(&s) != 0)
        return false;
      return (s & MAXSIM_STATUS_BUSY) != 0;
    };
#ifdef VX_CFG_VM_ENABLE
    h.mmu_fault_report = true;
#endif
    return h;
  }

  counted_transport_t transport_;
  bool open_ = false;
  vortex::CommandProcessor cp_;
  std::mutex host_mu_;
  std::map<uint64_t, uint64_t> host_regions_;
};

#include <callbacks.inc>
