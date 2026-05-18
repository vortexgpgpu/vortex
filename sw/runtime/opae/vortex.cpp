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

#include <common.h>

#include "driver.h"

#include <vortex_opae.h>

#ifdef SCOPE
#include "scope.h"
#endif

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unordered_map>
#include <uuid/uuid.h>

using namespace vortex;

#define CMD_MEM_READ     AFU_IMAGE_CMD_MEM_READ
#define CMD_MEM_WRITE    AFU_IMAGE_CMD_MEM_WRITE
#define CMD_RUN          AFU_IMAGE_CMD_RUN
#define CMD_DCR_WRITE    AFU_IMAGE_CMD_DCR_WRITE
#define CMD_DCR_READ     AFU_IMAGE_CMD_DCR_READ

#define MMIO_CMD_TYPE    (AFU_IMAGE_MMIO_CMD_TYPE * 4)
#define MMIO_CMD_ARG0    (AFU_IMAGE_MMIO_CMD_ARG0 * 4)
#define MMIO_CMD_ARG1    (AFU_IMAGE_MMIO_CMD_ARG1 * 4)
#define MMIO_CMD_ARG2    (AFU_IMAGE_MMIO_CMD_ARG2 * 4)
#define MMIO_STATUS      (AFU_IMAGE_MMIO_STATUS * 4)
#define MMIO_DEV_CAPS    (AFU_IMAGE_MMIO_DEV_CAPS * 4)
#define MMIO_ISA_CAPS    (AFU_IMAGE_MMIO_ISA_CAPS * 4)
#define MMIO_DCR_RSP     (AFU_IMAGE_MMIO_DCR_RSP * 4)
#define MMIO_SCOPE_READ  (AFU_IMAGE_MMIO_SCOPE_READ * 4)
#define MMIO_SCOPE_WRITE (AFU_IMAGE_MMIO_SCOPE_WRITE * 4)

#define STATUS_STATE_BITS 8

// ----- Command Processor regfile (host byte addresses) -----
// The AFU's MMIO demux routes byte addresses 0x1000..0x1FFF to the CP
// regfile (mapped to CP's native 0x000-based 12-bit address space).
// Same bit-12 split as the XRT integration; see VX_cp_axil_regfile §17.4.
#define CP_BASE              0x1000
#define CP_REG_CTRL          (CP_BASE + 0x000)   // bit0 = enable_global
#define CP_REG_STATUS        (CP_BASE + 0x004)
#define CP_REG_DEV_CAPS      (CP_BASE + 0x008)
#define CP_Q_RING_BASE_LO    (CP_BASE + 0x100)
#define CP_Q_RING_BASE_HI    (CP_BASE + 0x104)
#define CP_Q_HEAD_ADDR_LO    (CP_BASE + 0x108)
#define CP_Q_HEAD_ADDR_HI    (CP_BASE + 0x10C)
#define CP_Q_CMPL_ADDR_LO    (CP_BASE + 0x110)
#define CP_Q_CMPL_ADDR_HI    (CP_BASE + 0x114)
#define CP_Q_RING_SIZE_LOG2  (CP_BASE + 0x118)
#define CP_Q_CONTROL         (CP_BASE + 0x11C)
#define CP_Q_TAIL_LO         (CP_BASE + 0x120)
#define CP_Q_TAIL_HI         (CP_BASE + 0x124)
#define CP_Q_SEQNUM          (CP_BASE + 0x128)
#define CP_Q_ERROR           (CP_BASE + 0x12C)

#define CP_RING_SIZE_LOG2    16          // 64 KiB
#define CP_RING_SIZE         (1u << CP_RING_SIZE_LOG2)
#define CP_OPCODE_LAUNCH     0x06
#define CP_LAUNCH_BYTES      12          // 4-byte header + 8-byte arg0

#define CHECK_HANDLE(handle, _expr, _cleanup)                                  \
  auto handle = _expr;                                                         \
  if (handle == nullptr) {                                                     \
    printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr);                    \
    _cleanup                                                                   \
  }

#define CHECK_FPGA_ERR(_expr, _cleanup)                                        \
  do {                                                                         \
    auto err = _expr;                                                          \
    if (err == 0)                                                              \
      break;                                                                   \
    printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err,         \
           api_.fpgaErrStr(err));                                              \
    _cleanup                                                                   \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
    : fpga_(nullptr)
    , global_mem_(ALLOC_BASE_ADDR,
                  GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                  RAM_PAGE_SIZE,
                  CACHE_BLOCK_SIZE)
    , staging_wsid_(0)
    , staging_ioaddr_(0)
    , staging_ptr_(nullptr)
    , staging_size_(0)
    , clock_rate_(0)
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
    if (fpga_ != nullptr) {
      if (staging_size_ != 0) {
        api_.fpgaReleaseBuffer(fpga_, staging_wsid_);
        staging_size_ = 0;
      }
      api_.fpgaClose(fpga_);
    }
    drv_close();
  }

  int init() {
    fpga_token accel_token;
    fpga_properties filter;
    fpga_guid guid;
    uint32_t num_matches;

    memset(&api_, 0, sizeof(opae_drv_api_t));
    if (drv_init(&api_) != 0) {
      return -1;
    }

    // Set up a filter that will search for an accelerator
    CHECK_FPGA_ERR(api_.fpgaGetProperties(nullptr, &filter), {
      return -1;
    });

    CHECK_FPGA_ERR(api_.fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Add the desired UUID to the filter
    uuid_parse(AFU_ACCEL_UUID_S, guid);
    CHECK_FPGA_ERR(api_.fpgaPropertiesSetGUID(filter, guid), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Do the search across the available FPGA contexts
    CHECK_FPGA_ERR(api_.fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Not needed anymore
    CHECK_FPGA_ERR(api_.fpgaDestroyProperties(&filter), {
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    });

    if (num_matches < 1) {
      fprintf(stderr, "[VXDRV] Error: accelerator %s not found!\n", AFU_ACCEL_UUID_S);
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    }

    // Open accelerator
    CHECK_FPGA_ERR(api_.fpgaOpen(accel_token, &fpga_, 0), {
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    });

    // Done with token
    CHECK_FPGA_ERR(api_.fpgaDestroyToken(&accel_token), {
      api_.fpgaClose(fpga_);
      return -1;
    });

    {
      // Load ISA CAPS
      CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_ISA_CAPS, &isa_caps_), {
        api_.fpgaClose(fpga_);
        return -1;
      });

      // Load device CAPS
      CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_DEV_CAPS, &dev_caps_), {
        api_.fpgaClose(fpga_);
        return -1;
      });

      // Determine global memory size
      uint64_t num_banks, bank_size;
      this->get_caps(VX_CAPS_NUM_MEM_BANKS, &num_banks);
      this->get_caps(VX_CAPS_MEM_BANK_SIZE, &bank_size);
      global_mem_size_ = num_banks * bank_size;

      // Query actual FPGA clock rate; average high and low user clocks
      {
        uint64_t clk_high = 0, clk_low = 0;
        if (api_.fpgaGetUserClock(fpga_, &clk_high, &clk_low, 0) == FPGA_OK) {
          clock_rate_ = (clk_high + clk_low) / 2; // in MHz
        }
      }
    }

  #ifdef SCOPE
    {
      scope_callback_t callback;
      callback.registerWrite = [](vx_device_h hdevice, uint64_t value) -> int {
        auto device = (vx_device *)hdevice;
        return device->api_.fpgaWriteMMIO64(device->fpga_, 0, MMIO_SCOPE_WRITE, value);
      };

      callback.registerRead = [](vx_device_h hdevice, uint64_t *value) -> int {
        auto device = (vx_device *)hdevice;
        return device->api_.fpgaReadMMIO64(device->fpga_, 0, MMIO_SCOPE_READ, value);
      };

      CHECK_ERR(vx_scope_start(&callback, this, -1, -1), {
        api_.fpgaClose(fpga_);
        return err;
      });
    }
  #endif

    {
      // Honour common boolean conventions: empty, "0", "false", "no", "off"
      // all leave CP disabled; everything else enables it.
      const char* env = getenv("VORTEX_USE_CP");
      auto is_truthy = [](const char* s) {
        if (s == nullptr || s[0] == '\0') return false;
        if (s[0] == '0' && s[1] == '\0') return false;
        std::string v(s);
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v != "false" && v != "no" && v != "off";
      };
      if (is_truthy(env)) {
        CHECK_ERR(this->cp_init(), { return err; });
      }
    }

    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t * value) {
    uint64_t _value;
    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = (dev_caps_ >> 0) & 0xff;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = 1 << ((dev_caps_ >> 8) & 0x7);
      break;
    case VX_CAPS_NUM_WARPS:
      _value = 1 << ((dev_caps_ >> 11) & 0x7);
      break;
    case VX_CAPS_NUM_CORES: {
      uint32_t socket_size  = 1 << ((dev_caps_ >> 14) & 0x7);
      uint32_t cluster_size = 1 << ((dev_caps_ >> 17) & 0x7);
      uint32_t num_clusters = 1 << ((dev_caps_ >> 20) & 0x7);
      _value = num_clusters * cluster_size * socket_size;
    } break;
    case VX_CAPS_SOCKET_SIZE:
      _value = 1 << ((dev_caps_ >> 14) & 0x7);
      break;
    case VX_CAPS_NUM_CLUSTERS:
      _value = 1 << ((dev_caps_ >> 20) & 0x7);
      break;
    case VX_CAPS_ISSUE_WIDTH:
      _value = 1 << ((dev_caps_ >> 23) & 0x7);
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = global_mem_size_;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = 1ull << ((dev_caps_ >> 26) & 0xff);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = isa_caps_;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = 1 << ((dev_caps_ >> 34) & 0x7);
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (20 + ((dev_caps_ >> 37) & 0x1f));
      break;
    case VX_CAPS_CLOCK_RATE:
      _value = clock_rate_;
      break;
    case VX_CAPS_PEAK_MEM_BW:
      _value = PLATFORM_MEMORY_PEAK_BW;
      break;
    default:
      fprintf(stderr, "[VXDRV] Error: invalid caps id: %d\n", caps_id);
      std::abort();
      return -1;
    }

    *value = _value;

    return 0;
  }

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
    uint64_t addr;
    CHECK_ERR(global_mem_.allocate(size, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, size, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
    CHECK_ERR(global_mem_.reserve(dev_addr, size), {
      return err;
    });
    CHECK_ERR(this->mem_access(dev_addr, size, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
    return 0;
  }

  int mem_free(uint64_t dev_addr) {
    return global_mem_.release(dev_addr);
  }

  int mem_access(uint64_t /*dev_addr*/, uint64_t /*size*/, int /*flags*/) {
    return 0;
  }

  int mem_info(uint64_t * mem_free, uint64_t * mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size){
    if( dest_addr == src_addr) {
      return 0;
    }

    if (dest_addr + size > global_mem_size_ ||
        src_addr + size > global_mem_size_)
      return -1;

    CHECK_FPGA_ERR(api_.fpgaCopyBuffer(fpga_, dest_addr, src_addr, size), {
      return -1;
    });
    return 0;
  }

  int upload(uint64_t dev_addr, const void *host_ptr, uint64_t size) {
    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
      return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > global_mem_size_)
      return -1;

    // ensure ready for new command
    if (this->ready_wait(VX_MAX_TIMEOUT) != 0)
      return -1;

    if (this->ensure_staging(asize) != 0)
      return -1;

    // update staging buffer
    memcpy(staging_ptr_, host_ptr, size);

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG0, staging_ioaddr_ >> ls_shift), {
      return -1;
    });

    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG1, dev_addr >> ls_shift), {
      return -1;
    });

    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
      return -1;
    });

    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_TYPE, CMD_MEM_WRITE), {
      return -1;
    });

    // Wait for the write operation to finish
    if (this->ready_wait(VX_MAX_TIMEOUT) != 0)
      return -1;

    return 0;
  }

  int download(void *host_ptr, uint64_t dev_addr, uint64_t size) {
    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
      return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > global_mem_size_)
      return -1;

    // flush GPU caches before reading back results
    {
      uint64_t num_cores;
      CHECK_ERR(this->get_caps(VX_CAPS_NUM_CORES, &num_cores), { return err; });
      uint32_t dummy;
      for (uint32_t cid = 0; cid < (uint32_t)num_cores; ++cid) {
        CHECK_ERR(this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy), { return err; });
      }
    }

    // ensure ready for new command
    if (this->ready_wait(VX_MAX_TIMEOUT) != 0)
      return -1;

    if (this->ensure_staging(asize) != 0)
      return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG0, staging_ioaddr_ >> ls_shift), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG1, dev_addr >> ls_shift), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_TYPE, CMD_MEM_READ), {
      return -1;
    });

    // Wait for the write operation to finish
    if (this->ready_wait(VX_MAX_TIMEOUT) != 0)
      return -1;

    // read staging buffer
    memcpy(host_ptr, staging_ptr_, size);

    return 0;
  }

  int start() {
    // DCRs already written by stub; just trigger execution
    if (cp_enabled_) return this->cp_post_launch();
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_TYPE, CMD_RUN), {
      return -1;
    });
    return 0;
  }

  int ready_wait(uint64_t timeout) {
    if (cp_enabled_) return this->cp_wait(timeout);
    std::unordered_map<uint32_t, std::stringstream> print_bufs;

    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;

    // to milliseconds
    uint64_t sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);

    for (;;) {
      uint64_t status;
      CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_STATUS, &status), {
        return -1;
      });

      // check for console data
      uint32_t cout_data = status >> STATUS_STATE_BITS;
      if (cout_data & 0x1) {
        // retrieve console data
        do {
          char cout_char = (cout_data >> 1) & 0xff;
          uint32_t cout_tid = (cout_data >> 9) & 0xff;
          auto &ss_buf = print_bufs[cout_tid];
          ss_buf << cout_char;
          if (cout_char == '\n') {
            std::cout << std::dec << "#" << cout_tid << ": " << ss_buf.str() << std::flush;
            ss_buf.str("");
          }
          CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_STATUS, &status), {
            return -1;
          });
          cout_data = status >> STATUS_STATE_BITS;
        } while (cout_data & 0x1);
      }

      uint32_t state = status & ((1 << STATUS_STATE_BITS) - 1);

      if (0 == state || 0 == timeout) {
        for (auto &buf : print_bufs) {
          auto str = buf.second.str();
          if (!str.empty()) {
            std::cout << "#" << buf.first << ": " << str << std::endl;
          }
        }
        if (state != 0) {
          fprintf(stdout, "[VXDRV] ready-wait timed out: state=%d\n", state);
          return -1;
        }
        break;
      }

      nanosleep(&sleep_time, nullptr);
      timeout -= sleep_time_ms;
    };

    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG0, addr), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG1, value), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_TYPE, CMD_DCR_WRITE), {
      return -1;
    });
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t * value) {
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG0, addr), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_ARG1, tag), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, MMIO_CMD_TYPE, CMD_DCR_READ), {
      return -1;
    });
    // ensure ready for new command
    if (this->ready_wait(VX_MAX_TIMEOUT) != 0)
      return -1;
    // read back the captured DCR response
    uint64_t rsp;
    CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_DCR_RSP, &rsp), {
      return -1;
    });
    *value = (uint32_t)rsp;
    return 0;
  }

  // ----- CP MMIO surface -----
  // The AFU's MMIO demux routes host byte offsets 0x1000..0x1FFF to the
  // CP regfile (mapped to CP-internal 0x000-based offsets). Callers
  // pass the CP-internal offset directly; we add the AFU base here.
  int cp_mmio_write(uint32_t off, uint32_t value) {
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, CP_BASE + off, value), {
      return -1;
    });
    return 0;
  }

  int cp_mmio_read(uint32_t off, uint32_t* value) {
    uint64_t v = 0;
    CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, CP_BASE + off, &v), {
      return -1;
    });
    *value = uint32_t(v);
    return 0;
  }

  // ----- Command Processor path -----
  // Same shape as the XRT runtime's cp_init / cp_post_launch / cp_wait
  // — allocate ring + head + completion buffers in device memory, program
  // CP queue 0 via the CP regfile (MMIO byte 0x1000+), then on each
  // vx_start() push a CMD_LAUNCH descriptor into the ring + commit Q_TAIL
  // and poll Q_SEQNUM until the engine retires it.
  int cp_init() {
    CHECK_ERR(this->mem_alloc(CP_RING_SIZE, VX_MEM_READ, &cp_ring_dev_addr_), { return err; });
    CHECK_ERR(this->mem_alloc(CACHE_BLOCK_SIZE, VX_MEM_WRITE, &cp_head_dev_addr_), { return err; });
    CHECK_ERR(this->mem_alloc(CACHE_BLOCK_SIZE, VX_MEM_WRITE, &cp_cmpl_dev_addr_), { return err; });

    std::vector<uint8_t> zeros_cl(CACHE_BLOCK_SIZE, 0);
    std::vector<uint8_t> zeros_ring(CP_RING_SIZE, 0);
    CHECK_ERR(this->upload(cp_ring_dev_addr_, zeros_ring.data(), CP_RING_SIZE), { return err; });
    CHECK_ERR(this->upload(cp_head_dev_addr_, zeros_cl.data(), CACHE_BLOCK_SIZE), { return err; });
    CHECK_ERR(this->upload(cp_cmpl_dev_addr_, zeros_cl.data(), CACHE_BLOCK_SIZE), { return err; });

    auto wr = [this](uint32_t off, uint32_t val) -> int {
      CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, off, val), { return -1; });
      return 0;
    };

    CHECK_ERR(wr(CP_Q_RING_BASE_LO,   (uint32_t)(cp_ring_dev_addr_ & 0xFFFFFFFFu)), { return err; });
    CHECK_ERR(wr(CP_Q_RING_BASE_HI,   (uint32_t)(cp_ring_dev_addr_ >> 32)),         { return err; });
    CHECK_ERR(wr(CP_Q_HEAD_ADDR_LO,   (uint32_t)(cp_head_dev_addr_ & 0xFFFFFFFFu)), { return err; });
    CHECK_ERR(wr(CP_Q_HEAD_ADDR_HI,   (uint32_t)(cp_head_dev_addr_ >> 32)),         { return err; });
    CHECK_ERR(wr(CP_Q_CMPL_ADDR_LO,   (uint32_t)(cp_cmpl_dev_addr_ & 0xFFFFFFFFu)), { return err; });
    CHECK_ERR(wr(CP_Q_CMPL_ADDR_HI,   (uint32_t)(cp_cmpl_dev_addr_ >> 32)),         { return err; });
    CHECK_ERR(wr(CP_Q_RING_SIZE_LOG2, CP_RING_SIZE_LOG2),                            { return err; });
    CHECK_ERR(wr(CP_Q_CONTROL,        0x1),                                          { return err; });
    CHECK_ERR(wr(CP_REG_CTRL,         0x1),                                          { return err; });

    cp_enabled_         = true;
    cp_tail_            = 0;
    cp_expected_seqnum_ = 0;

    printf("info: CP enabled — ring=0x%lx head=0x%lx cmpl=0x%lx\n",
           cp_ring_dev_addr_, cp_head_dev_addr_, cp_cmpl_dev_addr_);
    return 0;
  }

  int cp_post_launch() {
    uint8_t cl[CACHE_BLOCK_SIZE] = {0};
    cl[0] = CP_OPCODE_LAUNCH;

    uint64_t ring_offset = cp_tail_ & (CP_RING_SIZE - 1);
    if (ring_offset + CACHE_BLOCK_SIZE > CP_RING_SIZE) {
      fprintf(stderr, "[VXDRV] CP ring wraparound mid-CL not yet supported\n");
      return -1;
    }
    CHECK_ERR(this->upload(cp_ring_dev_addr_ + ring_offset, cl, CACHE_BLOCK_SIZE), { return err; });

    cp_tail_           += CP_LAUNCH_BYTES;
    cp_expected_seqnum_ += 1;
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, CP_Q_TAIL_LO,
                                        (uint32_t)(cp_tail_ & 0xFFFFFFFFu)), { return -1; });
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, CP_Q_TAIL_HI,
                                        (uint32_t)(cp_tail_ >> 32)),         { return -1; });
    return 0;
  }

  int cp_wait(uint64_t timeout) {
    // Poll Q_SEQNUM via MMIO read until the engine retires the command —
    // see the XRT runtime's cp_wait for the rationale (xrtBOSync / opae
    // BO sync don't tick the simulated clock; only register traffic does).
    for (;;) {
      uint64_t seqnum64 = 0;
      CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, CP_Q_SEQNUM, &seqnum64), { return -1; });
      uint32_t seqnum32 = (uint32_t)seqnum64;
      if ((uint64_t)seqnum32 >= cp_expected_seqnum_) break;
      if (0 == timeout) return -1;
      timeout -= 1;
    }
    // Engine retired (Phase 2b shortcut: on KMU grant, not actual Vortex
    // completion). Wait for the AFU FSM to drop back to STATE_IDLE — the
    // saw_busy guard ensures this only fires after Vortex really finished.
    // No hard spin cap: each MMIO read ticks the sim a handful of cycles,
    // and sgemm-class kernels need many more than a fixed cap allows.
    for (;;) {
      uint64_t status;
      CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, MMIO_STATUS, &status), { return -1; });
      uint32_t state = status & ((1 << STATUS_STATE_BITS) - 1);
      if (state == 0) break;
      if (0 == timeout) return -1;
      timeout -= 1;
    }
    return 0;
  }


private:

  int ensure_staging(uint64_t size) {
    if (staging_size_ >= size)
      return 0;

    if (staging_size_ != 0) {
      api_.fpgaReleaseBuffer(fpga_, staging_wsid_);
      staging_size_ = 0;
    }

    // allocate new buffer
    CHECK_FPGA_ERR(api_.fpgaPrepareBuffer(fpga_, size, (void **)&staging_ptr_, &staging_wsid_, 0), {
      return -1;
    });

    // get the physical address of the buffer in the accelerator
    CHECK_FPGA_ERR(api_.fpgaGetIOAddress(fpga_, staging_wsid_, &staging_ioaddr_), {
      api_.fpgaReleaseBuffer(fpga_, staging_wsid_);
      return -1;
    });

    staging_size_ = size;

    return 0;
  }

  opae_drv_api_t api_;
  fpga_handle fpga_;
  MemoryAllocator global_mem_;
  uint64_t dev_caps_;
  uint64_t isa_caps_;
  uint64_t global_mem_size_;
  uint64_t staging_wsid_;
  uint64_t staging_ioaddr_;
  uint8_t* staging_ptr_;
  uint64_t staging_size_;
  uint64_t clock_rate_;

  // Command Processor state (populated by cp_init() when VORTEX_USE_CP=1).
  bool     cp_enabled_         = false;
  uint64_t cp_ring_dev_addr_   = 0;
  uint64_t cp_head_dev_addr_   = 0;
  uint64_t cp_cmpl_dev_addr_   = 0;
  uint64_t cp_tail_            = 0;
  uint64_t cp_expected_seqnum_ = 0;
};

#include <callbacks.inc>