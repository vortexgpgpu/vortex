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

#include <vortex.h>
#include <malloc.h>
#include <utils.h>
#include <VX_config.h>
#include <VX_types.h>
#include <stdarg.h>
#include <util.h>
#include <limits>
#include <vector>
#include <string>
#include <unordered_map>
#include <array>

#ifdef SCOPE
#include "scope.h"
#endif

// XRT includes
#ifndef XRTSIM
#include "experimental/xrt_bo.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#include "experimental/xrt_error.h"
#else
#include <fpga.h>
#endif

using namespace vortex;

#ifndef XRTSIM
#define CPP_API
#endif

//#define BANK_INTERLEAVE

#define MMIO_CTL_ADDR   0x00
#define MMIO_DEV_ADDR   0x10
#define MMIO_ISA_ADDR   0x1C
#define MMIO_DCR_ADDR   0x28
#define MMIO_SCP_ADDR   0x34
#define MMIO_MEM_ADDR   0x40

#define CTL_AP_START    (1<<0)
#define CTL_AP_DONE     (1<<1)
#define CTL_AP_IDLE     (1<<2)
#define CTL_AP_READY    (1<<3)
#define CTL_AP_RESET    (1<<4)
#define CTL_AP_RESTART  (1<<7)

struct platform_info_t {
    const char* prefix_name;
    uint8_t     lg2_num_banks;
    uint8_t     lg2_bank_size;
    uint64_t    mem_base;
};

static const platform_info_t g_platforms [] = {
    {"vortex_xrtsim",  4, 0x10, 0x0}, // 64 KB banks
    {"xilinx_u50",     4, 0x1C, 0x0}, // 16 MB banks
    {"xilinx_u200",    4, 0x1C, 0x0}, // 16 MB banks
    {"xilinx_u280",    4, 0x1C, 0x0}, // 16 MB banks
    {"xilinx_vck5000", 0, 0x21, 0xC000000000},
};

#ifdef CPP_API

    typedef xrt::device xrt_device_t;
    typedef xrt::ip xrt_kernel_t;
    typedef xrt::bo xrt_buffer_t;

#else

    typedef xrtDeviceHandle xrt_device_t;
    typedef xrtKernelHandle xrt_kernel_t;
    typedef xrtBufferHandle xrt_buffer_t;

#endif

#define RAM_PAGE_SIZE 4096

#define DEFAULT_DEVICE_INDEX 0

#define DEFAULT_XCLBIN_PATH "vortex_afu.xclbin"

#define KERNEL_NAME "vortex_afu"

#ifndef NDEBUG
#define DBGPRINT(format, ...) do { printf("[VXDRV] " format "", ##__VA_ARGS__); } while (0)
#else
#define DBGPRINT(format, ...) ((void)0)
#endif

#define CHECK_HANDLE(handle, _expr, _cleanup)   \
    auto handle = _expr;                        \
    if (handle == nullptr) {                    \
        printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr); \
        _cleanup                                \
    }

#define CHECK_ERR(_expr, _cleanup)              \
    do {                                        \
        auto err = _expr;                       \
        if (err == 0)                           \
            break;                              \
        printf("[VXDRV] Error: '%s' returned %d!\n", #_expr, (int)err); \
        _cleanup                                \
    } while (false)

#ifndef CPP_API

static void dump_xrt_error(xrtDeviceHandle xrtDevice, xrtErrorCode err) {
    size_t len = 0;
    xrtErrorGetString(xrtDevice, err, nullptr, 0, &len);
    std::vector<char> buf(len);
    xrtErrorGetString(xrtDevice, err, buf.data(), buf.size(), nullptr);
    printf("[VXDRV] detail: %s!\n", buf.data());
}

#endif

static int get_platform_info(const std::string& device_name, platform_info_t* platform_info) {
    for (size_t i = 0; i < (sizeof(g_platforms)/sizeof(platform_info_t)); ++i) {
        auto& platform = g_platforms[i];
        if (device_name.rfind(platform.prefix_name, 0) == 0) {
            *platform_info = platform;
            return 0;
        }
    }
    return -1;
}

/*static void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}*/

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:

    vx_device(xrt_device_t& device, xrt_kernel_t& kernel, const platform_info_t& platform)
        : xrtDevice_(device)
        , xrtKernel_(kernel)
        , platform_(platform)
        , global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, RAM_PAGE_SIZE, CACHE_BLOCK_SIZE)
    {}

#ifndef CPP_API

    ~vx_device() {
        profiling_remove(profiling_id_);
        for (auto& entry : xrtBuffers_) {
        #ifdef BANK_INTERLEAVE
            xrtBOFree(entry);
        #else
            xrtBOFree(entry.second.xrtBuffer);
        #endif
        }
        if (xrtKernel_) {
            xrtKernelClose(xrtKernel_);
        }
        if (xrtDevice_) {
            xrtDeviceClose(xrtDevice_);
        }
    }

#endif

    int init() {
        CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_RESET), {
            return err;
        });

        uint32_t num_banks = 1 << platform_.lg2_num_banks;
        uint64_t bank_size = 1ull << platform_.lg2_bank_size;

        for (uint32_t i = 0; i < num_banks; ++i) {
            uint32_t reg_addr = MMIO_MEM_ADDR + (i * 12);
            uint64_t reg_value = platform_.mem_base + i * bank_size;
            CHECK_ERR(this->write_register(reg_addr, reg_value & 0xffffffff), {
                return err;
            });

            CHECK_ERR(this->write_register(reg_addr + 4, (reg_value >> 32) & 0xffffffff), {
                return err;
            });
        #ifndef BANK_INTERLEAVE
            break;
        #endif
        }

        CHECK_ERR(this->read_register(MMIO_DEV_ADDR, (uint32_t*)&dev_caps_), {
            return err;
        });

        CHECK_ERR(this->read_register(MMIO_DEV_ADDR + 4, (uint32_t*)&dev_caps_ + 1), {
            return err;
        });

        CHECK_ERR(this->read_register(MMIO_ISA_ADDR, (uint32_t*)&isa_caps_), {
            return err;
        });

        CHECK_ERR(this->read_register(MMIO_ISA_ADDR + 4, (uint32_t*)&isa_caps_ + 1), {
            return err;
        });

        global_mem_size_ = num_banks * bank_size;

    #ifdef BANK_INTERLEAVE
        xrtBuffers_.reserve(num_banks);
        for (uint32_t i = 0; i < num_banks; ++i) {
        #ifdef CPP_API
            xrtBuffers_.emplace_back(xrtDevice_, bank_size, xrt::bo::flags::normal, i);
        #else
            CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice_, bank_size, XRT_BO_FLAGS_NONE, i), {
                return -1;
            });
            xrtBuffers_.push_back(xrtBuffer);
        #endif
            printf("*** allocated bank%u/%u, size=%lu\n", i, num_banks, bank_size);
        }
    #endif

        CHECK_ERR(dcr_initialize(this), {
            return err;
        });

        profiling_id_ = profiling_add(this);

        return 0;
    }

    int get_caps(uint32_t caps_id, uint64_t *value) {
        uint64_t _value;

        switch (caps_id) {
        case VX_CAPS_VERSION:
            _value = (dev_caps_ >> 0) & 0xff;
            break;
        case VX_CAPS_NUM_THREADS:
            _value = (dev_caps_ >> 8) & 0xff;
            break;
        case VX_CAPS_NUM_WARPS:
            _value = (dev_caps_ >> 16) & 0xff;
            break;
        case VX_CAPS_NUM_CORES:
            _value = (dev_caps_ >> 24) & 0xffff;
            break;
        case VX_CAPS_NUM_BARRIERS:
            _value = (dev_caps_ >> 40) & 0xff;
            break;
        case VX_CAPS_CACHE_LINE_SIZE:
            _value = CACHE_BLOCK_SIZE;
            break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
            _value = global_mem_size_;
            break;
        case VX_CAPS_LOCAL_MEM_SIZE:
            _value = 1ull << ((dev_caps_ >> 48) & 0xff);
            break;
        case VX_CAPS_LOCAL_MEM_ADDR:
            _value = LMEM_BASE_ADDR;
            break;
        case VX_CAPS_ISA_FLAGS:
            _value = isa_caps_;
            break;
        default:
            fprintf(stderr, "[VXDRV] Error: invalid caps id: %d\n", caps_id);
            std::abort();
            return -1;
        }

        *value = _value;

        return 0;
    }

    int mem_alloc(uint64_t size, int flags, uint64_t* dev_addr) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        uint64_t addr;
        CHECK_ERR(global_mem_.allocate(asize, &addr), {
            return err;
        });
    #ifndef BANK_INTERLEAVE
        uint32_t bank_id;
        CHECK_ERR(this->get_bank_info(addr, &bank_id, nullptr), {
            global_mem_.release(addr);
            return err;
        });
        CHECK_ERR(get_buffer(bank_id, nullptr), {
            global_mem_.release(addr);
            return err;
        });
    #endif
        CHECK_ERR(this->mem_access(addr, size, flags), {
            global_mem_.release(addr);
            return err;
        });
        *dev_addr = addr;
        return 0;
    }

    int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {\
        CHECK_ERR(global_mem_.reserve(dev_addr, size), {
            return err;
        });
    #ifndef BANK_INTERLEAVE
        uint32_t bank_id;
        CHECK_ERR(this->get_bank_info(dev_addr, &bank_id, nullptr), {
            global_mem_.release(dev_addr);
            return err;
        });
        CHECK_ERR(get_buffer(bank_id, nullptr), {
            global_mem_.release(dev_addr);
            return err;
        });
    #endif
        CHECK_ERR(this->mem_access(dev_addr, size, flags), {
            global_mem_.release(dev_addr);
            return err;
        });
        return 0;
    }

    int mem_free(uint64_t dev_addr) {
        CHECK_ERR(global_mem_.release(dev_addr), {
            return err;
        });
    #ifdef BANK_INTERLEAVE
        if (0 == global_mem_.allocated()) {
        #ifndef CPP_API
            for (auto& entry : xrtBuffers_) {
                xrtBOFree(entry);
            }
        #endif
            xrtBuffers_.clear();
        }
    #else
        uint32_t bank_id;
        CHECK_ERR(this->get_bank_info(dev_addr, &bank_id, nullptr), {
            return err;
        });
        auto it = xrtBuffers_.find(bank_id);
        if (it != xrtBuffers_.end()) {
            auto count = --it->second.count;
            if (0 == count) {
                printf("freeing bank%d...\n", bank_id);
            #ifndef CPP_API
                xrtBOFree(it->second.xrtBuffer);
            #endif
                xrtBuffers_.erase(it);
            }
        } else {
            fprintf(stderr, "[VXDRV] Error: invalid device memory address: 0x%lx\n", dev_addr);
            return -1;
        }
    #endif
        return 0;
    }

    int mem_access(uint64_t /*dev_addr*/, uint64_t /*size*/, int /*flags*/) {
        return 0;
    }

    int mem_info(uint64_t* mem_free, uint64_t* mem_used) const {
        if (mem_free)
            *mem_free = global_mem_.free();
        if (mem_used)
            *mem_used = global_mem_.allocated();
        return 0;
    }

    int write_register(uint32_t addr, uint32_t value) {
    #ifdef CPP_API
        xrtKernel_.write_register(addr, value);
    #else
        CHECK_ERR(xrtKernelWriteRegister(xrtKernel_, addr, value), {
            dump_xrt_error(xrtDevice_, err);
            return err;
        });
    #endif
        DBGPRINT("*** write_register: addr=0x%x, value=0x%x\n", addr, value);
        return 0;
    }

    int read_register(uint32_t addr, uint32_t* value) {
    #ifdef CPP_API
        *value = xrtKernel_.read_register(addr);
    #else
        CHECK_ERR(xrtKernelReadRegister(xrtKernel_, addr, value), {
            dump_xrt_error(xrtDevice_, err);
            return err;
        });
    #endif
        DBGPRINT("*** read_register: addr=0x%x, value=0x%x\n", addr, *value);
        return 0;
    }

    int upload(uint64_t dev_addr, const void* src, uint64_t size) {
        auto host_ptr = (const uint8_t*)src;

        // check alignment
        if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
            return -1;

        auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

        // bound checking
        if (dev_addr + asize > global_mem_size_)
            return -1;

        for (uint64_t end = dev_addr + asize; dev_addr < end;
            dev_addr += CACHE_BLOCK_SIZE,
            host_ptr += CACHE_BLOCK_SIZE) {
        #ifdef BANK_INTERLEAVE
            asize = CACHE_BLOCK_SIZE;
        #else
            end = 0;
        #endif
            uint32_t bo_index;
            uint64_t bo_offset;
            xrt_buffer_t xrtBuffer;
            CHECK_ERR(this->get_bank_info(dev_addr, &bo_index, &bo_offset), {
                return err;
            });
            CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
                return err;
            });
        #ifdef CPP_API
            xrtBuffer.write(host_ptr, asize, bo_offset);
            xrtBuffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset);
        #else
            CHECK_ERR(xrtBOWrite(xrtBuffer, host_ptr, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return err;
            });
            CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return err;
            });
        #endif
        }
        return 0;
    }

    int download(void* dest, uint64_t dev_addr, uint64_t size) {
        auto host_ptr = (uint8_t*)dest;

        // check alignment
        if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
            return -1;

        auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

        // bound checking
        if (dev_addr + asize > global_mem_size_)
            return -1;

        for (uint64_t end = dev_addr + asize; dev_addr < end;
            dev_addr += CACHE_BLOCK_SIZE,
            host_ptr += CACHE_BLOCK_SIZE) {
        #ifdef BANK_INTERLEAVE
            asize = CACHE_BLOCK_SIZE;
        #else
            end = 0;
        #endif
            uint32_t bo_index;
            uint64_t bo_offset;
            xrt_buffer_t xrtBuffer;
            CHECK_ERR(this->get_bank_info(dev_addr, &bo_index, &bo_offset), {
                return err;
            });
            CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
                return err;
            });
        #ifdef CPP_API
            xrtBuffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset);
            xrtBuffer.read(host_ptr, asize, bo_offset);
        #else
            CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return err;
            });
            CHECK_ERR(xrtBORead(xrtBuffer, host_ptr, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return err;
            });
        #endif
        }
        return 0;
    }

    int start(uint64_t krnl_addr, uint64_t args_addr) {
        // set kernel info
        CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, krnl_addr & 0xffffffff), {
            return err;
        });
        CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, krnl_addr >> 32), {
            return err;
        });
        CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ARG0, args_addr & 0xffffffff), {
            return err;
        });
        CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ARG1, args_addr >> 32), {
            return err;
        });

        profiling_begin(profiling_id_);

        // start execution
        CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_START), {
            return err;
        });

        // clear mpm cache
        mpm_cache_.clear();

        return 0;
    }

    int ready_wait(uint64_t timeout) {
        struct timespec sleep_time;
    #ifndef NDEBUG
        sleep_time.tv_sec = 1;
        sleep_time.tv_nsec = 0;
    #else
        sleep_time.tv_sec = 0;
        sleep_time.tv_nsec = 1000000;
    #endif

        // to milliseconds
        uint64_t sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);

        for (;;) {
            uint32_t status = 0;
            CHECK_ERR(this->read_register(MMIO_CTL_ADDR, &status), {
                return err;
            });
            bool is_done = (status & CTL_AP_DONE) == CTL_AP_DONE;
            if (is_done)
                break;
            if (0 == timeout) {
                return -1;
            }
            nanosleep(&sleep_time, nullptr);
            timeout -= sleep_time_ms;
        };

        profiling_end(profiling_id_);

        return 0;
    }

    int dcr_write(uint32_t addr, uint32_t value) {
        CHECK_ERR(this->write_register(MMIO_DCR_ADDR, addr), {
            return err;
        });
        CHECK_ERR(this->write_register(MMIO_DCR_ADDR + 4, value), {
            return err;
        });
        dcrs_.write(addr, value);
        return 0;
    }

    int dcr_read(uint32_t addr, uint32_t* value) const {
        return dcrs_.read(addr, value);
    }

    int mpm_query(uint32_t addr, uint32_t core_id, uint64_t* value) {
        uint32_t offset = addr - VX_CSR_MPM_BASE;
        if (offset > 31)
            return -1;
        if (mpm_cache_.count(core_id) == 0) {
            uint64_t mpm_mem_addr = IO_MPM_ADDR + core_id * 32 * sizeof(uint64_t);
            CHECK_ERR(this->download(mpm_cache_[core_id].data(), mpm_mem_addr, 32 * sizeof(uint64_t)), {
                return err;
            });
        }
        *value = mpm_cache_.at(core_id).at(offset);
        return 0;
    }

private:

    xrt_device_t xrtDevice_;
    xrt_kernel_t xrtKernel_;
    const platform_info_t platform_;
    MemoryAllocator global_mem_;
    uint64_t dev_caps_;
    uint64_t isa_caps_;
    uint64_t global_mem_size_;
    DeviceConfig dcrs_;
    std::unordered_map<uint32_t, std::array<uint64_t, 32>> mpm_cache_;
    int profiling_id_;

#ifdef BANK_INTERLEAVE

    std::vector<xrt_buffer_t> xrtBuffers_;

    int get_bank_info(uint64_t addr, uint32_t* pIdx, uint64_t* pOff) {
        uint32_t num_banks = 1 << platform_.lg2_num_banks;
        uint64_t block_addr = addr / CACHE_BLOCK_SIZE;
        uint32_t index = block_addr & (num_banks-1);
        uint64_t offset = (block_addr >> platform_.lg2_num_banks) * CACHE_BLOCK_SIZE;
        if (pIdx) {
            *pIdx = index;
        }
        if (pOff) {
            *pOff = offset;
        }
        printf("get_bank_info(addr=0x%lx, bank=%d, offset=0x%lx\n", addr, index, offset);
        return 0;
    }

    int get_buffer(uint32_t bank_id, xrt_buffer_t* pBuf) {
        if (pBuf) {
            *pBuf = xrtBuffers_.at(bank_id);
        }
        return 0;
    }

#else

    struct buf_cnt_t {
        xrt_buffer_t xrtBuffer;
        uint32_t count;
    };

    std::unordered_map<uint32_t, buf_cnt_t> xrtBuffers_;

    int get_bank_info(uint64_t addr, uint32_t* pIdx, uint64_t* pOff) {
        uint32_t num_banks = 1 << platform_.lg2_num_banks;
        uint64_t bank_size = 1ull << platform_.lg2_bank_size;
        uint32_t index = addr >> platform_.lg2_bank_size;
        uint64_t offset = addr & (bank_size-1);
        if (index > num_banks) {
            fprintf(stderr, "[VXDRV] Error: address out of range: 0x%lx\n", addr);
            return -1;
        }
        if (pIdx) {
            *pIdx = index;
        }
        if (pOff) {
            *pOff = offset;
        }
        printf("get_bank_info(addr=0x%lx, bank=%d, offset=0x%lx\n", addr, index, offset);
        return 0;
    }

    int get_buffer(uint32_t bank_id, xrt_buffer_t* pBuf) {
        auto it = xrtBuffers_.find(bank_id);
        if (it != xrtBuffers_.end()) {
            if (pBuf) {
                *pBuf = it->second.xrtBuffer;
            } else {
                printf("reusing bank%d...\n", bank_id);
                ++it->second.count;
            }
        } else {
            printf("allocating bank%d...\n", bank_id);
            uint64_t bank_size = 1ull << platform_.lg2_bank_size;
        #ifdef CPP_API
            xrt::bo xrtBuffer(xrtDevice_, bank_size, xrt::bo::flags::normal, bank_id);
        #else
            CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice_, bank_size, XRT_BO_FLAGS_NONE, bank_id), {
                return -1;
            });
        #endif
            xrtBuffers_.insert({bank_id, {xrtBuffer, 1}});
            if (pBuf) {
                *pBuf = xrtBuffer;
            }
        }
        return 0;
    }

#endif
};

struct vx_buffer {
    vx_device* device;
    uint64_t addr;
    uint64_t size;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return -1;

    int device_index = DEFAULT_DEVICE_INDEX;
    const char* device_index_s = getenv("XRT_DEVICE_INDEX");
    if (device_index_s != nullptr) {
        device_index = atoi(device_index_s);
    }

    const char* xlbin_path_s = getenv("XRT_XCLBIN_PATH");
    if (xlbin_path_s == nullptr) {
        xlbin_path_s = DEFAULT_XCLBIN_PATH;
    }

#ifdef CPP_API

    auto xrtDevice = xrt::device(device_index);
    auto uuid = xrtDevice.load_xclbin(xlbin_path_s);
    auto xrtKernel = xrt::ip(xrtDevice, uuid, KERNEL_NAME);
    auto xclbin = xrt::xclbin(xlbin_path_s);

    auto device_name = xrtDevice.get_info<xrt::info::device::name>();

    /*{
        uint32_t num_banks = 0;
        uint64_t bank_size = 0;
        uint64_t mem_base  = 0;

        auto mem_json = nlohmann::json::parse(xrtDevice.get_info<xrt::info::device::memory>());
        if (!mem_json.is_null()) {
            uint32_t index = 0;
            for (auto& mem : mem_json["board"]["memory"]["memories"]) {
                auto enabled = mem["enabled"].get<std::string>();
                if (enabled == "true") {
                    if (index == 0) {
                        mem_base = std::stoull(mem["base_address"].get<std::string>(), nullptr, 16);
                        bank_size = std::stoull(mem["range_bytes"].get<std::string>(), nullptr, 16);
                    }
                    ++index;
                }
            }
            num_banks = index;
        }

        fprintf(stderr, "[VXDRV] memory description: base=0x%lx, size=0x%lx, count=%d\n", mem_base, bank_size, num_banks);
    }*/

    /*{
        std::cout << "Device" << device_index << " : " << xrtDevice.get_info<xrt::info::device::name>() << std::endl;
        std::cout << "  bdf      : " << xrtDevice.get_info<xrt::info::device::bdf>() << std::endl;
        std::cout << "  kdma     : " << xrtDevice.get_info<xrt::info::device::kdma>() << std::endl;
        std::cout << "  max_freq : " << xrtDevice.get_info<xrt::info::device::max_clock_frequency_mhz>() << std::endl;
        std::cout << "  memory   : " << xrtDevice.get_info<xrt::info::device::memory>() << std::endl;
        std::cout << "  thermal  : " << xrtDevice.get_info<xrt::info::device::thermal>() << std::endl;
        std::cout << "  m2m      : " << std::boolalpha << xrtDevice.get_info<xrt::info::device::m2m>() << std::dec << std::endl;
        std::cout << "  nodma    : " << std::boolalpha << xrtDevice.get_info<xrt::info::device::nodma>() << std::dec << std::endl;

        std::cout << "Memory info :" << std::endl;
        for (const auto& mem_bank : xclbin.get_mems()) {
            std::cout << "  index : " << mem_bank.get_index() << std::endl;
            std::cout << "  tag : " << mem_bank.get_tag() << std::endl;
            std::cout << "  type : " << (int)mem_bank.get_type() << std::endl;
            std::cout << "  base_address : 0x" << std::hex << mem_bank.get_base_address() << std::endl;
            std::cout << "  size : 0x" << (mem_bank.get_size_kb() * 1000) << std::dec << std::endl;
            std::cout << "  used :" << mem_bank.get_used() << std::endl;
        }
    }*/

    // get platform info
    platform_info_t platform_info;
    CHECK_ERR(get_platform_info(device_name, &platform_info), {
        fprintf(stderr, "[VXDRV] Error: platform not supported: %s\n", device_name.c_str());
        return err;
    });

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info), {
        return -1;
    });

#else

    CHECK_HANDLE(xrtDevice, xrtDeviceOpen(device_index), {
        return -1;
    });

#ifndef XRTSIM

    CHECK_ERR(xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s), {
        dump_xrt_error(xrtDevice, err);
        xrtDeviceClose(xrtDevice);
        return err;
    });

    xuid_t uuid;
    CHECK_ERR(xrtDeviceGetXclbinUUID(xrtDevice, uuid), {
        dump_xrt_error(xrtDevice, err);
        xrtDeviceClose(xrtDevice);
        return err;
    });

    CHECK_HANDLE(xrtKernel, xrtPLKernelOpenExclusive(xrtDevice, uuid, KERNEL_NAME), {
        xrtDeviceClose(xrtDevice);
        return -1;
    });

#else

    xrtKernelHandle xrtKernel = nullptr;

#endif

    int device_name_size;
    xrtXclbinGetXSAName(xrtDevice, nullptr, 0, &device_name_size);
    std::vector<char> device_name(device_name_size);
    xrtXclbinGetXSAName(xrtDevice, device_name.data(), device_name_size, nullptr);

    // get platform info
    platform_info_t platform_info;
    CHECK_ERR(get_platform_info(device_name.data(), &platform_info), {
        fprintf(stderr, "[VXDRV] Error: platform not supported: %s\n", device_name.data());
        return err;
    });

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info), {
        xrtKernelClose(xrtKernel);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

#endif

    // initialize device
    CHECK_ERR(device->init(), {
        return err;
    });

#ifdef SCOPE
    {
        scope_callback_t callback;
        callback.registerWrite = [](vx_device_h hdevice, uint64_t value)->int {
            auto device = (vx_device*)hdevice;
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
        callback.registerRead = [](vx_device_h hdevice, uint64_t* value)->int {
            auto device = (vx_device*)hdevice;
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
        int ret = vx_scope_start(&callback, device, 0, -1);
        if (ret != 0) {
            delete device;
            return ret;
        }
    }
#endif

    DBGPRINT("DEV_OPEN: hdevice=%p\n", (void*)device);

    *hdevice = device;

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("DEV_CLOSE: hdevice=%p\n", hdevice);

#ifdef SCOPE
    vx_scope_stop(hdevice);
#endif

    auto device = (vx_device*)hdevice;

    delete device;

    return 0;
}

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    uint64_t _value;

    CHECK_ERR(device->get_caps(caps_id, &_value), {
        return err;
    });

    DBGPRINT("DEV_CAPS: hdevice=%p, caps_id=%d, value=%ld\n", hdevice, caps_id, _value);

    *value = _value;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer) {
   if (nullptr == hdevice
    || nullptr == hbuffer
    || 0 == size)
        return -1;

    DBGPRINT("MEM_ALLOC: size=%ld, flags=0x%d\n", size, flags);

    auto device = ((vx_device*)hdevice);

    uint64_t dev_addr;
    CHECK_ERR(device->mem_alloc(size, flags, &dev_addr), {
        return err;
    });

    auto buffer = new vx_buffer{device, dev_addr, size};
    if (nullptr == buffer) {
        device->mem_free(dev_addr);
        return -1;
    }

    DBGPRINT("MEM_ALLOC: hdevice=%p, size=%ld, flags=0x%d, hbuffer=%p\n", hdevice, size, flags, (void*)buffer);

    *hbuffer = buffer;

    return 0;
}

extern int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice
     || nullptr == hbuffer
     || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);

    CHECK_ERR(device->mem_reserve(address, size, flags), {
        return err;
    });

    auto buffer = new vx_buffer{device, address, size};
    if (nullptr == buffer) {
        device->mem_free(address);
        return -1;
    }

    DBGPRINT("MEM_RESERVE: hdevice=%p, address=0x%lx, size=%ld, flags=0x%d, hbuffer=%p\n", hdevice, address, size, flags, (void*)buffer);

    *hbuffer = buffer;

    return 0;
}

extern int vx_mem_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return 0;

    DBGPRINT("MEM_FREE: hbuffer=%p\n", hbuffer);

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    vx_mem_access(hbuffer, 0, buffer->size, 0);

    int err = device->mem_free(buffer->addr);

    delete buffer;

    return err;
}

extern int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags) {
   if (nullptr == hbuffer)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((offset + size) > buffer->size)
        return -1;

    DBGPRINT("MEM_ACCESS: hbuffer=%p, offset=%ld, size=%ld, flags=%d\n", hbuffer, offset, size, flags);

    return device->mem_access(buffer->addr + offset, size, flags);
}

extern int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);

    DBGPRINT("MEM_ADDRESS: hbuffer=%p, address=0x%lx\n", hbuffer, buffer->addr);

    *address = buffer->addr;

    return 0;
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint64_t _mem_free, _mem_used;

    CHECK_ERR(device->mem_info(&_mem_free, &_mem_used), {
        return err;
    });

    DBGPRINT("MEM_INFO: hdevice=%p, mem_free=%ld, mem_used=%ld\n", hdevice, _mem_free, _mem_used);

    if (mem_free) {
        *mem_free = _mem_free;
    }

    if (mem_used) {
        *mem_used = _mem_used;
    }

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size) {
    if (nullptr == hbuffer || nullptr == host_ptr)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((dst_offset + size) > buffer->size)
        return -1;

    DBGPRINT("COPY_TO_DEV: hbuffer=%p, host_addr=%p, dst_offset=%ld, size=%ld\n", hbuffer, host_ptr, dst_offset, size);

    CHECK_ERR(device->upload(buffer->addr + dst_offset, host_ptr, size), {
        return err;
    });

    return 0;
}

extern int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size) {
    if (nullptr == hbuffer || nullptr == host_ptr)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((src_offset + size) > buffer->size)
        return -1;

    DBGPRINT("COPY_FROM_DEV: hbuffer=%p, host_addr=%p, src_offset=%ld, size=%ld\n", hbuffer, host_ptr, src_offset, size);

    CHECK_ERR(device->download(host_ptr, buffer->addr + src_offset, size), {
        return err;
    });

    return 0;
}

extern int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments) {
    if (nullptr == hdevice || nullptr == hkernel || nullptr == harguments)
        return -1;

    DBGPRINT("START: hdevice=%p, hkernel=%p, harguments=%p\n", hdevice, hkernel, harguments);

    //wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    auto device = (vx_device*)hdevice;
    auto kernel = ((vx_buffer*)hkernel);
    auto arguments = ((vx_buffer*)harguments);

    return device->start(kernel->addr, arguments->addr);
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("READY_WAIT: hdevice=%p, timeout=%ld\n", hdevice, timeout);

    auto device = ((vx_device*)hdevice);

    CHECK_ERR(device->ready_wait(timeout), {
        return err;
    });

    return  0;
}

extern int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t* value) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    uint32_t _value;

    CHECK_ERR(device->dcr_read(addr, &_value), {
        return err;
    });

    DBGPRINT("DCR_READ: hdevice=%p, addr=0x%x, value=0x%x\n", hdevice, addr, _value);

    *value = _value;

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("DCR_WRITE: hdevice=%p, addr=0x%x, value=0x%x\n", hdevice, addr, value);

    auto device = (vx_device*)hdevice;

    return device->dcr_write(addr, value);
}

extern int vx_mpm_query(vx_device_h hdevice, uint32_t addr, uint32_t core_id, uint64_t* value) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint64_t _value;

    CHECK_ERR(device->mpm_query(addr, core_id, &_value), {
        return err;
    });

    DBGPRINT("MPM_QUERY: hdevice=%p, addr=0x%x, core_id=%d, value=0x%lx\n", hdevice, addr, core_id, _value);

    *value = _value;

    return 0;
}
