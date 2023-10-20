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
#include <unordered_map>

#ifdef SCOPE
#include "scope.h"
#endif

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#include "experimental/xrt_error.h"

#define CPP_API
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
    {"xilinx_u50",     4, 0x1C, 0x0},
    {"xilinx_u200",    4, 0x1C, 0x0},
    {"xilinx_u280",    4, 0x1C, 0x0},
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

using namespace vortex;

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
    {}

#ifndef CPP_API
    
    ~vx_device() {
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
            return -1;
        });

        uint32_t num_banks = 1 << platform_.lg2_num_banks;
        uint64_t bank_size = 1ull << platform_.lg2_bank_size;

        for (uint32_t i = 0; i < num_banks; ++i) {
            uint32_t reg_addr = MMIO_MEM_ADDR + (i * 12);
            uint64_t reg_value = platform_.mem_base + i * bank_size;
            CHECK_ERR(this->write_register(reg_addr, reg_value & 0xffffffff), {
                return -1;
            });

            CHECK_ERR(this->write_register(reg_addr + 4, (reg_value >> 32) & 0xffffffff), {
                return -1;
            });
        #ifndef BANK_INTERLEAVE
            break;
        #endif
        }

        CHECK_ERR(this->read_register(MMIO_DEV_ADDR, (uint32_t*)&this->dev_caps), {
            return -1;
        });
        
        CHECK_ERR(this->read_register(MMIO_DEV_ADDR + 4, (uint32_t*)&this->dev_caps + 1), {
            return -1;
        });

        CHECK_ERR(this->read_register(MMIO_ISA_ADDR, (uint32_t*)&this->isa_caps), {
            return -1;
        });

        CHECK_ERR(this->read_register(MMIO_ISA_ADDR + 4, (uint32_t*)&this->isa_caps + 1), {
            return -1;
        });

        this->global_mem_size = num_banks * bank_size;

        this->global_mem_ = std::make_shared<vortex::MemoryAllocator>(
            ALLOC_BASE_ADDR, ALLOC_MAX_ADDR, RAM_PAGE_SIZE, CACHE_BLOCK_SIZE);

        uint64_t local_mem_size = 0;
        vx_dev_caps(this, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
        if (local_mem_size <= 1) {        
            this->local_mem_ = std::make_shared<vortex::MemoryAllocator>(
                SMEM_BASE_ADDR, local_mem_size, RAM_PAGE_SIZE, 1);
        }

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

        return 0;
    }

    int mem_alloc(uint64_t size, int type, uint64_t* dev_addr) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

        uint64_t addr;

        if (type == VX_MEM_TYPE_GLOBAL) {
            CHECK_ERR(global_mem_->allocate(asize, &addr), {
                return -1;
            });
        #ifndef BANK_INTERLEAVE
            uint32_t bank_id;
            CHECK_ERR(this->get_bank_info(addr, &bank_id, nullptr), {
                return -1;
            });
            CHECK_ERR(get_buffer(bank_id, nullptr), {
                return -1;
            });
        #endif
        } else if (type == VX_MEM_TYPE_LOCAL) {
            if CHECK_ERR(local_mem_->allocate(asize, &addr), {
                return -1;
            });
        } else {
            return -1;
        }       
        *dev_addr = addr;
        return 0;
    }

    int mem_free(uint64_t dev_addr) {    
        if (dev_addr >= SMEM_BASE_ADDR) {
            CHECK_ERR(local_mem_->release(dev_addr), {
                return -1;
            });    
        } else {
            CHECK_ERR(global_mem_->release(dev_addr), {
                return -1;
            });    
        #ifdef BANK_INTERLEAVE
            if (0 == global_mem_->allocated()) {
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
                return -1;
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
        }
        return 0;
    }

    int mem_info(int type, uint64_t* mem_free, uint64_t* mem_used) const {
        if (type == VX_MEM_TYPE_GLOBAL) {
            if (mem_free)
                *mem_free = global_mem_->free();
            if (mem_used)
                *mem_used = global_mem_->allocated();
        } else if (type == VX_MEM_TYPE_LOCAL) {
            if (mem_free)
                *mem_free = local_mem_->free();
            if (mem_used)
                *mem_free = local_mem_->allocated();
        } else {
            return -1;
        }
        return 0;
    }

    int write_register(uint32_t addr, uint32_t value) {
    #ifdef CPP_API
        xrtKernel_.write_register(addr, value);
    #else        
        CHECK_ERR(xrtKernelWriteRegister(xrtKernel_, addr, value), {
            dump_xrt_error(xrtDevice_, err);
            return -1;
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
            return -1;
        });
    #endif
        DBGPRINT("*** read_register: addr=0x%x, value=0x%x\n", addr, *value);
        return 0;
    }

    int upload(uint64_t dev_addr, uint8_t* host_ptr, uint64_t asize) {    
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
                return -1;
            });            
            CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
                return -1;
            });
        #ifdef CPP_API        
            xrtBuffer.write(host_ptr, asize, bo_offset);
            xrtBuffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset);
        #else
            CHECK_ERR(xrtBOWrite(xrtBuffer, host_ptr, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return -1;
            });        
            CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return -1;
            });    
        #endif
        }
        return 0;
    }

    int download(uint8_t* host_ptr, uint64_t dev_addr, uint64_t asize) {
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
                return -1;
            });
            CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
                return -1;
            });
        #ifdef CPP_API
            xrtBuffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset);
            xrtBuffer.read(host_ptr, asize, bo_offset);
        #else        
            CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return -1;
            });
            CHECK_ERR(xrtBORead(xrtBuffer, host_ptr, asize, bo_offset), {
                dump_xrt_error(xrtDevice_, err);
                return -1;
            });         
        #endif
        }
        return 0;
    }

    DeviceConfig dcrs;
    uint64_t dev_caps;
    uint64_t isa_caps;
    uint64_t global_mem_size;

private:

    xrt_device_t xrtDevice_;
    xrt_kernel_t xrtKernel_;
    const platform_info_t platform_;    
    std::shared_ptr<vortex::MemoryAllocator> global_mem_;
    std::shared_ptr<vortex::MemoryAllocator> local_mem_;

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

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = (device->dev_caps >> 0) & 0xff;
        break;
    case VX_CAPS_NUM_THREADS:
        *value = (device->dev_caps >> 8) & 0xff;
        break;
    case VX_CAPS_NUM_WARPS:
        *value = (device->dev_caps >> 16) & 0xff;
        break;
    case VX_CAPS_NUM_CORES:
        *value = (device->dev_caps >> 24) & 0xffff;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
   case VX_CAPS_GLOBAL_MEM_SIZE:
        *value = device->global_mem_size;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = 1ull << ((device->dev_caps >> 40) & 0xff);
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = (uint64_t(device->dcrs.read(VX_DCR_BASE_STARTUP_ADDR1)) << 32) | 
                           device->dcrs.read(VX_DCR_BASE_STARTUP_ADDR0);
        break;
    case VX_CAPS_ISA_FLAGS:
        *value = device->isa_caps;
        break;
    default:
        fprintf(stderr, "[VXDRV] Error: invalid caps id: %d\n", caps_id);
        std::abort();
        return -1;
    }

    return 0;
}

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
        return -1;
    });

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info), {
        return -1;
    });

#else

    CHECK_HANDLE(xrtDevice, xrtDeviceOpen(device_index), { 
        return -1; 
    });

    CHECK_ERR(xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s), {
        dump_xrt_error(xrtDevice, err);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    xuid_t uuid;
    CHECK_ERR(xrtDeviceGetXclbinUUID(xrtDevice, uuid), {
        dump_xrt_error(xrtDevice, err);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    CHECK_HANDLE(xrtKernel, xrtPLKernelOpenExclusive(xrtDevice, uuid, KERNEL_NAME), {
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    int device_name_size;
    xrtXclbinGetXSAName(xrtDevice, nullptr, 0, &device_name_size);
    std::vector<char> device_name(device_name_size);
    xrtXclbinGetXSAName(xrtDevice, device_name.data(), device_name_size, nullptr);

    // get platform info
    platform_info_t platform_info;
    CHECK_ERR(get_platform_info(device_name.data(), &platform_info), {
        fprintf(stderr, "[VXDRV] Error: platform not supported: %s\n", device_name.data());
        return -1;
    });

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info), {
        xrtKernelClose(xrtKernel);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

#endif    

    // initialize device
    CHECK_ERR(device->init(), {
        return -1;
    });

#ifdef SCOPE
    {
        scope_callback_t callback;
        callback.registerWrite = [](vx_device_h hdevice, uint64_t value)->int { 
            auto device = (vx_device*)hdevice;
            uint32_t value_lo = (uint32_t)(value);
            uint32_t value_hi = (uint32_t)(value >> 32);
            CHECK_ERR(device->write_register(MMIO_SCP_ADDR, value_lo), {
                return -1;
            });
            CHECK_ERR(device->write_register(MMIO_SCP_ADDR + 4, value_hi), {
                return -1;
            });
            return 0;
        };
        callback.registerRead = [](vx_device_h hdevice, uint64_t* value)->int {
            auto device = (vx_device*)hdevice;
            uint32_t value_lo, value_hi;
            CHECK_ERR(device->read_register(MMIO_SCP_ADDR, &value_lo), {
                return -1;
            });
            CHECK_ERR(device->read_register(MMIO_SCP_ADDR + 4, &value_hi), {
                return -1;
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
        
    CHECK_ERR(dcr_initialize(device), {
        delete device;
        return -1;
    });    

#ifdef DUMP_PERF_STATS
    perf_add_device(device);
#endif

    *hdevice = device;

    DBGPRINT("device creation complete!\n");

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

#ifdef SCOPE
    vx_scope_stop(hdevice);
#endif

    auto device = (vx_device*)hdevice;

    delete device;

    DBGPRINT("device destroyed!\n");

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int type, uint64_t* dev_addr) {
   if (nullptr == hdevice 
    || nullptr == dev_addr
    || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);
    return device->mem_alloc(size, type, dev_addr);
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_addr) {
    if (nullptr == hdevice)
        return -1;

    if (0 == dev_addr)
        return 0;

    auto device = (vx_device*)hdevice;
    return device->mem_free(dev_addr);
}

extern int vx_mem_info(vx_device_h hdevice, int type, uint64_t* mem_free, uint64_t* mem_used) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    return device->mem_info(type, mem_free, mem_used);
}

extern int vx_copy_to_dev(vx_device_h hdevice, uint64_t dev_addr, const void* host_ptr, uint64_t size) {
    if (nullptr == hdevice)
        return -1;
    
    auto device = (vx_device*)hdevice;

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
        return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > device->global_mem_size)
        return -1;

    CHECK_ERR(device->upload(dev_addr, host_ptr, asize), {
        return -1;
    });

    DBGPRINT("COPY_TO_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld\n", dev_addr, (uintptr_t)host_ptr, size);
    
    return 0;
}

extern int vx_copy_from_dev(vx_device_h hdevice, void* host_ptr, uint64_t dev_addr, uint64_t size) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
        return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > device->global_mem_size)
        return -1;

    CHECK_ERR(device->download(host_ptr, dev_addr, asize), {
        return -1;
    });

    DBGPRINT("COPY_FROM_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld\n", dev_addr, (uintptr_t)host_ptr, asize);
    
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    //wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    CHECK_ERR(device->write_register(MMIO_CTL_ADDR, CTL_AP_START), {
        return -1;
    });
    
    DBGPRINT("START\n");

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

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
        CHECK_ERR(device->read_register(MMIO_CTL_ADDR, &status), {
            return -1;
        });
        bool is_done = (status & CTL_AP_DONE) == CTL_AP_DONE;
        if (is_done || 0 == timeout) {            
            break;
        }
        nanosleep(&sleep_time, nullptr);
        timeout -= sleep_time_ms;
    };

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
   
    CHECK_ERR(device->write_register(MMIO_DCR_ADDR, addr), {
        return -1;
    });

    CHECK_ERR(device->write_register(MMIO_DCR_ADDR + 4, value), {
        return -1;
    });

    // save the value
    DBGPRINT("DCR_WRITE: addr=0x%x, value=0x%lx\n", addr, value);
    device->dcrs.write(addr, value);
    
    return 0;
}
