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

#define MMIO_CTL_ADDR   0x00
#define MMIO_DEV_ADDR   0x10
#define MMIO_ISA_ADDR   0x1C
#define MMIO_DCR_ADDR   0x28
#define MMIO_MEM_ADDR   0x34
#define MMIO_SCP_ADDR   0x40

#define CTL_AP_START    (1<<0)
#define CTL_AP_DONE     (1<<1)
#define CTL_AP_IDLE     (1<<2)
#define CTL_AP_READY    (1<<3)
#define CTL_AP_RESET    (1<<4)
#define CTL_AP_RESTART  (1<<7)

struct platform_info_t {
    const char* prefix_name;
    uint32_t    num_banks;
    uint64_t    mem_base;
    uint64_t    bank_size;
};

static const platform_info_t g_platforms [] = {
    {"xilinx_u50",     32, 0x0, 0x10000000},
    {"xilinx_u200",    32, 0x0, 0x10000000},
    {"xilinx_u280",    32, 0x0, 0x10000000},
    {"xilinx_vck5000", 1,  0xC000000000, 0x200000000},
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
            *platform_info = platform;;
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

    vx_device(xrt_device_t& device, xrt_kernel_t& kernel, uint32_t num_banks, uint64_t bank_size)
        : xrtDevice(device)
        , xrtKernel(kernel)
        , num_banks_(num_banks)
        , bank_size_(bank_size)
        , mem_allocator_(
            ALLOC_BASE_ADDR, 
            ALLOC_MAX_ADDR,
            4096,            
            CACHE_BLOCK_SIZE)
    {}

#ifndef CPP_API
    
    ~vx_device() {
        for (auto it : xrtBuffers_) {
            xrtBOFree(it.second.xrtBuffer);
        }
        if (this->xrtKernel) {
            xrtKernelClose(this->xrtKernel); 
        }
        if (this->xrtDevice) {
            xrtDeviceClose(this->xrtDevice);
        }
    }

#endif

    int mem_alloc(uint64_t size, uint64_t* dev_maddr) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

        uint64_t addr;

        CHECK_ERR(mem_allocator_.allocate(asize, &addr), {
            return -1;
        });

        uint32_t bank_id;

        CHECK_ERR(this->get_bank_info(addr, &bank_id, nullptr), {
            return -1;
        });

        CHECK_ERR(get_buffer(bank_id, nullptr), {
            return -1;
        });

        *dev_maddr = addr;

        return 0;
    }

    int mem_free(uint64_t dev_maddr) {
        
        uint32_t bank_id;

        CHECK_ERR(this->get_bank_info(dev_maddr, &bank_id, nullptr), {
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
            CHECK_ERR(mem_allocator_.release(dev_maddr), {
                return -1;
            });
        } else {
            fprintf(stderr, "[VXDRV] Error: invalid device memory address: 0x%lx\n", dev_maddr);
            return -1;
        }   

        return 0;
    }

    uint64_t mem_used() const {
        return mem_allocator_.allocated();
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
        #ifdef CPP_API
            xrt::bo xrtBuffer(xrtDevice, bank_size_, xrt::bo::flags::normal, bank_id);
        #else
            CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice, bank_size_, XRT_BO_FLAGS_NONE, bank_id), {
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

    int get_bank_info(uint64_t addr, uint32_t* pIdx, uint64_t* pOff) {
        uint32_t index  = addr / bank_size_;
        uint64_t offset = addr % bank_size_;

        printf("get_bank_info(addr=0x%lx, bank=%d, offset=0x%lx\n", addr, index, offset);
        
        if (index > num_banks_) {
            fprintf(stderr, "[VXDRV] Error: address out of range: 0x%lx\n", addr);
            return -1;
        }
        
        if (pIdx) {
            *pIdx = index;
        }

        if (pOff) {
            *pOff = offset;
        }
        
        return 0;
    }

    xrt_device_t xrtDevice;
    xrt_kernel_t xrtKernel;

    struct buf_cnt_t {
        xrt_buffer_t xrtBuffer;
        uint32_t count;
    };

    uint32_t num_banks_;
    uint64_t bank_size_;

    DeviceConfig dcrs;
    uint64_t dev_caps;
    uint64_t isa_caps;

private:

    std::unordered_map<uint32_t, buf_cnt_t> xrtBuffers_;
    vortex::MemoryAllocator mem_allocator_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_buffer {
public:
    vx_buffer(vx_device* device, uint8_t* host_ptr, uint64_t size) 
        : device(device)
        , host_ptr(host_ptr)
        , size(size)
    {}
    
    ~vx_buffer() {
        if (host_ptr != nullptr) {
            aligned_free(host_ptr);
        }
    }
    
    vx_device* device;
    uint8_t* host_ptr;
    uint64_t size;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = (device->dev_caps >> 0) & 0xffff;
        break;
    case VX_CAPS_MAX_CORES:
        *value = (device->dev_caps >> 16) & 0xffff;
        break;
    case VX_CAPS_MAX_WARPS:
        *value = (device->dev_caps >> 32) & 0xffff;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = (device->dev_caps >> 48) & 0xffff;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = LOCAL_MEM_SIZE;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = device->dcrs.read(DCR_BASE_STARTUP_ADDR);
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

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info.num_banks, platform_info.bank_size), {
        return -1;
    });

    xrtKernel.write_register(MMIO_CTL_ADDR, CTL_AP_RESET);

    xrtKernel.write_register(MMIO_MEM_ADDR, platform_info.mem_base & 0xffffffff);
    xrtKernel.write_register(MMIO_MEM_ADDR + 4, (platform_info.mem_base >> 32) & 0xffffffff);

    auto dev_caps_lo = xrtKernel.read_register(MMIO_DEV_ADDR);
    auto dev_caps_hi = xrtKernel.read_register(MMIO_DEV_ADDR + 4);

    auto isa_caps_lo = xrtKernel.read_register(MMIO_ISA_ADDR);
    auto isa_caps_hi = xrtKernel.read_register(MMIO_ISA_ADDR + 4);

    device->dev_caps = (uint64_t(dev_caps_hi) << 32) | dev_caps_lo;
    device->isa_caps = (uint64_t(isa_caps_hi) << 32) | isa_caps_lo;

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

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel, platform_info.num_banks, platform_info.bank_size), {
        xrtKernelClose(xrtKernel);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_CTL_ADDR, CTL_AP_RESET), {
        dump_xrt_error(xrtDevice, err);
        return -1;
    });

    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_MEM_ADDR, platform_info.mem_base & 0xffffffff), {
        dump_xrt_error(xrtDevice, err);
        return -1;
    });

    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_MEM_ADDR + 4, (platform_info.mem_base >> 32) & 0xffffffff), {
        dump_xrt_error(xrtDevice, err);
        return -1;
    });

    CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_DEV_ADDR, (uint32_t*)&device->dev_caps), {
        dump_xrt_error(xrtDevice, err);
        delete device;
        return -1;
    });
    
    CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_DEV_ADDR + 4, (uint32_t*)&device->dev_caps + 1), {
        dump_xrt_error(xrtDevice, err);
        delete device;
        return -1;
    });

    CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_ISA_ADDR, (uint32_t*)&device->isa_caps), {
        dump_xrt_error(xrtDevice, err);
        delete device;
        return -1;
    });

    CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_ISA_ADDR + 4, (uint32_t*)&device->isa_caps + 1), {
        dump_xrt_error(xrtDevice, err);
        delete device;
        return -1;
    });

#endif 

#ifdef SCOPE
    {
        scope_callback_t callback;
        callback.registerWrite = [](vx_device_h hdevice, uint64_t value)->int { 
            auto device = (vx_device*)hdevice;
            uint32_t value_lo = (uint32_t)(value);
            uint32_t value_hi = (uint32_t)(value >> 32);
        #ifdef CPP_API
            device->xrtKernel.write_register(MMIO_SCP_ADDR, value_lo);
            device->xrtKernel.write_register(MMIO_SCP_ADDR + 4, value_hi);
        #else        
            CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_SCP_ADDR, value_lo), {
                dump_xrt_error(device->xrtDevice, err);
                return -1;
            });
            CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_SCP_ADDR + 4, value_hi), {
                dump_xrt_error(device->xrtDevice, err);
                return -1;
            });        
        #endif
            return 0;
        };
        callback.registerRead = [](vx_device_h hdevice, uint64_t* value)->int {
            auto device = (vx_device*)hdevice;
            uint32_t value_lo, value_hi;
        #ifdef CPP_API
            device->xrtKernel.read_register(MMIO_SCP_ADDR, &value_lo);
            device->xrtKernel.read_register(MMIO_SCP_ADDR + 4, &value_hi);
        #else        
            CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_SCP_ADDR, &value_lo), {
                dump_xrt_error(device->xrtDevice, err);
                return -1;
            });
            CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_SCP_ADDR + 4, &value_hi), {
                dump_xrt_error(device->xrtDevice, err);
                return -1;
            });
        #endif
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

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
   if (nullptr == hdevice 
    || nullptr == dev_maddr
    || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);
    return device->mem_alloc(size, dev_maddr);
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    return device->mem_free(dev_maddr);
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_total) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    if (mem_free) {
        *mem_free = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR) - device->mem_used();
    }
    if (mem_total) {
        *mem_total = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR);
    }
    return 0;
}

extern int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    uint8_t* host_ptr = (uint8_t*)aligned_malloc(asize, CACHE_BLOCK_SIZE);
    if (host_ptr == nullptr) {
        fprintf(stderr, "[VXDRV] Error: allocation failed\n");
        return -1;
    }

    auto buffer = new vx_buffer(device, host_ptr, asize);

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    auto buffer = (vx_buffer*)hbuffer;

    return buffer->host_ptr;
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    delete buffer;

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    if (nullptr == hbuffer)
        return -1;

    uint64_t dev_mem_size = LOCAL_MEM_SIZE; 
    int64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    auto buffer = (vx_buffer*)hbuffer;
    auto device = buffer->device;

    auto host_ptr = buffer->host_ptr + src_offset;

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned((uintptr_t)host_ptr, CACHE_BLOCK_SIZE))
        return -1;

    // bound checking
    if (src_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;
        
    uint32_t bo_index;
    uint64_t bo_offset;
    CHECK_ERR(device->get_bank_info(dev_maddr, &bo_index, &bo_offset), {
        return -1;
    });

    xrt_buffer_t xrtBuffer;
    CHECK_ERR(device->get_buffer(bo_index, &xrtBuffer), {
        return -1;
    });
    

#ifdef CPP_API
    
    xrtBuffer.write(host_ptr, asize, bo_offset);
    xrtBuffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset);

#else

    CHECK_ERR(xrtBOWrite(xrtBuffer, host_ptr, asize, bo_offset), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });
    
    CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });
 
#endif

    DBGPRINT("COPY_TO_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld, bank=%d, offset=0x%lx\n", dev_maddr, (uintptr_t)host_ptr, size, bo_index, bo_offset);
    
    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    auto buffer = (vx_buffer*)hbuffer;
    auto device = buffer->device;
    
    uint64_t dev_mem_size = LOCAL_MEM_SIZE;  
    int64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    auto host_ptr = buffer->host_ptr + dest_offset;

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned((uintptr_t)host_ptr, CACHE_BLOCK_SIZE))
        return -1; 

    // bound checking
    if (dest_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;

    uint32_t bo_index;
    uint64_t bo_offset;
    CHECK_ERR(device->get_bank_info(dev_maddr, &bo_index, &bo_offset), {
        return -1;
    });   

    xrt_buffer_t xrtBuffer;
    CHECK_ERR(device->get_buffer(bo_index, &xrtBuffer), {
        return -1;
    });

#ifdef CPP_API

    xrtBuffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset);
    xrtBuffer.read(host_ptr, asize, bo_offset);

#else
    
    CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_FROM_DEVICE, asize, bo_offset), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });

    CHECK_ERR(xrtBORead(xrtBuffer, host_ptr, asize, bo_offset), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });
   
#endif

    DBGPRINT("COPY_FROM_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld, bank=%d, offset=0x%lx\n", dev_maddr, (uintptr_t)host_ptr, asize, bo_index, bo_offset);
    
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    //wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

#ifdef CPP_API

    device->xrtKernel.write_register(MMIO_CTL_ADDR, CTL_AP_START);

#else

    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_CTL_ADDR, CTL_AP_START), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });

#endif    
    
    DBGPRINT("START");

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

    #ifdef CPP_API

        status = device->xrtKernel.read_register(MMIO_CTL_ADDR);

    #else       

        CHECK_ERR(xrtKernelReadRegister(device->xrtKernel, MMIO_CTL_ADDR, &status), {
            dump_xrt_error(device->xrtDevice, err);
            return -1;
        });
    
    #endif

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

#ifdef CPP_API

    device->xrtKernel.write_register(MMIO_DCR_ADDR, addr);
    device->xrtKernel.write_register(MMIO_DCR_ADDR + 4, value);

#else
   
    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_DCR_ADDR, addr), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });

    CHECK_ERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_DCR_ADDR + 4, value), {
        dump_xrt_error(device->xrtDevice, err);
        return -1;
    });
 
#endif

    // save the value
    DBGPRINT("DCR_WRITE: addr=0x%x, value=0x%lx\n", addr, value);
    device->dcrs.write(addr, value);
    
    return 0;
}
