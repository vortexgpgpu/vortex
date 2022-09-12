#include <vortex.h>
#include <vx_malloc.h>
#include <vx_utils.h>
#include <VX_config.h>
#include <VX_types.h>
#include <vx_malloc.h>
#include <util.h>

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

#define CTL_AP_START    (1<<0)
#define CTL_AP_DONE     (1<<1)
#define CTL_AP_IDLE     (1<<2)
#define CTL_AP_READY    (1<<3)
#define CTL_AP_RESTART  (1<<7)

#define RAM_PAGE_SIZE 4096

#define BANK_SIZE (0x100000 * 256)

#define NUM_BANKS 16

#define DEFAULT_DEVICE_INDEX 0

#define DEFAULT_XCLBIN_PATH "vortex_afu.xclbin"

#define KERNEL_NAME "vortex_afu"

#ifndef NDEBUG
#define DBGPRINT(format, ...) printf("[VXDRV] " #format "\n", ##__VA_ARGS__)
#else
#define DBGPRINT(x)
#endif

#define CHECK_HANDLE(handle, _expr, _cleanup)   \
    auto handle = _expr;                        \
    if (handle == nullptr) {                    \
        printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr); \
        _cleanup                                \
    }

#define CHECK_XERR(_expr, _cleanup)             \
    do {                                        \
        xrtErrorCode err = _expr;               \
        if (err == 0)                           \
            break;                              \
        size_t len = 0;                         \
        xrtErrorGetString(xrtDevice, err, nullptr, 0, &len); \
        std::vector<char> buf(len);             \
        xrtErrorGetString(xrtDevice, err, buf.data(), buf.size(), nullptr); \
        printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err, buf.data()); \
        _cleanup                                \
    } while (false)

#define CHECK_ERR(_expr, _cleanup)              \
    do {                                        \
        xrtErrorCode err = _expr;               \
        if (err == 0)                           \
            break;                              \
        printf("[VXDRV] Error: '%s' returned %d!\n", #_expr, (int)err); \
        _cleanup                                \
    } while (false)

using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device(xrtDeviceHandle xrtDevice = nullptr, xrtKernelHandle xrtKernel = nullptr)
        : xrtDevice(xrtDevice)
        , xrtKernel(xrtKernel)
        , mem_allocator(
            ALLOC_BASE_ADDR, 
            ALLOC_BASE_ADDR + LOCAL_MEM_SIZE,
            4096,            
            CACHE_BLOCK_SIZE)
    {
        for (int i = 0; i < NUM_BANKS; ++i) {
            this->xrtBuffers[i] = nullptr;
        }

    }
    
    ~vx_device() {
        for (int i = 0; i < NUM_BANKS; ++i) {
            if (this->xrtBuffers[i]) {
                xrtBOFree(this->xrtBuffers[i]);
            }
        }
        if (this->xrtKernel) {
            xrtKernelClose(this->xrtKernel); 
        }
        if (this->xrtDevice) {
            xrtDeviceClose(this->xrtDevice);
        }
    }

    int findBO(uint64_t dev_addr, xrtBufferHandle* pBuf, uint64_t* pOff) {
        uint32_t index  = dev_addr / BANK_SIZE;
        uint32_t offset = dev_addr % BANK_SIZE;
        if (index > NUM_BANKS) {
            fprintf(stderr, "[VXDRV] Error: address out of range: %ld\n", dev_addr);
            return -1;
        }
        *pBuf = xrtBuffers[index];
        *pOff = offset;
        return 0;
    }

    xrtDeviceHandle xrtDevice;
    xrtKernelHandle xrtKernel;
    xrtBufferHandle xrtBuffers[NUM_BANKS];

    vortex::MemoryAllocator mem_allocator;
    DeviceConfig dcrs;
    uint64_t dev_caps;
    uint64_t isa_caps;
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
    case VX_CAPS_ALLOC_BASE_ADDR:
        *value = ALLOC_BASE_ADDR;
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

    CHECK_HANDLE(xrtDevice, xrtDeviceOpen(device_index), { 
        return -1; 
    });

    CHECK_XERR(xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s), {
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    xuid_t uuid;
    CHECK_XERR(xrtDeviceGetXclbinUUID(xrtDevice, uuid), {
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    CHECK_HANDLE(xrtKernel, xrtPLKernelOpenExclusive(xrtDevice, uuid, KERNEL_NAME), {
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    CHECK_HANDLE(device, new vx_device(xrtDevice, xrtKernel), {
        xrtKernelClose(xrtKernel);
        xrtDeviceClose(xrtDevice);
        return -1;
    });

    for (int i = 0; i < NUM_BANKS; ++i) {
        CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice, BANK_SIZE, 0, i), {
            delete device;
            return -1;
        });
        device->xrtBuffers[i] = xrtBuffer;
    }

    CHECK_XERR(xrtKernelReadRegister(device->xrtKernel, MMIO_DEV_ADDR, (uint32_t*)&device->dev_caps), {
        delete device;
        return -1;
    });
    
    CHECK_XERR(xrtKernelReadRegister(device->xrtKernel, MMIO_DEV_ADDR + 4, (uint32_t*)&device->dev_caps + 1), {
        delete device;
        return -1;
    });

    CHECK_XERR(xrtKernelReadRegister(device->xrtKernel, MMIO_ISA_ADDR, (uint32_t*)&device->isa_caps), {
        delete device;
        return -1;
    });

    CHECK_XERR(xrtKernelReadRegister(device->xrtKernel, MMIO_ISA_ADDR + 4, (uint32_t*)&device->isa_caps + 1), {
        delete device;
        return -1;
    });
        
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

    auto device = (vx_device*)hdevice;    

    delete device;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
   if (nullptr == hdevice 
    || nullptr == dev_maddr
    || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);
    return device->mem_allocator.allocate(size, dev_maddr);
}

int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    return device->mem_allocator.release(dev_maddr);
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

    auto buffer = (vx_buffer*)hbuffer;
    auto device = buffer->device;
    auto xrtDevice = device->xrtDevice;

    uint64_t dev_mem_size = LOCAL_MEM_SIZE; 
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

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

    xrtBufferHandle xrtBuffer;
    uint64_t bo_offset;
    CHECK_ERR(device->findBO(dev_maddr, &xrtBuffer, &bo_offset), {
        return -1;
    });

    CHECK_XERR(xrtBOWrite(xrtBuffer, host_ptr, asize, bo_offset), {
        return -1;
    });
    
    CHECK_XERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_TO_DEVICE, asize, bo_offset), {
        return -1;
    });
    
    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    auto buffer = (vx_buffer*)hbuffer;
    auto device = buffer->device;
    auto xrtDevice = device->xrtDevice;

    uint64_t dev_mem_size = LOCAL_MEM_SIZE;  
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

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

    xrtBufferHandle xrtBuffer;
    uint64_t bo_offset;
    CHECK_ERR(device->findBO(dev_maddr, &xrtBuffer, &bo_offset), {
        return -1;
    });
    
    CHECK_XERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_FROM_DEVICE, size, bo_offset), {
        return -1;
    });

    CHECK_XERR(xrtBORead(xrtBuffer, host_ptr, size, bo_offset), {
        return -1;
    });
    
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    auto xrtDevice = device->xrtDevice;

    CHECK_XERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_CTL_ADDR, CTL_AP_START), {
        return -1;
    });

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    auto xrtDevice = device->xrtDevice;

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
        CHECK_XERR(xrtKernelReadRegister(device->xrtKernel, MMIO_CTL_ADDR, &status), {
            return -1;
        });

        bool is_idle = (status & CTL_AP_IDLE) == CTL_AP_IDLE;
        if (is_idle || 0 == timeout) {            
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
    auto xrtDevice = device->xrtDevice;

    CHECK_XERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_DCR_ADDR, addr), {
        return -1;
    });

    CHECK_XERR(xrtKernelWriteRegister(device->xrtKernel, MMIO_DCR_ADDR + 4, value), {
        return -1;
    });

    // save the value
    device->dcrs.write(addr, value);
    
    return 0;
}