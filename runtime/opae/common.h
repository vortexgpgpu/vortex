#pragma once

#include <vortex.h>
#include <utils.h>
#include <malloc.h>
#include "driver.h"

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
        printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err, api.fpgaErrStr(err)); \
        _cleanup                                \
    } while (false)

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device() 
        : mem_allocator(
            ALLOC_BASE_ADDR, 
            ALLOC_MAX_ADDR,
            4096,            
            CACHE_BLOCK_SIZE)
    {}
    
    ~vx_device() {}

    opae_drv_api_t api;

    fpga_handle fpga;
    vortex::MemoryAllocator mem_allocator;    
    DeviceConfig dcrs;
    unsigned version;
    unsigned num_cores;
    unsigned num_warps;
    unsigned num_threads;
    uint64_t isa_caps;
};

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_buffer_ {
    uint64_t wsid;
    void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    uint64_t size;
} vx_buffer_t;
