#include <vortex.h>
#include <utils.h>
#include <malloc.h>
#include "driver.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <uuid/uuid.h>
#include <unistd.h>
#include <assert.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <list>

#include <VX_config.h>
#include <VX_types.h>
#include <vortex_afu.h>

#ifdef SCOPE
#include "scope.h"
#endif

///////////////////////////////////////////////////////////////////////////////

#define CMD_MEM_READ        AFU_IMAGE_CMD_MEM_READ
#define CMD_MEM_WRITE       AFU_IMAGE_CMD_MEM_WRITE
#define CMD_RUN             AFU_IMAGE_CMD_RUN
#define CMD_DCR_WRITE       AFU_IMAGE_CMD_DCR_WRITE

#define MMIO_CMD_TYPE       (AFU_IMAGE_MMIO_CMD_TYPE * 4)
#define MMIO_CMD_ARG0       (AFU_IMAGE_MMIO_CMD_ARG0 * 4)
#define MMIO_CMD_ARG1       (AFU_IMAGE_MMIO_CMD_ARG1 * 4)
#define MMIO_CMD_ARG2       (AFU_IMAGE_MMIO_CMD_ARG2 * 4)
#define MMIO_STATUS         (AFU_IMAGE_MMIO_STATUS   * 4)
#define MMIO_DEV_CAPS       (AFU_IMAGE_MMIO_DEV_CAPS * 4)
#define MMIO_ISA_CAPS       (AFU_IMAGE_MMIO_ISA_CAPS * 4)
#define MMIO_SCOPE_READ     (AFU_IMAGE_MMIO_SCOPE_READ * 4)
#define MMIO_SCOPE_WRITE    (AFU_IMAGE_MMIO_SCOPE_WRITE * 4)

#define STATUS_STATE_BITS   8

#define RAM_PAGE_SIZE 4096

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
            RAM_PAGE_SIZE,            
            CACHE_BLOCK_SIZE)
    {}
    
    ~vx_device() {}

    opae_drv_api_t api;

    fpga_handle fpga;
    vortex::MemoryAllocator mem_allocator;    
    DeviceConfig dcrs;
    uint64_t dev_caps;
    uint64_t isa_caps;
    uint64_t mem_size;
};

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_buffer_ {
    uint64_t wsid;
    void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    uint64_t size;
} vx_buffer_t;

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);

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
        *value = (device->dev_caps >> 32) & 0xff;
        break;
    case VX_CAPS_NUM_CLUSTERS:
        *value = (device->dev_caps >> 40) & 0xff;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = device->mem_size;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = device->dcrs.read(DCR_BASE_STARTUP_ADDR0);
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
        return  -1;

    vx_device* device;

    fpga_handle accel_handle;
    fpga_token accel_token;
    fpga_properties filter;    
    fpga_guid guid; 

    uint32_t num_matches;

    opae_drv_api_t api;
    memset(&api, 0, sizeof(opae_drv_api_t));
    if (drv_init(&api) !=0) {
        return -1;
    }
    
    // Set up a filter that will search for an accelerator
    CHECK_ERR(api.fpgaGetProperties(nullptr, &filter), {
        return -1;
    });
    
    CHECK_ERR(api.fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR), {
        api.fpgaDestroyProperties(&filter);
        return -1;
    });

    // Add the desired UUID to the filter
    std::string s_uuid(AFU_ACCEL_UUID);
    std::replace(s_uuid.begin(), s_uuid.end(), '_', '-');
    uuid_parse(s_uuid.c_str(), guid);    
    CHECK_ERR(api.fpgaPropertiesSetGUID(filter, guid), {        
        api.fpgaDestroyProperties(&filter);
        return -1;
    });

    // Do the search across the available FPGA contexts
    CHECK_ERR(api.fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches), {
        api.fpgaDestroyProperties(&filter);
        return -1;
    });

    // Not needed anymore
    CHECK_ERR(api.fpgaDestroyProperties(&filter), {
        api.fpgaDestroyToken(&accel_token);
        return -1;
    });

    if (num_matches < 1) {
        fprintf(stderr, "[VXDRV] Error: accelerator %s not found!\n", AFU_ACCEL_UUID);
        api.fpgaDestroyToken(&accel_token);
        return -1;
    }

    // Open accelerator
    CHECK_ERR(api.fpgaOpen(accel_token, &accel_handle, 0), {
        api.fpgaDestroyToken(&accel_token);
        return -1;
    });

    // Done with token
    CHECK_ERR(api.fpgaDestroyToken(&accel_token), {
        api.fpgaClose(accel_handle);
        return -1;
    });

    // allocate device object
    device = new vx_device();
    if (nullptr == device) {
        api.fpgaClose(accel_handle);
        return -1;
    }

    device->api = api;
    device->fpga = accel_handle;

    {   
        // retrieve FPGA local memory size
        CHECK_ERR(api.fpgaPropertiesGetLocalMemorySize(filter, &device->mem_size), {
            api.fpgaClose(accel_handle);
            return -1;
        });

        // Load ISA CAPS
        CHECK_ERR(api.fpgaReadMMIO64(device->fpga, 0, MMIO_ISA_CAPS, &device->isa_caps), {
            api.fpgaClose(accel_handle);
            return -1;
        });

        // Load device CAPS
        
        CHECK_ERR(api.fpgaReadMMIO64(device->fpga, 0, MMIO_DEV_CAPS, &device->dev_caps), {
            api.fpgaClose(accel_handle);
            return -1;
        });
    }
    
#ifdef SCOPE
    {
        scope_callback_t callback;
        callback.registerWrite = [](vx_device_h hdevice, uint64_t value)->int { 
            auto device = (vx_device*)hdevice;
            return device->api.fpgaWriteMMIO64(device->fpga, 0, MMIO_SCOPE_WRITE, value);
        };
        callback.registerRead = [](vx_device_h hdevice, uint64_t* value)->int {
            auto device = (vx_device*)hdevice;
            return device->api.fpgaReadMMIO64(device->fpga, 0, MMIO_SCOPE_READ, value);
        };
        int ret = vx_scope_start(&callback, device, 0, -1);
        if (ret != 0) {
            api.fpgaClose(accel_handle);
            return ret;
        }
    }
#endif

    int err = dcr_initialize(device);
    if (err != 0) {
        delete device;
        return err;
    }

#ifdef DUMP_PERF_STATS
    perf_add_device(device);
#endif    

    *hdevice = device;    

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

#ifdef SCOPE
    vx_scope_stop(hdevice);
#endif

#ifdef DUMP_PERF_STATS
    perf_remove_device(hdevice);
#endif

    api.fpgaClose(device->fpga);

    delete device;

    drv_close();

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

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    if (0 == dev_maddr)
        return 0;

    auto device = ((vx_device*)hdevice);
    return device->mem_allocator.release(dev_maddr);
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_total) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);
    if (mem_free) {
        *mem_free = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR) - device->mem_allocator.allocated();
    }
    if (mem_total) {
        *mem_total = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR);
    }
    return 0;
}

extern int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer) {
    void* host_ptr;
    uint64_t wsid;
    uint64_t io_addr;
    vx_buffer_t* buffer;

    if (nullptr == hdevice
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

    size_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    CHECK_ERR(api.fpgaPrepareBuffer(device->fpga, asize, &host_ptr, &wsid, 0), {
        return -1;
    });

    // Get the physical address of the buffer in the accelerator
    CHECK_ERR(api.fpgaGetIOAddress(device->fpga, wsid, &io_addr), {
        api.fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    });

    // allocate buffer object
    buffer = (vx_buffer_t*)malloc(sizeof(vx_buffer_t));
    if (nullptr == buffer) {
        api.fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    }

    buffer->wsid     = wsid;
    buffer->host_ptr = host_ptr;
    buffer->io_addr  = io_addr;
    buffer->hdevice  = hdevice;
    buffer->size     = asize;

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    auto buffer = ((vx_buffer_t*)hbuffer);
    return buffer->host_ptr;
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = ((vx_buffer_t*)hbuffer);
    auto device = ((vx_device*)buffer->hdevice);
    auto& api = device->api;

    api.fpgaReleaseBuffer(device->fpga, buffer->wsid);

    free(buffer);

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    std::unordered_map<uint32_t, std::stringstream> print_bufs;
    
    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

    struct timespec sleep_time; 

    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;

    // to milliseconds
    uint64_t sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);
    
    for (;;) {
        uint64_t status;
        CHECK_ERR(api.fpgaReadMMIO64(device->fpga, 0, MMIO_STATUS, &status), {
            return -1; 
        });

        // check for console data
        uint32_t cout_data = status >> STATUS_STATE_BITS;
        if (cout_data & 0x1) {
            // retrieve console data
            do {
                char cout_char = (cout_data >> 1) & 0xff;
                uint32_t cout_tid = (cout_data >> 9) & 0xff;
                auto& ss_buf = print_bufs[cout_tid];
                ss_buf << cout_char;
                if (cout_char == '\n') {
                    std::cout << std::dec << "#" << cout_tid << ": " << ss_buf.str() << std::flush;
                    ss_buf.str("");
                }
                CHECK_ERR(api.fpgaReadMMIO64(device->fpga, 0, MMIO_STATUS, &status), {
                    return -1; 
                });
                cout_data = status >> STATUS_STATE_BITS;
            } while (cout_data & 0x1);
        }

        uint32_t state = status & ((1 << STATUS_STATE_BITS)-1);

        if (0 == state || 0 == timeout) {
            for (auto& buf : print_bufs) {
                auto str = buf.second.str();
                if (!str.empty()) {
                std::cout << "#" << buf.first << ": " << str << std::endl;
                }
            }
            if (state != 0) {
                fprintf(stdout, "[VXDRV] ready-wait timed out: state=%d\n", state);
            }
            break;
        }

        nanosleep(&sleep_time, nullptr);
        timeout -= sleep_time_ms;
    };

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = ((vx_buffer_t*)hbuffer);
    auto device = ((vx_device*)buffer->hdevice);
    auto& api = device->api;

    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned(buffer->io_addr + src_offset, CACHE_BLOCK_SIZE))
        return -1;

    // bound checking
    if (src_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > device->mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, MAX_TIMEOUT) != 0)
        return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG0, (buffer->io_addr + src_offset) >> ls_shift), {
        return -1; 
    });    
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG1, dev_maddr >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_WRITE), {
        return -1; 
    });

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, MAX_TIMEOUT) != 0)
        return -1;

    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = ((vx_buffer_t*)hbuffer);
    auto device = ((vx_device*)buffer->hdevice);
    auto& api = device->api;

    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned(buffer->io_addr + dest_offset, CACHE_BLOCK_SIZE))
        return -1; 

    // bound checking
    if (dest_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > device->mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, MAX_TIMEOUT) != 0)
        return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG0, (buffer->io_addr + dest_offset) >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG1, dev_maddr >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_READ), {
        return -1; 
    });

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, MAX_TIMEOUT) != 0)
        return -1;

    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;   

    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, MAX_TIMEOUT) != 0)
        return -1;    
  
    // start execution    
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_RUN), {
        return -1; 
    });

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;    
  
    // write DCR value
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG0, addr), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG1, value), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_DCR_WRITE), {
        return -1; 
    });

    // save the value
    device->dcrs.write(addr, value);

    return 0;
}