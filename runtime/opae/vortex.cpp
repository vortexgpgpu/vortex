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
#include <memory>
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

#define RAM_PAGE_SIZE       4096

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
    vx_device() : 
        staging_wsid(0), 
        staging_ioaddr(0), 
        staging_ptr(nullptr),
        staging_size(0) 
    {}

    ~vx_device() {}

    int ensure_staging(uint64_t size) {
        size_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (staging_size >= asize)
            return 0;

        if (staging_size != 0) {
            // release existing buffer
            api.fpgaReleaseBuffer(fpga, staging_wsid);
            staging_size = 0;
        }

        // allocate new buffer
        CHECK_ERR(api.fpgaPrepareBuffer(fpga, asize, (void**)&staging_ptr, &staging_wsid, 0), {
            return -1;
        });

        // get the physical address of the buffer in the accelerator
        CHECK_ERR(api.fpgaGetIOAddress(fpga, staging_wsid, &staging_ioaddr), {
            api.fpgaReleaseBuffer(fpga, staging_wsid);
            return -1;
        });

        staging_size = asize;

        return 0;
    }

    opae_drv_api_t api;
    fpga_handle fpga;
    std::shared_ptr<vortex::MemoryAllocator> global_mem;
    std::shared_ptr<vortex::MemoryAllocator> local_mem;
    DeviceConfig dcrs;
    uint64_t dev_caps;
    uint64_t isa_caps;
    uint64_t global_mem_size;
    uint64_t staging_wsid;
    uint64_t staging_ioaddr;
    uint8_t* staging_ptr;
    uint64_t staging_size;
};

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
        // retrieve FPGA global memory size
        CHECK_ERR(api.fpgaPropertiesGetLocalMemorySize(filter, &device->global_mem_size), {
            // assume 8GB as default
            device->global_mem_size = GLOBAL_MEM_SIZE;
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

    device->global_mem = std::make_shared<vortex::MemoryAllocator>(
        ALLOC_BASE_ADDR, ALLOC_MAX_ADDR - ALLOC_BASE_ADDR, RAM_PAGE_SIZE, CACHE_BLOCK_SIZE);

    uint64_t local_mem_size = 0;
    vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
    if (local_mem_size <= 1) {        
        device->local_mem = std::make_shared<vortex::MemoryAllocator>(
            SMEM_BASE_ADDR, local_mem_size, RAM_PAGE_SIZE, 1);
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

    // release staging buffer
    if (device->staging_size != 0) {
        api.fpgaReleaseBuffer(device->fpga, device->staging_wsid);
        device->staging_size = 0;
    }

    // close the device
    api.fpgaClose(device->fpga);

    delete device;

    drv_close();

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int type, uint64_t* dev_addr) {
    if (nullptr == hdevice 
     || nullptr == dev_addr
     || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);
    if (type == VX_MEM_TYPE_GLOBAL) {
        return device->global_mem->allocate(size, dev_addr);
    } else if (type == VX_MEM_TYPE_LOCAL) {        
        return device->local_mem->allocate(size, dev_addr);
    }
    return -1;
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_addr) {
    if (nullptr == hdevice)
        return -1;

    if (0 == dev_addr)
        return 0;

    auto device = ((vx_device*)hdevice);
    if (dev_addr >= SMEM_BASE_ADDR) {
        return device->local_mem->release(dev_addr);
    } else {    
        return device->global_mem->release(dev_addr);
    }
}

extern int vx_mem_info(vx_device_h hdevice, int type, uint64_t* mem_free, uint64_t* mem_used) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);    
    if (type == VX_MEM_TYPE_GLOBAL) {
        if (mem_free)
            *mem_free = device->global_mem->free();
        if (mem_used)
            *mem_used = device->global_mem->allocated();
    } else if (type == VX_MEM_TYPE_LOCAL) {
        if (mem_free)
            *mem_free = device->local_mem->free();
        if (mem_used)
            *mem_free = device->local_mem->allocated();
    } else {
        return -1;
    }
    return 0;
}

extern int vx_copy_to_dev(vx_device_h hdevice, uint64_t dev_addr, const void* host_ptr, uint64_t size) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    auto& api = device->api;

    if (device->ensure_staging(size) != 0)
        return -1; 

    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
        return -1;

    // bound checking
    if (dev_addr + asize > device->global_mem_size)
        return -1;

    // ensure ready for new command
    if (vx_ready_wait(hdevice, VX_MAX_TIMEOUT) != 0)
        return -1;

    // update staging buffer
    memcpy(device->staging_ptr, host_ptr, size);

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG0, device->staging_ioaddr >> ls_shift), {
        return -1; 
    });    
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG1, dev_addr >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_WRITE), {
        return -1; 
    });

    // Wait for the write operation to finish
    if (vx_ready_wait(hdevice, VX_MAX_TIMEOUT) != 0)
        return -1;

    return 0;
}

extern int vx_copy_from_dev(vx_device_h hdevice, void* host_ptr, uint64_t dev_addr, uint64_t size) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;
    auto& api = device->api;

    if (device->ensure_staging(size) != 0)
        return -1;

    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
        return -1;

    // bound checking
    if (dev_addr + asize > device->global_mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, VX_MAX_TIMEOUT) != 0)
        return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG0, device->staging_ioaddr >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG1, dev_addr >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_ARG2, asize >> ls_shift), {
        return -1; 
    });
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_READ), {
        return -1; 
    });

    // wait for the write operation to finish
    if (vx_ready_wait(hdevice, VX_MAX_TIMEOUT) != 0)
        return -1;

    // read staging buffer
    memcpy(host_ptr, device->staging_ptr, size);

    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;   

    auto device = ((vx_device*)hdevice);
    auto& api = device->api;

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, VX_MAX_TIMEOUT) != 0)
        return -1;    
  
    // start execution    
    CHECK_ERR(api.fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_RUN), {
        return -1; 
    });

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
