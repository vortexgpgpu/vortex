#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
#include <assert.h>
#include <cmath>

#if defined(USE_FPGA) || defined(USE_ASE) 
#include <opae/fpga.h>
#include <uuid/uuid.h>
#elif defined(USE_VLSIM)
#include "vlsim/fpga.h"
#endif

#include <vortex.h>
#include <VX_config.h>
#include "vortex_afu.h"

#ifdef SCOPE
#include "vx_scope.h"
#endif

#define CHECK_RES(_expr)                                \
   do {                                                 \
     fpga_result res = _expr;                           \
     if (res == FPGA_OK)                                \
       break;                                           \
     printf("[VXDRV] Error: '%s' returned %d, %s!\n",   \
            #_expr, (int)res, fpgaErrStr(res));         \
     return -1;                                         \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

#define CMD_MEM_READ        AFU_IMAGE_CMD_MEM_READ
#define CMD_MEM_WRITE       AFU_IMAGE_CMD_MEM_WRITE
#define CMD_RUN             AFU_IMAGE_CMD_RUN
#define CMD_CSR_READ        AFU_IMAGE_CMD_CSR_READ
#define CMD_CSR_WRITE       AFU_IMAGE_CMD_CSR_WRITE

#define MMIO_CMD_TYPE       (AFU_IMAGE_MMIO_CMD_TYPE * 4)
#define MMIO_IO_ADDR        (AFU_IMAGE_MMIO_IO_ADDR * 4)
#define MMIO_MEM_ADDR       (AFU_IMAGE_MMIO_MEM_ADDR * 4)
#define MMIO_DATA_SIZE      (AFU_IMAGE_MMIO_DATA_SIZE * 4)
#define MMIO_STATUS         (AFU_IMAGE_MMIO_STATUS * 4)
#define MMIO_CSR_CORE       (AFU_IMAGE_MMIO_CSR_CORE * 4)
#define MMIO_CSR_ADDR       (AFU_IMAGE_MMIO_CSR_ADDR * 4)
#define MMIO_CSR_DATA       (AFU_IMAGE_MMIO_CSR_DATA * 4)
#define MMIO_CSR_READ       (AFU_IMAGE_MMIO_CSR_READ * 4)

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_device_ {
    fpga_handle fpga;
    size_t mem_allocation;
    unsigned implementation_id;
    unsigned num_cores;
    unsigned num_warps;
    unsigned num_threads;
} vx_device_t;

typedef struct vx_buffer_ {
    uint64_t wsid;
    void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    size_t size;
} vx_buffer_t;

inline size_t align_size(size_t size, size_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(size_t addr, size_t alignment) {
    assert(0 == (alignment & (alignment - 1)));
    return 0 == (addr & (alignment - 1));
}

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, unsigned caps_id, unsigned *value) {
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = device->implementation_id;
        break;
    case VX_CAPS_MAX_CORES:
        *value = device->num_cores;
        break;
    case VX_CAPS_MAX_WARPS:
        *value = device->num_warps;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = device->num_threads;
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
        *value = STARTUP_ADDR;
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

    fpga_handle accel_handle;    
    vx_device_t* device;   

#ifndef USE_VLSIM
    fpga_result res;    
    fpga_token accel_token;
    fpga_properties filter = nullptr;    
    fpga_guid guid; 
    uint32_t num_matches;
    
    // Set up a filter that will search for an accelerator
    CHECK_RES(fpgaGetProperties(nullptr, &filter));
    res = fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);
    if (res != FPGA_OK) {
        fprintf(stderr, "[VXDRV] Error: fpgaGetProperties() returned %d, %s!\n", (int)res, fpgaErrStr(res));
        fpgaDestroyProperties(&filter);
        return -1;
    }

    // Add the desired UUID to the filter
    uuid_parse(AFU_ACCEL_UUID, guid);
    res = fpgaPropertiesSetGUID(filter, guid);
    if (res != FPGA_OK) {
        fprintf(stderr, "[VXDRV] Error: fpgaPropertiesSetGUID() returned %d, %s!\n", (int)res, fpgaErrStr(res));
        fpgaDestroyProperties(&filter);
        return -1;
    }

    // Do the search across the available FPGA contexts
    num_matches = 1;
    res = fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);
    if (res != FPGA_OK) {
        fprintf(stderr, "[VXDRV] Error: fpgaEnumerate() returned %d, %s!\n", (int)res, fpgaErrStr(res));
        fpgaDestroyProperties(&filter);
        return -1;
    }

    // Not needed anymore
    fpgaDestroyProperties(&filter);

    if (num_matches < 1) {
        fprintf(stderr, "[VXDRV] Error: accelerator %s not found!\n", AFU_ACCEL_UUID);
        fpgaDestroyToken(&accel_token);
        return -1;
    }

    // Open accelerator
    res = fpgaOpen(accel_token, &accel_handle, 0);
    if (res != FPGA_OK) {
        fprintf(stderr, "[VXDRV] Error: fpgaOpen() returned %d, %s!\n", (int)res, fpgaErrStr(res));
        fpgaDestroyToken(&accel_token);
        return -1;
    }

    // Done with token
    fpgaDestroyToken(&accel_token);
#else
    // Open accelerator
    CHECK_RES(fpgaOpen(NULL, &accel_handle, 0));
#endif

    // allocate device object
    device = (vx_device_t*)malloc(sizeof(vx_device_t));
    if (nullptr == device) {
        fpgaClose(accel_handle);
        return -1;
    }

    device->fpga = accel_handle;
    device->mem_allocation = ALLOC_BASE_ADDR;

    {   
        // Load device CAPS
        int ret = 0;
        ret |= vx_csr_get(device, 0, CSR_MIMPID, &device->implementation_id);
        ret |= vx_csr_get(device, 0, CSR_NC, &device->num_cores);        
        ret |= vx_csr_get(device, 0, CSR_NW, &device->num_warps);        
        ret |= vx_csr_get(device, 0, CSR_NT, &device->num_threads);        
        if (ret != FPGA_OK) {
            fpgaClose(accel_handle);
            return ret;
        }
    #ifndef NDEBUG    
        fprintf(stdout, "[VXDRV] DEVCAPS: version=%d, num_cores=%d, num_warps=%d, num_threads=%d\n", 
                device->implementation_id, device->num_cores, device->num_warps, device->num_threads);
    #endif
    }
    
#ifdef SCOPE
    {
        int ret = vx_scope_start(accel_handle, 0, -1);
        if (ret != 0) {
            fpgaClose(accel_handle);
            return ret;
        }
    }
#endif    

    *hdevice = device;

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

#ifdef SCOPE
    vx_scope_stop(device->fpga);
#endif

#ifdef DUMP_PERF_STATS
    vx_dump_perf(device, stdout);
#endif

    fpgaClose(device->fpga);

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    size_t dev_mem_size = LOCAL_MEM_SIZE;
    size_t asize = align_size(size, CACHE_BLOCK_SIZE);
    
    if (device->mem_allocation + asize > dev_mem_size)
        return -1;   

    *dev_maddr = device->mem_allocation;
    device->mem_allocation += asize;

    return 0;
}

extern int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer) {
    fpga_result res;
    void* host_ptr;
    uint64_t wsid;
    uint64_t io_addr;
    vx_buffer_t* buffer;

    if (nullptr == hdevice
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    size_t asize = align_size(size, CACHE_BLOCK_SIZE);

    res = fpgaPrepareBuffer(device->fpga, asize, &host_ptr, &wsid, 0);
    if (FPGA_OK != res) {
        return -1;
    }

    // Get the physical address of the buffer in the accelerator
    res = fpgaGetIOAddress(device->fpga, wsid, &io_addr);
    if (FPGA_OK != res) {
        fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    }

    // allocate buffer object
    buffer = (vx_buffer_t*)malloc(sizeof(vx_buffer_t));
    if (nullptr == buffer) {
        fpgaReleaseBuffer(device->fpga, wsid);
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

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
#ifdef USE_VLSIM
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);
    fpgaFlush(device);
#endif

    return buffer->host_ptr;
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    fpgaReleaseBuffer(device->fpga, buffer->wsid);

    free(buffer);

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (nullptr == hdevice)
        return -1;
    
    vx_device_t *device = ((vx_device_t*)hdevice);

    struct timespec sleep_time; 

#if defined(USE_ASE)
    sleep_time.tv_sec = 1;
    sleep_time.tv_nsec = 0;
#else
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;
#endif

    // to milliseconds
    long long sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);
    
    for (;;) {
        uint64_t data;
        CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_STATUS, &data));
        if (0 == data || 0 == timeout) {
            if (data != 0) {
                fprintf(stdout, "[VXDRV] ready-wait timed out: status=%ld\n", data);
            }
            break;
        }
        nanosleep(&sleep_time, nullptr);
        timeout -= sleep_time_ms;
    };

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t *buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    size_t dev_mem_size = LOCAL_MEM_SIZE; 
    size_t asize = align_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned(buffer->io_addr + src_offset, CACHE_BLOCK_SIZE))
        return -1;

    // bound checking
    if (src_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_IO_ADDR, (buffer->io_addr + src_offset) >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_MEM_ADDR, dev_maddr >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_DATA_SIZE, asize >> ls_shift));   
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_WRITE));

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dest_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t *buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    size_t dev_mem_size = LOCAL_MEM_SIZE;  
    size_t asize = align_size(size, CACHE_BLOCK_SIZE);

    // check alignment
    if (!is_aligned(dev_maddr, CACHE_BLOCK_SIZE))
        return -1;
    if (!is_aligned(buffer->io_addr + dest_offset, CACHE_BLOCK_SIZE))
        return -1; 

    // bound checking
    if (dest_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_IO_ADDR, (buffer->io_addr + dest_offset) >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_MEM_ADDR, dev_maddr >> ls_shift));    
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_DATA_SIZE, asize >> ls_shift));   
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_MEM_READ));

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;   

    vx_device_t *device = ((vx_device_t*)hdevice);

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;    
  
    // start execution    
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_RUN));

    return 0;
}

// set device constant registers
extern int vx_csr_set(vx_device_h hdevice, int core_id, int addr, unsigned value) {
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;    
  
    // write CSR value
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CORE, core_id));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_ADDR, addr));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_DATA, value));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_CSR_WRITE));

    return 0;
}

// get device constant registers
extern int vx_csr_get(vx_device_h hdevice, int core_id, int addr, unsigned* value) {
    if (nullptr == hdevice || nullptr == value)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;

    // write CSR value    
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CORE, core_id));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_ADDR, addr));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, CMD_CSR_READ));    

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;

    uint64_t value64;
    CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_READ, &value64));
    *value = (unsigned)value64;

    return 0;
}