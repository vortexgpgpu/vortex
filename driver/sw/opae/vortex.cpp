#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <uuid/uuid.h>

#include <opae/fpga.h>
#include <vortex.h>
#include "vortex_afu.h"

// MMIO Address Mappings
#define MMIO_COPY_IO_ADDRESS    0X120
#define MMIO_COPY_AVM_ADDRESS   0x100
#define MMIO_COPY_DATA_SIZE     0X118

#define MMIO_CMD_TYPE           0X110
#define MMIO_READY_FOR_CMD      0X198

#define MMIO_CMD_TYPE_READ      0 
#define MMIO_CMD_TYPE_WRITE     1 
#define MMIO_CMD_TYPE_START     2 
#define MMIO_CMD_TYPE_SNOOP     3 

#define CHECK_RES(_expr)                                            \
   do {                                                             \
     fpga_result res = _expr;                                       \
     if (res == FPGA_OK)                                            \
       break;                                                       \
     printf("OPAE Error: '%s' returned %d!\n", #_expr, (int)res);   \
     return -1;                                                     \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_device_ {
    fpga_handle fpga;
    size_t mem_allocation;
} vx_device_t;

typedef struct vx_buffer_ {
    uint64_t wsid;
    volatile void* host_ptr;
    uint64_t io_addr;
    fpga_handle fpga;
    size_t size;
} vx_buffer_t;

static size_t align_size(size_t size) {
    return VX_CACHE_LINESIZE * ((size + VX_CACHE_LINESIZE - 1) / VX_CACHE_LINESIZE);
}

///////////////////////////////////////////////////////////////////////////////

// Search for an accelerator matching the requested UUID and connect to it
// Convert this to void if required as storing the fpga_handle to params variable
extern int vx_dev_open(vx_device_h* hdevice) {
    fpga_properties filter = NULL;
    fpga_result res;
    fpga_guid guid;
    fpga_token accel_token;
    uint32_t num_matches;
    fpga_handle accel_handle;
    vx_device_t* device;

    if (NULL == hdevice)
        return  -1;

    // Set up a filter that will search for an accelerator
    fpgaGetProperties(NULL, &filter);
    fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);

    // Add the desired UUID to the filter
    uuid_parse(AFU_ACCEL_UUID, guid);
    fpgaPropertiesSetGUID(filter, guid);

    // Do the search across the available FPGA contexts
    num_matches = 1;
    fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);

    // Not needed anymore
    fpgaDestroyProperties(&filter);

    if (num_matches < 1) {
        fprintf(stderr, "Accelerator %s not found!\n", AFU_ACCEL_UUID);
        return NULL;
    }

    // Open accelerator
    res = fpgaOpen(accel_token, &accel_handle, 0);
    if (FPGA_OK != res) {
        return NULL;
    }

    // Done with token
    fpgaDestroyToken(&accel_token);

    // allocate device object
    device = (vx_device_t*)malloc(sizeof(vx_device_t));
    if (NULL == device) {
        fpgaClose(accel_handle);
        return NULL;
    }

    device->fpga = accel_handle;
    device->mem_allocation = VX_ALLOC_BASE_ADDR;

    *hdevice = device;

    return 0;
}

// Close the fpga when all the operations are done
extern int vx_dev_close(vx_device_h hdevice) {
    if (NULL == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    fpgaClose(device->fpga);

    free(device);

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (NULL == hdevice 
     || NULL == dev_maddr
     || 0 >= size)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);
    
    size_t asize = align_size(size);
    if (device->mem_allocation + asize > VX_ALLOC_BASE_ADDR)
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

    if (NULL == hdevice
     || 0 >= size
     || NULL == hbuffer)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    size_t asize = align_size(size);

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
    if (NULL == buffer) {
        fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    }

    buffer->wsid = wsid;
    buffer->host_ptr = host_ptr;
    buffer->io_addr = io_addr;
    buffer->fpga = device->fpga;
    buffer->size = size;

    *hbuffer = buffer;

    return 0;
}

extern volatile void* vx_host_ptr(vx_buffer_h hbuffer) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    if (NULL == buffer)
        return NULL;

    return buffer->host_ptr;
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    if (NULL == buffer)
        return -1;

    fpgaReleaseBuffer(buffer->fpga, buffer->wsid);

    free(buffer);

    return 0;
}

// Check if HW is ready for SW
static int ready_for_sw(fpga_handle hdevice) {
    uint64_t data = 0;
    struct timespec sleep_time; 
    
#ifdef USE_ASE
    sleep_time.tv_sec = 1;
    sleep_time.tv_nsec = 0;
#else
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;
#endif

    do {
        CHECK_RES(fpgaReadMMIO64(hdevice, 0, MMIO_READY_FOR_CMD, &data));
        nanosleep(&sleep_time, NULL);
    } while (data != 0x1);

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset) {
    if (NULL == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    // bound checking
    if (size + src_offset > buffer->size)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(buffer->fpga) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_AVM_ADDRESS, dev_maddr));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_IO_ADDRESS, (buffer->io_addr + src_offset)/VX_CACHE_LINESIZE));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_DATA_SIZE, size));   
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_CMD_TYPE, MMIO_CMD_TYPE_WRITE));

    // Wait for the write operation to finish
    return ready_for_sw(buffer->fpga);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dest_offset) {
    if (NULL == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    // bound checking
    if (size + dest_offset > buffer->size)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(buffer->fpga) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_AVM_ADDRESS, dev_maddr));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_IO_ADDRESS, (buffer->io_addr + dest_offset)/VX_CACHE_LINESIZE));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_DATA_SIZE, size));   
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_CMD_TYPE, MMIO_CMD_TYPE_READ));

    // Wait for the write operation to finish
    return ready_for_sw(buffer->fpga);
}

extern int vx_flush_caches(vx_device_h hdevice, size_t dev_maddr, size_t size) {
    if (NULL == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    // bound checking
    if (size + src_offset > buffer->size)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(buffer->fpga) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_AVM_ADDRESS, dev_maddr));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_IO_ADDRESS, (buffer->io_addr + src_offset)/VX_CACHE_LINESIZE));
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_COPY_DATA_SIZE, size));   
    CHECK_RES(fpgaWriteMMIO64(buffer->fpga, 0, MMIO_CMD_TYPE, MMIO_CMD_TYPE_SNOOP));

    // Wait for the write operation to finish
    return ready_for_sw(buffer->fpga);
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (NULL == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    // Ensure ready for new command
    if (ready_for_sw(device->fpga) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CMD_TYPE, MMIO_CMD_TYPE_START));

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (NULL == hdevice)
        return -1;
    
    vx_device_t *device = ((vx_device_t*)hdevice);

    uint64_t data = 0;
    struct timespec sleep_time; 

#ifdef USE_ASE
    sleep_time.tv_sec = 1;
    sleep_time.tv_nsec = 0;
#else
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;
#endif

    // to milliseconds
    long long sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);
    
    do {
        CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_READY_FOR_CMD, &data));
        nanosleep(&sleep_time, NULL);
        sleep_time_ms -= sleep_time_ms;
        if (timeout <= sleep_time_ms)
            break;        
    } while (data != 0x1);

    return 0;
}