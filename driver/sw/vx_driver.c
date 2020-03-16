#include "vx_driver.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <uuid/uuid.h>

#include <opae/fpga.h>

// MMIO Address Mappings
#define AFU_ID                  AFU_ACCEL_UUID

#define MMIO_COPY_IO_ADDRESS    0X120
#define MMIO_COPY_AVM_ADDRESS   0x100
#define MMIO_COPY_DATA_SIZE     0X118

#define MMIO_CMD_TYPE           0X110		// MMIO location set by SW to denote read/write. read: 3; write: 1; vortex: 7
#define MMIO_READY_FOR_CMD      0X198

#define CHECK_RES(_expr)                                            \
   do {                                                             \
     fpga_result res = _expr;                                       \
     if (res == FPGA_OK)                                            \
       break;                                                       \
     printf("OPAE Error: '%s' returned %d!\n", #_expr, (int)res);   \
     return -1;                                                     \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_buffer_ {
    uint64_t wsid;
    volatile void* host_ptr;
    uint64_t io_addr;
    fpga_handle hdevice;
    size_t size;
} vx_buffer_t;

static size_t align_size(size_t size) {
    return VX_CACHE_LINESIZE * ((size + VX_CACHE_LINESIZE - 1) / VX_CACHE_LINESIZE);
}

///////////////////////////////////////////////////////////////////////////////

// Search for an accelerator matching the requested UUID and connect to it
// Convert this to void if required as storing the fpga_handle to params variable
extern vx_device_h vx_dev_open(const char *accel_uuid) {
    fpga_properties filter = NULL;
    fpga_result res;
    fpga_guid guid;
    fpga_token accel_token;
    uint32_t num_matches;
    fpga_handle accel_handle;

    // Set up a filter that will search for an accelerator
    fpgaGetProperties(NULL, &filter);
    fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);

    // Add the desired UUID to the filter
    uuid_parse(accel_uuid, guid);
    fpgaPropertiesSetGUID(filter, guid);

    // Do the search across the available FPGA contexts
    num_matches = 1;
    fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);

    // Not needed anymore
    fpgaDestroyProperties(&filter);

    if (num_matches < 1) {
        fprintf(stderr, "Accelerator %s not found!\n", accel_uuid);
        return NULL;
    }

    // Open accelerator
    res = fpgaOpen(accel_token, &accel_handle, 0);
    if (FPGA_OK != res) {
        return NULL;
    }

    // Done with token
    fpgaDestroyToken(&accel_token);

    return accel_handle;
}

// Close the fpga when all the operations are done
extern int vx_dev_close(vx_device_h hdevice) {
    if (NULL == hdevice)
        return -1;

    fpgaClose(hdevice);

    return 0;
}

extern vx_buffer_h vx_buf_alloc(vx_device_h hdevice, size_t size) {
    fpga_result res;
    void* host_ptr;
    uint64_t wsid;
    uint64_t io_addr;
    vx_buffer_t* buffer;

    if (NULL == hdevice)
        return NULL;

    size_t asize = align_size(size);

    res = fpgaPrepareBuffer(hdevice, asize, &host_ptr, &wsid, 0);
    if (FPGA_OK != res) {
        return NULL;
    }

    // Get the physical address of the buffer in the accelerator
    res = fpgaGetIOAddress(hdevice, wsid, &io_addr);
    if (FPGA_OK != res) {
        fpgaReleaseBuffer(hdevice, wsid);
        return NULL;
    }

    buffer = (vx_buffer_t*)malloc(sizeof(vx_buffer_t));
    if (NULL == buffer) {
        fpgaReleaseBuffer(hdevice, wsid);
        return NULL;
    }

    buffer->wsid = wsid;
    buffer->host_ptr = host_ptr;
    buffer->io_addr = io_addr;
    buffer->hdevice = hdevice;
    buffer->size = size;

    return (vx_buffer_h)buffer;
}

extern volatile void* vs_buf_ptr(vx_buffer_h hbuffer) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    if (NULL == buffer)
        return NULL;

    return buffer->host_ptr;
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    if (NULL == buffer)
        return -1;

    fpgaReleaseBuffer(buffer->hdevice, buffer->wsid);

    free(hbuffer);

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

extern int vx_copy_to_fpga(vx_buffer_h hbuffer, size_t dest_addr, size_t size, size_t src_offset) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    // bound checking
    if (size + src_offset > buffer->size)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(buffer->hdevice) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_AVM_ADDRESS, dest_addr));
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_IO_ADDRESS, (buffer->io_addr + src_offset)/VX_CACHE_LINESIZE));
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_DATA_SIZE, size));   
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_CMD_TYPE, 1)); // WRITE CMD

    // Wait for the write operation to finish
    return ready_for_sw(buffer->hdevice);
}

extern int vx_copy_from_fpga(vx_buffer_h hbuffer, size_t src_addr, size_t size, size_t dest_offset) {
    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    // bound checking
    if (size + dest_offset > buffer->size)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(buffer->hdevice) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_AVM_ADDRESS, src_addr));
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_IO_ADDRESS, (buffer->io_addr + dest_offset)/VX_CACHE_LINESIZE));
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_COPY_DATA_SIZE, size));   
    CHECK_RES(fpgaWriteMMIO64(buffer->hdevice, 0, MMIO_CMD_TYPE, 3)); // READ CMD

    // Wait for the write operation to finish
    return ready_for_sw(buffer->hdevice);
}

extern int vx_start(vx_device_h hdevice) {
    if (NULL == hdevice)
        return -1;

    // Ensure ready for new command
    if (ready_for_sw(hdevice) != 0)
        return -1;

    CHECK_RES(fpgaWriteMMIO64(hdevice, 0, MMIO_CMD_TYPE, 7)); // START CMD    

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (NULL == hdevice)
        return -1;

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
        CHECK_RES(fpgaReadMMIO64(hdevice, 0, MMIO_READY_FOR_CMD, &data));
        nanosleep(&sleep_time, NULL);
        sleep_time_ms -= sleep_time_ms;
        if (timeout <= sleep_time_ms)
            break;        
    } while (data != 0x1);

    return 0;
}
