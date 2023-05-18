#ifndef __VX_DRIVER_H__
#define __VX_DRIVER_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>

using std::vector;

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;

typedef void* vx_buffer_h;

typedef struct subpacket_ { // for second half of header packet, these variables have different meanings
    uint64_t mmio_io_addr; // for header packet 2: mmio_cmdb_info
    uint64_t mmio_mem_addr; // for header packet 2: mmio_cmdb_db1
    uint64_t mmio_data_size; // for header packet 2: mmio_cmdb_db2
    uint64_t mmio_cmd_type; // for header packet 2: mmio_cmdb_db3
} subpacket;

class cmdbuffer {
    uint64_t maxBufferSize;
public:
    uint64_t bufferCount;
    vx_buffer_h buffer;
    vector<subpacket> fifo; 
    vx_buffer_h done_flag;
    cmdbuffer(int bufSize, vx_device_h device);
    ~cmdbuffer() {}
    
    bool createHeaderPacket(bool barrier);
    bool appendToCmdBuffer(subpacket subpkt);
    void displayCmdBuffer();
    bool flushCmdBuffer();
    void resetCmdBuffer();
};

// device caps ids
#define VX_CAPS_VERSION           0x0 
#define VX_CAPS_MAX_CORES         0x1
#define VX_CAPS_MAX_WARPS         0x2
#define VX_CAPS_MAX_THREADS       0x3
#define VX_CAPS_CACHE_LINE_SIZE   0x4
#define VX_CAPS_LOCAL_MEM_SIZE    0x5
#define VX_CAPS_ALLOC_BASE_ADDR   0x6
#define VX_CAPS_KERNEL_BASE_ADDR  0x7

#define MAX_TIMEOUT               (60*60*1000)   // 1hr 

// open the device and connect to it
int vx_dev_open(vx_device_h* hdevice);

// Close the device when all the operations are done
int vx_dev_close(vx_device_h hdevice);

// return device configurations
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value);

// Allocate shared buffer with device
int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer);

// release buffer
int vx_buf_free(vx_buffer_h hbuffer);

// Get host pointer address  
void* vx_host_ptr(vx_buffer_h hbuffer);

// allocate device memory and return address
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr);

// release device memory
int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr);

// Create and allocate command buffer
cmdbuffer* vx_create_command_buffer(uint64_t buf_size, vx_device_h device);

// Copy bytes from buffer to device local memory
int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset);

// Command buffer version of copy_to_dev
int vx_new_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset, cmdbuffer *cmdBuf, uint64_t cmd_type);

// Flush command buffer contents to Vortex
int vx_flush(cmdbuffer *cmdBuf);

// Wait for command buffer to finish
int cmdbuffer_wait(cmdbuffer *cmdBuf);

// Copy bytes from device local memory to buffer
int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dst_offset);

// Command Buffer version of vx_start
int vx_new_start(vx_device_h hdevice, cmdbuffer *cmdBuf);

// Start device execution
int vx_start(vx_device_h hdevice);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

////////////////////////////// UTILITY FUNCIONS ///////////////////////////////

// upload kernel bytes to device
int vx_upload_kernel_bytes(vx_device_h device, const void* content, uint64_t size, cmdbuffer* cmdBuf);

// upload kernel file to device
int vx_upload_kernel_file(vx_device_h device, const char* filename, cmdbuffer* cmdBuf);

// dump performance counters
int vx_dump_perf(vx_device_h device, FILE* stream);

//////////////////////////// DEPRECATED FUNCTIONS /////////////////////////////
int vx_alloc_dev_mem(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr);
int vx_alloc_shared_mem(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer);
int vx_buf_release(vx_buffer_h hbuffer);

#ifdef __cplusplus
}
#endif

#endif // __VX_DRIVER_H__
