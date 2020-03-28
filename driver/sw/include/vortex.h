#ifndef __VX_DRIVER_H__
#define __VX_DRIVER_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;

typedef void* vx_buffer_h;

#define VX_LOCAL_MEM_SIZE   0xffffffff

#define VX_ALLOC_BASE_ADDR  0x10000000

#define VX_KERNEL_BASE_ADDR 0x80000000

#define VX_CACHE_LINESIZE 64

// open the device and connect to it
int vx_dev_open(vx_device_h* hdevice);

// Close the device when all the operations are done
int vx_dev_close(vx_device_h hdevice);

// Allocate shared buffer with device
int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer);

// Get host pointer address  
volatile void* vx_host_ptr(vx_buffer_h hbuffer);

// release buffer
int vx_buf_release(vx_buffer_h hbuffer);

// allocate device memory and return address
int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr);

// Copy bytes from device local memory to buffer
int vx_flush_caches(vx_device_h hdevice, size_t dev_maddr, size_t size);

// Copy bytes from buffer to device local memory
int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset);

// Copy bytes from device local memory to buffer
int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dst_offset);

// Start device execution
int vx_start(vx_device_h hdevice);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, long long timeout);

////////////////////////////// UTILITY FUNCIONS ///////////////////////////////

// upload kernel bytes to device
int vx_upload_kernel_bytes(vx_device_h device, const void* content, size_t size);

// upload kernel file to device
int vx_upload_kernel_file(vx_device_h device, const char* filename);

#ifdef __cplusplus
}
#endif

#endif // __VX_DRIVER_H__
