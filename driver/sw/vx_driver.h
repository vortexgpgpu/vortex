#ifndef __VX_DRIVER_H__
#define __VX_DRIVER_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;

typedef void* vx_buffer_h;

#define VX_CACHE_LINESIZE 64

// open the device and connect to it
vx_device_h vx_dev_open();

// Close the device when all the operations are done
int vx_dev_close(vx_device_h hdevice);

// Allocate shared buffer with device
vx_buffer_h vx_buf_alloc(vx_device_h hdevice, size_t size);

// Get host pointer address  
void* vs_buf_ptr(vx_buffer_h hbuffer);

// release buffer
int vx_buf_release(vx_buffer_h hbuffer);

// Copy bytes from buffer to device local memory
int vx_copy_to_fpga(vx_buffer_h hbuffer, size_t dest_addr, size_t size, size_t src_offset);

// Copy bytes from device local memory to buffer
int vx_copy_from_fpga(vx_buffer_h hbuffer, size_t src_addr, size_t size, size_t dst_offset);

// Start device execution
int vx_start(vx_device_h hdevice);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, long long timeout);

#ifdef __cplusplus
}
#endif

#endif // __VX_DRIVER_H__
