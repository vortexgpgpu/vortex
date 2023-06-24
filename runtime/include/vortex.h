#ifndef __VX_VORTEX_H__
#define __VX_VORTEX_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;
typedef void* vx_buffer_h;

// device caps ids
#define VX_CAPS_VERSION           0x0 
#define VX_CAPS_NUM_THREADS       0x1
#define VX_CAPS_NUM_WARPS         0x2
#define VX_CAPS_NUM_CORES         0x3
#define VX_CAPS_NUM_CLUSTERS      0x4
#define VX_CAPS_CACHE_LINE_SIZE   0x5
#define VX_CAPS_LOCAL_MEM_SIZE    0x6
#define VX_CAPS_KERNEL_BASE_ADDR  0x7
#define VX_CAPS_ISA_FLAGS         0x8

// device isa flags
#define VX_ISA_STD_A              (1ull << 0)
#define VX_ISA_STD_C              (1ull << 2)
#define VX_ISA_STD_D              (1ull << 3)
#define VX_ISA_STD_E              (1ull << 4)
#define VX_ISA_STD_F              (1ull << 5)
#define VX_ISA_STD_H              (1ull << 7)
#define VX_ISA_STD_I              (1ull << 8)
#define VX_ISA_STD_N              (1ull << 13)
#define VX_ISA_STD_Q              (1ull << 16)
#define VX_ISA_STD_S              (1ull << 18)
#define VX_ISA_STD_U              (1ull << 20)
#define VX_ISA_BASE(flags)        (1 << (((flags >> 30) & 0x3) + 4))
#define VX_ISA_EXT_TEX            (1ull << 32)
#define VX_ISA_EXT_RASTER         (1ull << 33)
#define VX_ISA_EXT_ROP            (1ull << 34)
#define VX_ISA_EXT_IMADD          (1ull << 35)

#define MAX_TIMEOUT               (24*60*60*1000)   // 24hr 

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

// get device memory info
int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_total);

// Copy bytes from buffer to device local memory
int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset);

// Copy bytes from device local memory to buffer
int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dst_offset);

// Start device execution
int vx_start(vx_device_h hdevice);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

// write device configuration registers
int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value);

////////////////////////////// UTILITY FUNCIONS ///////////////////////////////

// upload kernel bytes to device
int vx_upload_kernel_bytes(vx_device_h device, const void* content, uint64_t size);

// upload kernel file to device
int vx_upload_kernel_file(vx_device_h device, const char* filename);

// performance counters
int vx_dump_perf(vx_device_h device, FILE* stream);
int vx_perf_counter(vx_device_h device, int counter, int core_id, uint64_t* value);

//////////////////////////// DEPRECATED FUNCTIONS /////////////////////////////
int vx_alloc_dev_mem(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr);
int vx_alloc_shared_mem(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer);
int vx_buf_release(vx_buffer_h hbuffer);

#ifdef __cplusplus
}
#endif

#endif // __VX_VORTEX_H__
