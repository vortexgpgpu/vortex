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

#ifndef __VX_VORTEX_H__
#define __VX_VORTEX_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;

// device caps ids
#define VX_CAPS_VERSION             0x0 
#define VX_CAPS_NUM_THREADS         0x1
#define VX_CAPS_NUM_WARPS           0x2
#define VX_CAPS_NUM_CORES           0x3
#define VX_CAPS_CACHE_LINE_SIZE     0x4
#define VX_CAPS_GLOBAL_MEM_SIZE     0x5
#define VX_CAPS_LOCAL_MEM_SIZE      0x6
#define VX_CAPS_KERNEL_BASE_ADDR    0x7
#define VX_CAPS_ISA_FLAGS           0x8

// device isa flags
#define VX_ISA_STD_A                (1ull << 0)
#define VX_ISA_STD_C                (1ull << 2)
#define VX_ISA_STD_D                (1ull << 3)
#define VX_ISA_STD_E                (1ull << 4)
#define VX_ISA_STD_F                (1ull << 5)
#define VX_ISA_STD_H                (1ull << 7)
#define VX_ISA_STD_I                (1ull << 8)
#define VX_ISA_STD_N                (1ull << 13)
#define VX_ISA_STD_Q                (1ull << 16)
#define VX_ISA_STD_S                (1ull << 18)
#define VX_ISA_STD_U                (1ull << 20)
#define VX_ISA_ARCH(flags)          (1 << (((flags >> 30) & 0x3) + 4))
#define VX_ISA_EXT_ICACHE           (1ull << 32)
#define VX_ISA_EXT_DCACHE           (1ull << 33)
#define VX_ISA_EXT_L2CACHE          (1ull << 34)
#define VX_ISA_EXT_L3CACHE          (1ull << 35)
#define VX_ISA_EXT_SMEM             (1ull << 36)

// device memory types
#define VX_MEM_TYPE_GLOBAL          0
#define VX_MEM_TYPE_LOCAL           1

// ready wait timeout
#define VX_MAX_TIMEOUT              (24*60*60*1000)   // 24 Hr

// open the device and connect to it
int vx_dev_open(vx_device_h* hdevice);

// Close the device when all the operations are done
int vx_dev_close(vx_device_h hdevice);

// return device configurations
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value);

// allocate device memory and return address
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int type, uint64_t* dev_addr);

// release device memory
int vx_mem_free(vx_device_h hdevice, uint64_t dev_addr);

// get device memory info
int vx_mem_info(vx_device_h hdevice, int type, uint64_t* mem_free, uint64_t* mem_used);

// Copy bytes from host to device memory
int vx_copy_to_dev(vx_device_h hdevice, uint64_t dev_addr, const void* host_ptr, uint64_t size);

// Copy bytes from device memory to host
int vx_copy_from_dev(vx_device_h hdevice, void* host_ptr, uint64_t dev_addr, uint64_t size);

// Start device execution
int vx_start(vx_device_h hdevice);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

// write device configuration registers
int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value);

////////////////////////////// UTILITY FUNCTIONS //////////////////////////////

// upload kernel bytes to device
int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size);

// upload kernel file to device
int vx_upload_kernel_file(vx_device_h hdevice, const char* filename);

// performance counters
int vx_dump_perf(vx_device_h hdevice, FILE* stream);
int vx_perf_counter(vx_device_h hdevice, int counter, int core_id, uint64_t* value);

#ifdef __cplusplus
}
#endif

#endif // __VX_VORTEX_H__
