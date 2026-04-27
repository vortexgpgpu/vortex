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
#include <VX_config.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vx_device_h;
typedef void* vx_buffer_h;

// device caps ids
#define VX_CAPS_VERSION             0x0   // implementation version
#define VX_CAPS_NUM_THREADS         0x1   // number of threads per warp
#define VX_CAPS_NUM_WARPS           0x2   // number of warps per core
#define VX_CAPS_NUM_CORES           0x3   // number of total cores
#define VX_CAPS_CACHE_LINE_SIZE     0x4   // cache line size in bytes
#define VX_CAPS_GLOBAL_MEM_SIZE     0x5   // global memory size in bytes
#define VX_CAPS_LOCAL_MEM_SIZE      0x6   // local memory size per core in bytes
#define VX_CAPS_ISA_FLAGS           0x7   // device ISA flags
#define VX_CAPS_NUM_MEM_BANKS       0x8   // number of memory banks
#define VX_CAPS_MEM_BANK_SIZE       0x9   // memory bank size in bytes
#define VX_CAPS_NUM_CLUSTERS        0xA   // number of clusters
#define VX_CAPS_SOCKET_SIZE         0xB   // number of cores per socket
#define VX_CAPS_ISSUE_WIDTH         0xC   // issue width per core
#define VX_CAPS_CLOCK_RATE          0xD   // pipeline clock rate in MHz
#define VX_CAPS_PEAK_MEM_BW         0xE   // peak memory bandwidth in megabytes per second

// device isa flags
#define VX_ISA_STD_A                (1ull << ISA_STD_A)
#define VX_ISA_STD_C                (1ull << ISA_STD_C)
#define VX_ISA_STD_D                (1ull << ISA_STD_D)
#define VX_ISA_STD_E                (1ull << ISA_STD_E)
#define VX_ISA_STD_F                (1ull << ISA_STD_F)
#define VX_ISA_STD_H                (1ull << ISA_STD_H)
#define VX_ISA_STD_I                (1ull << ISA_STD_I)
#define VX_ISA_STD_N                (1ull << ISA_STD_N)
#define VX_ISA_STD_Q                (1ull << ISA_STD_Q)
#define VX_ISA_STD_S                (1ull << ISA_STD_S)
#define VX_ISA_STD_V                (1ull << ISA_STD_V)
#define VX_ISA_ARCH(flags)          (1ull << (((flags >> 30) & 0x3) + 4))
#define VX_ISA_EXT_ICACHE           (1ull << (32+ISA_EXT_ICACHE))
#define VX_ISA_EXT_DCACHE           (1ull << (32+ISA_EXT_DCACHE))
#define VX_ISA_EXT_L2CACHE          (1ull << (32+ISA_EXT_L2CACHE))
#define VX_ISA_EXT_L3CACHE          (1ull << (32+ISA_EXT_L3CACHE))
#define VX_ISA_EXT_LMEM             (1ull << (32+ISA_EXT_LMEM))
#define VX_ISA_EXT_ZICOND           (1ull << (32+ISA_EXT_ZICOND))
#define VX_ISA_EXT_TEX              (1ull << (32+ISA_EXT_TEX))
#define VX_ISA_EXT_RASTER           (1ull << (32+ISA_EXT_RASTER))
#define VX_ISA_EXT_OM               (1ull << (32+ISA_EXT_OM))
#define VX_ISA_EXT_TCU              (1ull << (32+ISA_EXT_TCU))
#define VX_ISA_EXT_DXA              (1ull << (32+ISA_EXT_DXA))

// ready wait timeout
#define VX_MAX_TIMEOUT              (24*60*60*1000)   // 24 Hr

// device memory access
#define VX_MEM_READ                 0x1
#define VX_MEM_WRITE                0x2
#define VX_MEM_READ_WRITE           0x3
#define VX_MEM_PIN_MEMORY           0x4

// open the device and connect to it
int vx_dev_open(vx_device_h* hdevice);

// Close the device when all the operations are done
int vx_dev_close(vx_device_h hdevice);

// return device configurations
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value);

// allocate device memory and return address
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);

// reserve memory address range
int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer);

// release device memory
int vx_mem_free(vx_buffer_h hbuffer);

// set device memory access rights
int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags);

// return device memory address
int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address);

// get device memory info
int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used);

// Copy bytes from host to device memory
int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size);

// Copy bytes from device memory to host
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size);

// Copy bytes from device memory to device memory
int vx_copy_dev_to_dev(vx_buffer_h hdest_buffer, uint64_t dest_offset, vx_buffer_h hsrc_buffer, uint64_t src_offset, uint64_t size);
// Start device execution
int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments);

// Start device execution with grid
int vx_start_g(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments,
               uint32_t ndim, const uint32_t* grid_dim, const uint32_t* block_dim, uint32_t lmem_size);

// Wait for device ready with milliseconds timeout
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

// write device configuration registers
int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value);

// read device configuration registers
int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t tag, uint32_t* value);

// query device performance counter
int vx_mpm_query(vx_device_h hdevice, uint32_t mpm_class, uint32_t addr, uint32_t core_id, uint64_t* value);

////////////////////////////// UTILITY FUNCTIONS //////////////////////////////

// upload bytes to device
int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer);

// upload file to device
int vx_upload_kernel_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer);

// upload bytes to device
int vx_upload_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer);

// upload file to device
int vx_upload_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer);

// calculate cooperative threads array occupancy
int vx_check_occupancy(vx_device_h hdevice, uint32_t block_size, uint32_t* max_localmem);

// Return optimal grid/block dimensions for maximum occupancy given global work size
int vx_max_occupancy_grid(vx_device_h hdevice, uint32_t ndim, const uint32_t* global_dim,
                           uint32_t* grid_dim, uint32_t* block_dim);

// performance counters
int vx_dump_perf(vx_device_h hdevice, FILE* stream);

#ifdef __cplusplus
}
#endif

#endif // __VX_VORTEX_H__
