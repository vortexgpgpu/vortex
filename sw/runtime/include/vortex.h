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

// Legacy synchronous Vortex runtime API — a thin wrapper over vortex2.h.
// The shared handles (vx_device_h, vx_buffer_h), device-capability IDs
// (VX_CAPS_*), ISA flags (VX_ISA_*) and memory-access flags (VX_MEM_*)
// are owned by vortex2.h, the canonical runtime API.
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

// ready wait timeout
#define VX_MAX_TIMEOUT              (24*60*60*1000)   // 24 Hr

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

// Validate a cooperative-thread-array launch: checks block_size against the
// device's per-core thread budget and, when max_localmem != 0, that the
// kernel's per-block local-memory request fits the occupancy budget.
int vx_check_occupancy(vx_device_h hdevice, uint32_t block_size, uint32_t max_localmem);

// Return optimal grid/block dimensions for maximum occupancy given global work size
int vx_max_occupancy_grid(vx_device_h hdevice, uint32_t ndim, const uint32_t* global_dim,
                           uint32_t* grid_dim, uint32_t* block_dim);

// performance counters
int vx_dump_perf(vx_device_h hdevice, FILE* stream);

#ifdef __cplusplus
}
#endif

#endif // __VX_VORTEX_H__
