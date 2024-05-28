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

#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <vortex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  // open the device and connect to it
  int (*dev_open) (vx_device_h* hdevice);

  // Close the device when all the operations are done
  int (*dev_close) (vx_device_h hdevice);

  // return device configurations
  int (*dev_caps) (vx_device_h hdevice, uint32_t caps_id, uint64_t *value);

  // allocate device memory and return address
  int (*mem_alloc) (vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);

  // reserve memory address range
  int (*mem_reserve) (vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer);

  // release device memory
  int (*mem_free) (vx_buffer_h hbuffer);

  // set device memory access rights
  int (*mem_access) (vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags);

  // return device memory address
  int (*mem_address) (vx_buffer_h hbuffer, uint64_t* address);

  // get device memory info
  int (*mem_info) (vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used);

  // Copy bytes from host to device memory
  int (*copy_to_dev) (vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size);

  // Copy bytes from device memory to host
  int (*copy_from_dev) (void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size);

  // Start device execution
  int (*start) (vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments);

  // Wait for device ready with milliseconds timeout
  int (*ready_wait) (vx_device_h hdevice, uint64_t timeout);

  // read device configuration registers
  int (*dcr_read) (vx_device_h hdevice, uint32_t addr, uint32_t* value);

  // write device configuration registers
  int (*dcr_write) (vx_device_h hdevice, uint32_t addr, uint32_t value);

  // query device performance counter
  int (*mpm_query) (vx_device_h hdevice, uint32_t addr, uint32_t core_id, uint64_t* value);

} callbacks_t;

int vx_dev_init(callbacks_t* callbacks);

#ifdef __cplusplus
}
#endif

#endif