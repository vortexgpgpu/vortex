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

// ============================================================================
// callbacks.inc — generic vx_dev_init template, included once at the bottom
// of each backend's vortex.cpp (after the vx_device class is declared).
//
// Each backend's class must provide methods with these signatures:
//
//   int init();
//   int cp_reg_write(uint32_t off, uint32_t value);
//   int cp_reg_read (uint32_t off, uint32_t* value);
//   int host_mem_alloc(uint64_t size, void** host_ptr, uint64_t* cp_addr);
//   int host_mem_free (uint64_t cp_addr);
//
// The backend is a pure transport HAL — device memory allocation, DMA and
// capability decoding all live in the common core (sw/runtime/common). The
// CP is the sole command + DMA engine; backends only expose the register
// channel and CP-visible host memory above.
// ============================================================================

extern "C" int vx_dev_init(callbacks_t* callbacks) {
  if (nullptr == callbacks)
    return -1;

  // ----- Device lifecycle -----
  callbacks->dev_open = [](void** out_dev_ctx) -> int {
    if (nullptr == out_dev_ctx)
      return -1;
    auto device = new vx_device();
    if (device == nullptr)
      return -1;
    CHECK_ERR(device->init(), {
      delete device;
      return err;
    });
    DBGPRINT("DEV_OPEN: ctx=%p\n", (void*)device);
    *out_dev_ctx = device;
    return 0;
  };

  callbacks->dev_close = [](void* dev_ctx) -> int {
    if (nullptr == dev_ctx)
      return -1;
    DBGPRINT("DEV_CLOSE: ctx=%p\n", dev_ctx);
    delete reinterpret_cast<vx_device*>(dev_ctx);
    return 0;
  };

  // ----- CP register channel -----
  callbacks->cp_reg_write = [](void* dev_ctx, uint32_t off,
                               uint32_t value) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->cp_reg_write(off, value);
  };

  callbacks->cp_reg_read = [](void* dev_ctx, uint32_t off,
                              uint32_t* out_value) -> int {
    if (nullptr == dev_ctx || nullptr == out_value)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->cp_reg_read(off, out_value);
  };

  // ----- CP-visible host memory -----
  callbacks->host_mem_alloc = [](void* dev_ctx, uint64_t size,
                                 void** out_host_ptr,
                                 uint64_t* out_cp_addr) -> int {
    if (nullptr == dev_ctx || nullptr == out_host_ptr ||
        nullptr == out_cp_addr || 0 == size)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)
              ->host_mem_alloc(size, out_host_ptr, out_cp_addr);
  };

  callbacks->host_mem_free = [](void* dev_ctx, uint64_t cp_addr) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->host_mem_free(cp_addr);
  };

  return 0;
}
