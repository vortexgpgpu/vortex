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
// Each backend's class must provide methods with these signatures (pure-v2
// after Phase E of cp_pure_v2_callbacks_proposal):
//
//   int init();
//   int get_caps(uint32_t caps_id, uint64_t* value);
//   int mem_info(uint64_t* free, uint64_t* used);
//   int mem_alloc(uint64_t size, int flags, uint64_t* dev_addr);
//   int mem_reserve(uint64_t dev_addr, uint64_t size, int flags);
//   int mem_free(uint64_t dev_addr);
//   int mem_access(uint64_t dev_addr, uint64_t size, int flags);
//   int upload(uint64_t dst, const void* src, uint64_t size);
//   int download(void* dst, uint64_t src, uint64_t size);
//   int copy(uint64_t dst, uint64_t src, uint64_t size);
//   int cp_mmio_write(uint32_t off, uint32_t value);
//   int cp_mmio_read(uint32_t off, uint32_t* value);
//
// All kernel launches and DCR ops flow through the dispatcher's CP
// submission helpers (sw/runtime/common/vx_device.cpp); backends no longer
// expose start/ready_wait/dcr_write/dcr_read. The xrt/opae backends route
// cp_mmio_* to their AFU's CP regfile (host MMIO byte offset 0x1000+);
// simx/rtlsim route to a sim/common/CommandProcessor C++ instance.
// Legacy vortex.h symbols in the dispatcher are pure wrappers over
// vortex2.h symbols — they NEVER touch callbacks_t directly.
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

  // ----- Queries -----
  callbacks->query_caps = [](void* dev_ctx, uint32_t caps_id,
                             uint64_t* out_value) -> int {
    if (nullptr == dev_ctx || nullptr == out_value)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->get_caps(caps_id, out_value);
  };

  callbacks->memory_info = [](void* dev_ctx, uint64_t* out_free,
                              uint64_t* out_used) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->mem_info(out_free, out_used);
  };

  // ----- Memory -----
  callbacks->mem_alloc = [](void* dev_ctx, uint64_t size, uint32_t flags,
                            uint64_t* out_dev_addr) -> int {
    if (nullptr == dev_ctx || nullptr == out_dev_addr || 0 == size)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)
              ->mem_alloc(size, static_cast<int>(flags), out_dev_addr);
  };

  callbacks->mem_reserve = [](void* dev_ctx, uint64_t dev_addr, uint64_t size,
                              uint32_t flags) -> int {
    if (nullptr == dev_ctx || 0 == size)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)
              ->mem_reserve(dev_addr, size, static_cast<int>(flags));
  };

  callbacks->mem_free = [](void* dev_ctx, uint64_t dev_addr) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->mem_free(dev_addr);
  };

  callbacks->mem_access = [](void* dev_ctx, uint64_t dev_addr, uint64_t size,
                             uint32_t flags) -> int {
    if (nullptr == dev_ctx)
      return -1;
    if (0 == size)
      return 0;   // no-op; legacy upload path passes size=0 for empty BSS
    return reinterpret_cast<vx_device*>(dev_ctx)
              ->mem_access(dev_addr, size, static_cast<int>(flags));
  };

  // ----- DMA -----
  callbacks->mem_upload = [](void* dev_ctx, uint64_t dst, const void* src,
                             uint64_t size) -> int {
    if (nullptr == dev_ctx || (nullptr == src && size != 0))
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->upload(dst, src, size);
  };

  callbacks->mem_download = [](void* dev_ctx, void* dst, uint64_t src,
                               uint64_t size) -> int {
    if (nullptr == dev_ctx || (nullptr == dst && size != 0))
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->download(dst, src, size);
  };

  callbacks->mem_copy = [](void* dev_ctx, uint64_t dst, uint64_t src,
                           uint64_t size) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->copy(dst, src, size);
  };

  // ----- CP control plane (sole control path) -----
  callbacks->cp_mmio_write = [](void* dev_ctx, uint32_t off,
                                uint32_t value) -> int {
    if (nullptr == dev_ctx)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)->cp_mmio_write(off, value);
  };

  callbacks->cp_mmio_read = [](void* dev_ctx, uint32_t off,
                               uint32_t* out_value) -> int {
    if (nullptr == dev_ctx || nullptr == out_value)
      return -1;
    return reinterpret_cast<vx_device*>(dev_ctx)
              ->cp_mmio_read(off, out_value);
  };

  return 0;
}
