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
// stub/vortex.cpp — libvortex.so dispatcher.
//
// libvortex.so is the user-facing front of the runtime. Its job is to:
//   1. Aggregate the backend-agnostic entry points (common/vx_*.cpp,
//      legacy_runtime.cpp, etc.) into a single dynamic library that
//      every test/program links against.
//   2. At first vx_device_open(), dlopen the backend library named by
//      $VORTEX_DRIVER (default "simx"), resolve its vx_dev_init, and
//      hand the populated callbacks_t back to common/device.cpp via the
//      dispatcher_get_callbacks() accessor.
//
// common/ stays backend-agnostic: it includes "dispatcher.h" and asks
// for a callbacks table; it never touches dlopen, getenv("VORTEX_DRIVER"),
// or the library-name string. That contract lives entirely here.
// ============================================================================

#include "dispatcher.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>

namespace {

// Per-process handle on the dlopened backend library (libvortex-<NAME>.so).
// One backend per process; reused across vx_device_open calls.
void*       g_backend_lib = nullptr;
callbacks_t g_backend_cb  {};

} // anonymous namespace

namespace vx {

vx_result_t dispatcher_get_callbacks(const callbacks_t** out) {
    if (!out) return VX_ERR_INVALID_VALUE;

    if (g_backend_lib != nullptr) {
        *out = &g_backend_cb;
        return VX_SUCCESS;
    }

    const char* drv = std::getenv("VORTEX_DRIVER");
    if (drv == nullptr) drv = "simx";   // default backend
    std::string lib = std::string("libvortex-") + drv + ".so";

    void* h = dlopen(lib.c_str(), RTLD_LAZY);
    if (h == nullptr) {
        std::cerr << "vortex: cannot open backend library '" << lib
                  << "': " << dlerror() << std::endl;
        return VX_ERR_DEVICE_LOST;
    }

    using vx_dev_init_t = int (*)(callbacks_t*);
    auto init = reinterpret_cast<vx_dev_init_t>(dlsym(h, "vx_dev_init"));
    if (init == nullptr) {
        std::cerr << "vortex: backend library '" << lib
                  << "' is missing vx_dev_init: " << dlerror() << std::endl;
        dlclose(h);
        return VX_ERR_DEVICE_LOST;
    }

    if (init(&g_backend_cb) != 0) {
        std::cerr << "vortex: vx_dev_init failed in '" << lib << "'"
                  << std::endl;
        dlclose(h);
        return VX_ERR_DEVICE_LOST;
    }

    g_backend_lib = h;
    *out = &g_backend_cb;
    return VX_SUCCESS;
}

} // namespace vx
