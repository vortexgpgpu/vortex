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
// stub/vortex.cpp — build-target anchor for the dispatcher library
// (libvortex.so).
//
// The real entry points live in common/:
//
//   common/vx_*.cpp           — vortex2.h C entry points
//                               (vx_device_open, vx_buffer_create,
//                                vx_queue_create, vx_enqueue_*,
//                                vx_event_*, ...). Internally use
//                                vx::Device / Buffer / Queue / Event,
//                                which dispatch to the loaded backend
//                                via a CallbacksAdapter holding the
//                                backend's callbacks_t (filled at
//                                dlopen + vx_dev_init time by
//                                common/vx_device.cpp).
//
//   common/legacy_runtime.cpp — every legacy vortex.h C entry point
//                               implemented as a pure wrapper over
//                               vortex2.h symbols in the same library.
//                               Never touches callbacks_t directly.
//
//   common/legacy_utils.cpp,  — vx_upload_kernel_*, vx_check_occupancy,
//   common/legacy_perf.cpp      vx_mpm_query, vx_dump_perf. These call
//                               vortex.h primitives which route through
//                               the legacy wrapper above.
//
// This translation unit is intentionally empty of code; the Makefile
// includes it as a source so the build target name (libvortex.so) is
// anchored here.
// ============================================================================
