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

#ifndef __VRT_C_H__
#define __VRT_C_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// C surface mirroring the subset of the V80 runtime the AVED backend uses:
// open a device against a device binary, resolve a kernel by name, and read
// and write its registers. There is no buffer API because the Command
// Processor owns device memory; the backend never allocates through it.
//
// The device-binary and BDF arguments are accepted and ignored: the model is
// linked in-process, so there is nothing to program or address. They stay in
// the signature so the backend shares one call site with the hardware path.

typedef void* vrtDeviceHandle;
typedef void* vrtKernelHandle;

vrtDeviceHandle vrtDeviceOpen(const char* bdf, const char* vbin_path);

int vrtDeviceClose(vrtDeviceHandle dhdl);

vrtKernelHandle vrtKernelOpen(vrtDeviceHandle dhdl, const char* name);

int vrtKernelClose(vrtKernelHandle khdl);

int vrtKernelWriteRegister(vrtKernelHandle khdl, uint32_t offset, uint32_t data);

int vrtKernelReadRegister(vrtKernelHandle khdl, uint32_t offset, uint32_t* data);

#ifdef __cplusplus
}
#endif

#endif // __VRT_C_H__
