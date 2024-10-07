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

#ifndef __FPGA_H__
#define __FPGA_H__

#include <stdint.h>
#include <uuid/uuid.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copyright (C) 2019, Xilinx Inc - All rights reserved.
 * Xilinx Runtime (XRT) APIs
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/**
 * XCL BO Flags bits layout
 * bits  0 ~ 15: DDR BANK index
 * bits 24 ~ 31: BO flags
 */
#define XRT_BO_FLAGS_MEMIDX_MASK	(0xFFFFFFUL)
#define	XCL_BO_FLAGS_NONE		    (0)
#define	XCL_BO_FLAGS_CACHEABLE		(1U << 24)
#define	XCL_BO_FLAGS_KERNBUF		(1U << 25)
#define	XCL_BO_FLAGS_SGL		    (1U << 26)
#define	XCL_BO_FLAGS_SVM		    (1U << 27)
#define	XCL_BO_FLAGS_DEV_ONLY		(1U << 28)
#define	XCL_BO_FLAGS_HOST_ONLY		(1U << 29)
#define	XCL_BO_FLAGS_P2P		    (1U << 30)
#define	XCL_BO_FLAGS_EXECBUF		(1U << 31)

#define XRT_BO_FLAGS_NONE      XCL_BO_FLAGS_NONE
#define XRT_BO_FLAGS_CACHEABLE XCL_BO_FLAGS_CACHEABLE
#define XRT_BO_FLAGS_DEV_ONLY  XCL_BO_FLAGS_DEV_ONLY
#define XRT_BO_FLAGS_HOST_ONLY XCL_BO_FLAGS_HOST_ONLY
#define XRT_BO_FLAGS_P2P       XCL_BO_FLAGS_P2P
#define XRT_BO_FLAGS_SVM       XCL_BO_FLAGS_SVM

enum xclBOSyncDirection {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE,
};

typedef void *xrtDeviceHandle;

typedef void *xrtKernelHandle;

typedef void* xrtXclbinHandle;

typedef void *xrtBufferHandle;

typedef uint64_t xrtErrorCode;

typedef uint64_t xrtBufferFlags;

typedef uint32_t xrtMemoryGroup;

typedef uuid_t xuid_t;

xrtDeviceHandle xrtDeviceOpen(unsigned int index);

int xrtXclbinGetXSAName(xrtDeviceHandle dhdl, char* name, int size, int* ret_size);

int xrtDeviceClose(xrtDeviceHandle dhdl);

int xrtKernelClose(xrtKernelHandle kernelHandle);

xrtBufferHandle xrtBOAlloc(xrtDeviceHandle dhdl, size_t size, xrtBufferFlags flags, xrtMemoryGroup grp);

int xrtBOFree(xrtBufferHandle bhdl);

int xrtBOWrite(xrtBufferHandle bhdl, const void* src, size_t size, size_t offset);

int xrtBORead(xrtBufferHandle bhdl, void* dst, size_t size, size_t offset);

int xrtBOSync(xrtBufferHandle bhdl, enum xclBOSyncDirection dir, size_t size, size_t offset);

int xrtKernelWriteRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t data);

int xrtKernelReadRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t* data);

int xrtErrorGetString(xrtDeviceHandle, xrtErrorCode error, char* out, size_t len, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif // __FPGA_H__
