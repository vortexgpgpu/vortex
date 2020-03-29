//
// Copyright (c) 2017, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// Neither the name of the Intel Corporation nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <uuid/uuid.h>

#include <opae/fpga.h>

// State from the AFU's JSON file, extracted using OPAE's afu_json_mgr script
#include "afu_json_info.h"

#define CACHELINE_BYTES 64
#define CL(x) ((x) * CACHELINE_BYTES)


//
// Search for an accelerator matching the requested UUID and connect to it.
//
static fpga_handle connect_to_accel(const char *accel_uuid)
{
    fpga_properties filter = NULL;
    fpga_guid guid;
    fpga_token accel_token;
    uint32_t num_matches;
    fpga_handle accel_handle;
    fpga_result r;

    // Don't print verbose messages in ASE by default
    //setenv("ASE_LOG", "0", 0);

    // Set up a filter that will search for an accelerator
    fpgaGetProperties(NULL, &filter);
    fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);

    // Add the desired UUID to the filter
    uuid_parse(accel_uuid, guid);
    fpgaPropertiesSetGUID(filter, guid);

    // Do the search across the available FPGA contexts
    num_matches = 1;
    fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);

    // Not needed anymore
    fpgaDestroyProperties(&filter);

    if (num_matches < 1)
    {
        fprintf(stderr, "Accelerator %s not found!\n", accel_uuid);
        return 0;
    }

    // Open accelerator
    r = fpgaOpen(accel_token, &accel_handle, 0);
    assert(FPGA_OK == r);

    // Done with token
    fpgaDestroyToken(&accel_token);

    return accel_handle;
}


//
// Allocate a buffer in I/O memory, shared with the FPGA.
//
static volatile void* alloc_buffer(fpga_handle accel_handle,
                                   ssize_t size,
                                   uint64_t *wsid,
                                   uint64_t *io_addr)
{
    fpga_result r;
    volatile void* buf;

    r = fpgaPrepareBuffer(accel_handle, size, (void*)&buf, wsid, 0);
    if (FPGA_OK != r) return NULL;

    // Get the physical address of the buffer in the accelerator
    r = fpgaGetIOAddress(accel_handle, *wsid, io_addr);
    assert(FPGA_OK == r);

    return buf;
}


int main(int argc, char *argv[])
{
    fpga_handle accel_handle;
    volatile char *buf;
    volatile char *buf_r;
    uint64_t wsid1;
    uint64_t wsid2;
    uint64_t buf_pa;
    uint64_t ret_buf_pa;
    uint64_t buf_rpa;
    uint64_t ret_buf_rpa;
    fpga_result r;

    // Find and connect to the accelerator
    accel_handle = connect_to_accel(AFU_ACCEL_UUID);

    // Allocate a single page memory buffer for write
    buf = (volatile char*)alloc_buffer(accel_handle, 4 * getpagesize(),
                                       &wsid1, &buf_pa);
    // Allocate a single page memory buffer for read
    buf_r = (volatile char*)alloc_buffer(accel_handle, 4 * getpagesize(),
                                       &wsid2, &buf_rpa);
    assert(NULL != buf);

    //// Set the low byte of the shared buffer to 0.  The FPGA will write
    //// a non-zero value to it.
    //buf[0] = 0;

    // Set the low byte of the shared buffer buf_r to 0.  The FPGA will read
    // the values and write to buf address 
    buf[0] = 5;
    buf_r[0] = 5;

    // Tell the accelerator the address of the buffer using cache line
    // addresses.  The accelerator will respond by writing to the buffer.
    r = fpgaWriteMMIO64(accel_handle, 0, 0, buf_pa / CL(1));
    printf("Write address is %08lx\n", buf_pa);
    printf("Write address div 64 is %08lx\n", buf_pa/ CL(1));
    assert(FPGA_OK == r);

    // Wait for response from FPGA. Check using fpgaReadMMIO
    //r = fpgaReadMMIO64(accel_handle, 0, 0, &ret_buf_pa);
    //printf("Returned write is %08lx\n", ret_buf_pa);
    //assert(FPGA_OK == r);

///////////////////// Added to check fpgaRead
    // Wait for response from FPGA. Check using fpgaReadMMIO
    r = fpgaReadMMIO64(accel_handle, 0, 5 * sizeof(uint64_t), &ret_buf_rpa);
    printf("Returned read at 10 is %08lx\n", ret_buf_rpa);
    assert(FPGA_OK == r);
///////////////////////////////////////////////


    // Tell the accelerator the address of the buffer using cache line
    // addresses.  The accelerator will read from the buffer.
    // Write the address to MMIO 1
    r = fpgaWriteMMIO64(accel_handle, 0, sizeof(uint64_t), buf_rpa / CL(1));
    printf("Read address is %08lx\n", buf_rpa);
    printf("Read address div64 is %08lx\n", buf_rpa / CL(1));
    assert(FPGA_OK == r);

    // Wait for response from FPGA. Check using fpgaReadMMIO
    //r = fpgaReadMMIO64(accel_handle, 0, sizeof(uint64_t), &ret_buf_rpa);
    //printf("Returned write is %08lx\n", ret_buf_rpa);
    //assert(FPGA_OK == r);








    // Update this
    // Spin, waiting for the value in memory to change to something non-zero.
    while (5 == buf[0])
    {
        // A well-behaved program would use _mm_pause(), nanosleep() or
        // equivalent to save power here.
    };

    // Print the string written by the FPGA
    printf("%d\n", buf[0]);

    do {
        //printf("%d\n", buf[0]);
    } while (10 != buf[0]);

    // Done
    fpgaReleaseBuffer(accel_handle, wsid1);
    fpgaReleaseBuffer(accel_handle, wsid2);
    fpgaClose(accel_handle);

    return 0;
}
