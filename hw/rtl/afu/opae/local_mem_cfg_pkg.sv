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

//`include "platform_afu_top_config.vh"

`ifdef PLATFORM_PROVIDES_LOCAL_MEMORY

package local_mem_cfg_pkg;

    parameter LOCAL_MEM_VERSION_NUMBER = 1;

    parameter LOCAL_MEM_ADDR_WIDTH = `PLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH;
    parameter LOCAL_MEM_DATA_WIDTH = `PLATFORM_PARAM_LOCAL_MEMORY_DATA_WIDTH;

    parameter LOCAL_MEM_BURST_CNT_WIDTH = `PLATFORM_PARAM_LOCAL_MEMORY_BURST_CNT_WIDTH;

    // Number of bytes in a data line
    parameter LOCAL_MEM_DATA_N_BYTES = LOCAL_MEM_DATA_WIDTH / 8;


    // Base types
    // --------------------------------------------------------------------

    typedef logic [LOCAL_MEM_ADDR_WIDTH-1:0] t_local_mem_addr;
    typedef logic [LOCAL_MEM_DATA_WIDTH-1:0] t_local_mem_data;

    typedef logic [LOCAL_MEM_BURST_CNT_WIDTH-1:0] t_local_mem_burst_cnt;

    // Byte-level mask of a data line
    typedef logic [LOCAL_MEM_DATA_N_BYTES-1:0] t_local_mem_byte_mask;

endpackage // local_mem_cfg_pkg

`endif // PLATFORM_PROVIDES_LOCAL_MEMORY
