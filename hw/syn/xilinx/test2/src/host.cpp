/**
* Copyright (C) 2019-2021 Xilinx, Inc
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

#include "cmdlineparser.h"
#include <iostream>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"

#define CSR_CTL     0x0
#define CSR_A       0x10
#define CSR_B       0x1C
#define CSR_C       0x28
#define CSR_L       0x34

#define IP_START    0x1
#define IP_IDLE     0x4

#define DATA_SIZE   16

#define BANK_SIZE   0x10000000
#define NUM_BANKS   16

uint64_t ba_addr = 0x80000000;
uint64_t bb_addr = 0x40000000;
uint64_t bc_addr = 0x10000000;

void BufferInfo(uint64_t addr, int* pIdx, int* pOff) {
    int index  = addr / BANK_SIZE;
    int offset = addr % BANK_SIZE;
    if (index > NUM_BANKS) {
        fprintf(stderr, "[VXDRV] Error: address out of range: 0x%lx\n", addr);
        exit(-1);
    }
    *pIdx = index;
    *pOff = offset;
}

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    std::cout << "Load the kernel ip" << std::endl;
    auto ip = xrt::ip(device, uuid, "krnl_vadd_rtl");

    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;

    std::cout << "Allocate resource buffers..." << std::endl;
    xrt::bo buffers[NUM_BANKS];
    for (int i = 0; i < NUM_BANKS; ++i) {
        buffers[i] = xrt::bo(device, BANK_SIZE, xrt::bo::flags::normal, i);
    }

    int ba_idx, bb_idx, bc_idx;
    int ba_offset, bb_offset, bc_offset;

    BufferInfo(ba_addr, &ba_idx, &ba_offset);
    BufferInfo(bb_addr, &bb_idx, &bb_offset);
    BufferInfo(bc_addr, &bc_idx, &bc_offset);

    auto ba = buffers[ba_idx];
    auto bb = buffers[bb_idx];
    auto bc = buffers[bc_idx];

    // upload source buffers  
    std::cout << "Writing the input data..." << std::endl;  
    
    std::vector<int> src_buf(DATA_SIZE), dst_buf(DATA_SIZE), ref_buf(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; ++i) {
        src_buf[i] = i;
        dst_buf[i] = 0xdeadbeef;
        ref_buf[i] = i + i;
    }

    ba.write(src_buf.data(), vector_size_bytes, ba_offset);
    ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, vector_size_bytes, ba_offset);

    bb.write(src_buf.data(), vector_size_bytes, bb_offset);
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, vector_size_bytes, bb_offset);

    std::cout << "Setting IP registers..." << std::endl;
    ip.write_register(CSR_A, ba_addr);
    ip.write_register(CSR_A + 4, ba_addr >> 32);
    ip.write_register(CSR_B, bb_addr);
    ip.write_register(CSR_B + 4, bb_addr >> 32);
    ip.write_register(CSR_C, bc_addr);
    ip.write_register(CSR_C + 4, bc_addr >> 32);
    ip.write_register(CSR_L, DATA_SIZE);

    // Start execution

    std::cout << "IP Start..." << std::endl;
    ip.write_register(CSR_CTL, IP_START);

    // Wait until the IP is DONE

    uint32_t axi_ctrl = 0;
    while ((axi_ctrl & IP_IDLE) != IP_IDLE) {
        axi_ctrl = ip.read_register(CSR_CTL);
    }

    std::cout << "IP Done!" << std::endl;

    // Get the output

    std::cout << "Reading output data..." << std::endl;

    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE, vector_size_bytes, bc_offset);
    bc.read(dst_buf.data(), vector_size_bytes, bc_offset);

    // Validate our results
    std::cout << "Validating results..." << std::endl;
    int errors = 0;
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (dst_buf[i] != ref_buf[i]) {
            std::cout << "*** missmatch: (" << i << ") actual=" << dst_buf[i] << ", expected=" << ref_buf[i] << std::endl;
            ++errors;
        }
    }

    if (errors != 0) {
        std::cout << "TEST FAILED!\n";
        return errors;
    }

    std::cout << "TEST PASSED!\n";
        
    return 0;
}
