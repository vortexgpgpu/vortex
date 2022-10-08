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

#define MMIO_CTL_ADDR   0x00
#define MMIO_DEV_ADDR   0x10
#define MMIO_ISA_ADDR   0x1C
#define MMIO_DCR_ADDR   0x28

#define CTL_AP_START    (1<<0)
#define CTL_AP_DONE     (1<<1)
#define CTL_AP_IDLE     (1<<2)
#define CTL_AP_READY    (1<<3)
#define CTL_AP_RESET    (1<<4)
#define CTL_AP_RESTART  (1<<7)

#define DCR_BASE_STARTUP_ADDR 1

#define DATA_SIZE   16

#define BANK_SIZE   0x10000000
#define NUM_BANKS   16

uint64_t kernel_addr = 0x80000000;

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

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
    auto ip = xrt::ip(device, uuid, "vortex_afu");

    std::cout << "Reset device..." << std::endl;
    ip.write_register(MMIO_CTL_ADDR, CTL_AP_RESET);

    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;

    std::cout << "Allocate resource buffers..." << std::endl;
    xrt::bo buffers[NUM_BANKS];
    for (int i = 0; i < NUM_BANKS; ++i) {
        buffers[i] = xrt::bo(device, BANK_SIZE, xrt::bo::flags::normal, i);
    }

    // Update DCRs
    ip.write_register(MMIO_DCR_ADDR,     DCR_BASE_STARTUP_ADDR);
    ip.write_register(MMIO_DCR_ADDR + 4, kernel_addr);
    
    // Upload kernel  
    
    std::cout << "Upload kernel..." << std::endl;    
    std::vector<uint32_t> kernel_buf(DATA_SIZE);
    kernel_buf[0] = 0x0005000b; // tmc(0);  terminate program
    for (int i = 1; i < DATA_SIZE; ++i) {
        kernel_buf[i] = 0x0;
    }

    int kernel_idx;
    int kernel_offset;
    BufferInfo(kernel_addr, &kernel_idx, &kernel_offset);
    auto buffer = buffers[kernel_idx];

    buffer.write(kernel_buf.data(), vector_size_bytes, kernel_offset);
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, vector_size_bytes, kernel_offset);

    // Start execution

    //wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    std::cout << "IP Start..." << std::endl;
    ip.write_register(MMIO_CTL_ADDR, CTL_AP_START);

    // Wait until the IP is DONE

    uint32_t axi_ctrl = 0;
    while ((axi_ctrl & CTL_AP_IDLE) != CTL_AP_IDLE) {
        axi_ctrl = ip.read_register(MMIO_CTL_ADDR);
    }

    std::cout << "IP Done!" << std::endl;

    return 0;
}
