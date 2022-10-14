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
#include <unordered_map>

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

#define BANK_SIZE   0x10000000
#define NUM_BANKS   16

uint32_t count       = 16;
uint32_t args_addr   = 0x7ffff000;
uint32_t kernel_addr = 0x80000000;
uint32_t src_addr    = 0x20000000;
uint32_t dst_addr    = 0x10000000;

uint32_t kernel_bin [] = {
	0x008000ef, // jal	ra,80000008 <main>
	0x0000000b, // 0xb
	0x7ffff7b7, // lui	a5,0x7ffff
	0x0007a703, // lw	a4,0(a5) # 7ffff000 <reg_t6+0x7fffefe1>
	0x0047a683, // lw	a3,4(a5)
	0x0087a583, // lw	a1,8(a5)
	0xcc5027f3, // csrr	a5,0xcc5
	0x02e787b3, // mul	a5,a5,a4
	0x02070863, // beqz	a4,80000050 <main+0x48>
	0x00f70733, // add	a4,a4,a5
	0x00271713, // slli	a4,a4,0x2
	0x00279793, // slli	a5,a5,0x2
	0x00d787b3, // add	a5,a5,a3
	0x00d70733, // add	a4,a4,a3
	0x40d585b3, // sub	a1,a1,a3
	0x0007a603, // lw	a2,0(a5)
	0x00f586b3, // add	a3,a1,a5
	0x00478793, // addi	a5,a5,4
	0x00c6a023, // sw	a2,0(a3)
	0xfef718e3, // bne	a4,a5,8000003c <main+0x34>
	0x00008067 // ret
};

typedef struct {
  uint32_t count;
  uint32_t src_addr;
  uint32_t dst_addr;  
} kernel_args_t;

static int get_bank_info(uint64_t dev_addr, uint32_t* pIdx, uint32_t* pOff) {
    uint32_t index  = dev_addr / BANK_SIZE;
    uint32_t offset = dev_addr % BANK_SIZE;
    if (index > NUM_BANKS) {
        fprintf(stderr, "[VXDRV] 	0xaddress // out of range: 0x%lx\n", dev_addr);
        return -1;
    }
    *pIdx = index;
    *pOff = offset;
    return 0;
}

class ResourceManager {
public:
    xrt::bo get_buffer(xrt::device& device, uint32_t bank_id) {
        auto it = xrtBuffers_.find(bank_id);
        if (it != xrtBuffers_.end()) {            
            return it->second;
        } else {            
            xrt::bo xrtBuffer(device, BANK_SIZE, xrt::bo::flags::normal, bank_id);
            xrtBuffers_.insert({bank_id, xrtBuffer});
            return xrtBuffer;
        }
    }

private:
    struct buf_cnt_t {
        xrt::bo  xrtBuffer;
        uint32_t count;
    };

    std::unordered_map<uint32_t, xrt::bo> xrtBuffers_;
};

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

    ResourceManager res_mgr;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    std::cout << "Load the kernel ip" << std::endl;
    auto ip = xrt::ip(device, uuid, "vortex_afu");

    std::cout << "Reset device..." << std::endl;
    ip.write_register(MMIO_CTL_ADDR, CTL_AP_RESET);

    // Update DCRs
    ip.write_register(MMIO_DCR_ADDR,     DCR_BASE_STARTUP_ADDR);
    ip.write_register(MMIO_DCR_ADDR + 4, kernel_addr);

    uint32_t kernel_idx;
    uint32_t kernel_offset;
    get_bank_info(kernel_addr, &kernel_idx, &kernel_offset);
    auto kernel_bo = res_mgr.get_buffer(device, kernel_idx);

    uint32_t args_idx;
    uint32_t args_offset;
    get_bank_info(args_addr, &args_idx, &args_offset);
    auto args_bo = res_mgr.get_buffer(device, args_idx);

    uint32_t src_idx;
    uint32_t src_offset;
    get_bank_info(src_addr, &src_idx, &src_offset);
    auto src_bo = res_mgr.get_buffer(device, src_idx);

    uint32_t dst_idx;
    uint32_t dst_offset;
    get_bank_info(dst_addr, &dst_idx, &dst_offset);
    auto dst_bo = res_mgr.get_buffer(device, dst_idx);

    std::cout << "Upload kernel (bank=" << kernel_idx << ", offset=" << kernel_offset << ")..." << std::endl;
    uint32_t kernel_size = sizeof(kernel_bin);
    kernel_bo.write(kernel_bin, kernel_size, kernel_offset);
    kernel_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kernel_size, kernel_offset);

    std::cout << "Upload kernel arguments (bank=" << args_idx << ", offset=" << args_offset << ")..." << std::endl;
    kernel_args_t kernel_args{count, src_addr, dst_addr};
    uint32_t args_size = sizeof(kernel_args);
    args_bo.write(&kernel_args, args_size, args_offset);
    args_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, args_size, args_offset);

    std::cout << "Upload source buffer (bank=" << src_idx << ", offset=" << src_offset << ")..." << std::endl;
    uint32_t src_size = count * sizeof(uint32_t);
    {
        auto src_map = reinterpret_cast<uint32_t*>(src_bo.map<uint8_t*>() + src_offset);
        for (uint32_t i = 0; i < count; ++i) 
            src_map[i] = i;
    }
    src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, src_size, src_offset);

    auto dst_size = count * sizeof(uint32_t);
    std::cout << "Clear destination buffer (bank=" << dst_idx << ", offset=" << dst_offset << ")..." << std::endl; 
    {
        auto dst_map = reinterpret_cast<uint32_t*>(dst_bo.map<uint8_t*>() + dst_offset);
        for (uint32_t i = 0; i < count; ++i) 
            dst_map[i] = 0xdeadbeef;
    }
    dst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, dst_size, dst_offset);

    // Start execution

    //wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    std::cout << "IP Start..." << std::endl;
    ip.write_register(MMIO_CTL_ADDR, CTL_AP_START);

    // Wait until the IP is DONE

    uint32_t axi_ctrl = 0;
    while ((axi_ctrl & CTL_AP_DONE) != CTL_AP_DONE) {
        axi_ctrl = ip.read_register(MMIO_CTL_ADDR);
    }

    std::cout << "IP Done!" << std::endl;

    // check output
    uint32_t errors = 0;
    std::cout << "Download destination buffer (bank=" << dst_idx << ", offset=" << dst_offset << ")..." << std::endl;
    dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, dst_size, dst_offset);
    {
        auto dst_map = reinterpret_cast<uint32_t*>(dst_bo.map<uint8_t*>() + dst_offset);        
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t ref = i;
            if (dst_map[i] != ref) {
                std::cout << "Error (" << i << "): actual=" << dst_map[i] << ", expected=" << ref << std::endl;
                ++errors;
            }
        }
    }
    
    if (errors != 0) {
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
