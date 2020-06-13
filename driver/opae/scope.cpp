#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <assert.h>
#include "scope.h"
#include "vortex_afu.h"

#define CHECK_RES(_expr)                            \
   do {                                             \
     fpga_result res = _expr;                       \
     if (res == FPGA_OK)                            \
       break;                                       \
     printf("OPAE Error: '%s' returned %d, %s!\n",  \
            #_expr, (int)res, fpgaErrStr(res));     \
     return -1;                                     \
   } while (false)

#define MMIO_CSR_SCOPE_CMD      (AFU_IMAGE_MMIO_CSR_SCOPE_CMD * 4)
#define MMIO_CSR_SCOPE_DATA     (AFU_IMAGE_MMIO_CSR_SCOPE_DATA * 4)

struct scope_signal_t {
    int width;
    const char* name;
};

static const scope_signal_t scope_signals[] = {
    { 32, "icache_req_addr" },
    { 2,  "icache_req_warp_num" },
    { 2,  "icache_req_tag" },    
    { 32, "icache_rsp_data" },    
    { 2,  "icache_rsp_tag" },
    { 32, "dcache_req_addr" },
    { 2,  "dcache_req_warp_num" },         
    { 2,  "dcache_req_tag" },
    { 32, "dcache_rsp_data" },    
    { 2 , "dcache_rsp_tag" },     
    { 32, "dram_req_addr" },
    { 29, "dram_req_tag" },
    { 29, "dram_rsp_tag" }, 
    { 32, "snp_req_addr" },
    { 1,  "snp_req_invalidate" },
    { 16, "snp_req_tag" },
    { 16, "snp_rsp_tag" },    
    { 2,  "decode_warp_num" },
    { 32, "decode_curr_PC" },
    { 1,  "decode_is_jal" },
    { 5,  "decode_rs1" },
    { 5,  "decode_rs2" },    
    { 2,  "execute_warp_num" },
    { 5,  "execute_rd" },
    { 32, "execute_a" },
    { 32, "execute_b" },    
    { 2,  "writeback_warp_num" },    
    { 2,  "writeback_wb" },
    { 5,  "writeback_rd" },
    { 32, "writeback_data" },    

    { 1, "icache_req_valid" },
    { 1, "icache_req_ready" },
    { 1, "icache_rsp_valid" },
    { 1, "icache_rsp_ready" },
    { 4, "dcache_req_valid" },  
    { 1, "dcache_req_ready" }, 
    { 4, "dcache_rsp_valid" }, 
    { 1, "dcache_rsp_ready" },
    { 1, "dram_req_valid" },   
    { 1, "dram_req_ready" },
    { 1, "dram_rsp_valid" },
    { 1, "dram_rsp_ready" },
    { 1, "snp_req_valid" },   
    { 1, "snp_req_ready" },
    { 1, "snp_rsp_valid" },
    { 1, "snp_rsp_ready" },
    { 4, "decode_valid" },
    { 4, "execute_valid" },
    { 4, "writeback_valid" },    
    { 1, "schedule_delay" },
    { 1, "memory_delay" },
    { 1, "exec_delay" },
    { 1, "gpr_stage_delay" },
};

static const int num_signals = sizeof(scope_signals) / sizeof(scope_signal_t);

int vx_scope_start(fpga_handle hfpga, uint64_t delay) {    
    if (nullptr == hfpga)
        return -1;

    // set start delay
    uint64_t cmd_delay = ((delay << 3) | 4);
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, cmd_delay));    

    return 0;
}

int vx_scope_stop(fpga_handle hfpga, uint64_t delay) {    
    if (nullptr == hfpga)
        return -1;

    // stop recording
    uint64_t cmd_stop = ((delay << 3) | 5);
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, cmd_stop));

    std::ofstream ofs("vx_scope.vcd");

    ofs << "$timescale 1 ns $end" << std::endl;
    ofs << "$var reg 1 0 clk $end" << std::endl;

    int fwidth = 0;
    for (int i = 0; i < num_signals; ++i) {
        ofs << "$var reg " << scope_signals[i].width << " " << (i+1) << " " << scope_signals[i].name << " $end" << std::endl;
        fwidth += scope_signals[i].width;
    }

    ofs << "enddefinitions $end" << std::endl;

    uint64_t frame_width, max_frames, data_valid;    

    // wait for recording to terminate
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 0));
    do {        
        CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));        
        if (data_valid)
            break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } while (true);

    std::cout << "scope trace dump begin..." << std::endl;    

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 2));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &frame_width));
    std::cout << "scope::frame_width=" << frame_width << std::endl;

    assert(fwidth == (int)frame_width);

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 3));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &max_frames));
    std::cout << "scope::max_frames=" << max_frames << std::endl;    

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 1));

    std::vector<char> signal_data(frame_width+1);

    uint64_t frame_offset = 0;
    uint64_t frame_no = 0;
    uint64_t timestamp = 0;    
    int signal_id = 0;
    int signal_offset = 0;

    auto print_header = [&] () {
        ofs << '#' << timestamp++ << std::endl;
        ofs << "b0 0" << std::endl;
        ofs << '#' << timestamp++ << std::endl;
        ofs << "b1 0" << std::endl;
        
        uint64_t delta;
        fpga_result res = fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &delta);
        assert(res == FPGA_OK);

        while (delta != 0) {
            ofs << '#' << timestamp++ << std::endl;
            ofs << "b0 0" << std::endl;
            ofs << '#' << timestamp++ << std::endl;
            ofs << "b1 0" << std::endl;
            --delta;
        }

        signal_id = num_signals;
    };

    print_header();

    do {
        if (frame_no == (max_frames-1)) {
            // verify last frame is valid
            CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 0));
            CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));  
            assert(data_valid == 1);
            CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 1));
        }

        uint64_t word;
        CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &word));
        
        do {          
            int signal_width = scope_signals[signal_id-1].width;
            int word_offset = frame_offset % 64;

            signal_data[signal_width - signal_offset - 1] = ((word >> word_offset) & 0x1) ? '1' : '0';

            ++signal_offset;
            ++frame_offset;

            if (signal_offset == signal_width) {
                signal_data[signal_width] = 0; // string null termination
                ofs << 'b' << signal_data.data() << ' ' << signal_id << std::endl;
                signal_offset = 0;            
                --signal_id;
            }

            if (frame_offset == frame_width) {   
                assert(0 == signal_offset);   
                frame_offset = 0;
                ++frame_no;
                if (frame_no != max_frames) {                
                    print_header();
                }                        
            }
        } while ((frame_offset % 64) != 0);

    } while (frame_no != max_frames);

    std::cout << "scope trace dump done!" << std::endl;

    // verify data not valid
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_CSR_SCOPE_CMD, 0));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));  
    assert(data_valid == 0);

    return 0;
}