#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <assert.h>
#include <VX_config.h>
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


template<int N> 
constexpr bool static_print() {
    return (0 < N < 100); 
}

#define MMIO_SCOPE_READ     (AFU_IMAGE_MMIO_SCOPE_READ * 4)
#define MMIO_SCOPE_WRITE    (AFU_IMAGE_MMIO_SCOPE_WRITE * 4)

struct scope_signal_t {
    int width;
    const char* name;
};

constexpr int ilog2(int n) {
    return (n > 1) ? 1 +
     ilog2(n >> 1) : 0;
}

static constexpr int NW_BITS = ilog2(NUM_WARPS);

#ifdef EXT_F_ENABLE
static constexpr int NR_BITS = ilog2(64);
#else
static constexpr int NR_BITS = ilog2(32);
#endif

static constexpr int EX_BITS  = 3;
static constexpr int OP_BITS  = 4;
static constexpr int MOD_BITS = 3;

static constexpr int ICORE_TAG_WIDTH = NW_BITS;
static constexpr int DCORE_TAG_WIDTH = ilog2(LSUQ_SIZE);

static constexpr scope_signal_t scope_signals[] = {

    { 32, "dram_req_addr" },
    { 1,  "dram_req_rw" },
    { 16, "dram_req_byteen" },
    { 128, "dram_req_data" },
    { 29, "dram_req_tag" },
    { 128, "dram_rsp_data" },
    { 29, "dram_rsp_tag" }, 

    { 32, "snp_req_addr" },
    { 1,  "snp_req_invalidate" },
    { 16, "snp_req_tag" },
    { 16, "snp_rsp_tag" },    
    
    { NW_BITS, "icache_req_wid" },
    { 32, "icache_req_addr" },    
    { ICORE_TAG_WIDTH, "icache_req_tag" },  
    { 32, "icache_rsp_data" },    
    { ICORE_TAG_WIDTH, "icache_rsp_tag" },

    { NW_BITS, "dcache_req_wid" },         
    { 32, "dcache_req_pc" },
    { NUM_THREADS * 32, "dcache_req_addr" },
    { 1,  "dcache_req_rw" },
    { NUM_THREADS * 4,  "dcache_req_byteen" },
    { NUM_THREADS * 32, "dcache_req_data" },
    { DCORE_TAG_WIDTH, "dcache_req_tag" },
    { NUM_THREADS * 32, "dcache_rsp_data" },    
    { DCORE_TAG_WIDTH, "dcache_rsp_tag" }, 

    { NW_BITS, "issue_wid" },
    { NUM_THREADS, "issue_tmask" },
    { 32, "issue_pc" },
    { EX_BITS, "issue_ex_type" },
    { OP_BITS, "issue_op_type" },
    { MOD_BITS, "issue_op_mod" },
    { 1, "issue_wb" },
    { NR_BITS, "issue_rd" },
    { NR_BITS, "issue_rs1" },
    { NR_BITS, "issue_rs2" },
    { NR_BITS, "issue_rs3" },
    { 32, "issue_imm" },
    { 1, "issue_rs1_is_pc" },
    { 1, "issue_rs2_is_imm" },
    
    { NW_BITS, "gpr_rsp_wid" },
    { 32, "gpr_rsp_pc" },
    { NUM_THREADS * 32, "gpr_rsp_a" },
    { NUM_THREADS * 32, "gpr_rsp_b" },    
    { NUM_THREADS * 32, "gpr_rsp_c" },    
    
    { NW_BITS, "writeback_wid" },    
    { 32, "writeback_pc" },
    { NR_BITS,  "writeback_rd" },
    { NUM_THREADS * 32, "writeback_data" },    

    { 32, "bank_addr_st0" },    
    { 32, "bank_addr_st1" },    
    { 32, "bank_addr_st2" },      
    { 1,  "scope_bank_is_mrvq_st1" },
    { 1,  "scope_bank_miss_st1" },
    { 1,  "scope_bank_dirty_st1" },
    { 1,  "scope_bank_force_miss_st1" },

    ///////////////////////////////////////////////////////////////////////////
    
    { 1, "dram_req_valid" },   
    { 1, "dram_req_ready" },
    { 1, "dram_rsp_valid" },
    { 1, "dram_rsp_ready" },
    
    { 1, "snp_req_valid" },   
    { 1, "snp_req_ready" },
    { 1, "snp_rsp_valid" },
    { 1, "snp_rsp_ready" },

    { 1, "icache_req_valid" },
    { 1, "icache_req_ready" },
    { 1, "icache_rsp_valid" },
    { 1, "icache_rsp_ready" },

    { NUM_THREADS, "dcache_req_valid" },  
    { 1, "dcache_req_ready" }, 
    { NUM_THREADS, "dcache_rsp_valid" }, 
    { 1, "dcache_rsp_ready" },
    
    { 1, "bank_valid_st0" },
    { 1, "bank_valid_st1" },
    { 1, "bank_valid_st2" },
    { 1, "bank_stall_pipe" },

    { 1, "issue_valid" },   
    { 1, "issue_ready" },   
    { 1, "gpr_rsp_valid" },
    { 1, "writeback_valid" }, 
    { 1, "scoreboard_delay" },
    { 1, "gpr_delay" },
    { 1, "execute_delay" },
    { 1, "busy" },
};

static constexpr int num_signals = sizeof(scope_signals) / sizeof(scope_signal_t);

constexpr int calcFrameWidth(int index = 0) {
    return (index < num_signals) ? (scope_signals[index].width + calcFrameWidth(index + 1)) : 0;
}

static constexpr int fwidth = calcFrameWidth();
static_assert(fwidth == 1766, "invalid size");

int vx_scope_start(fpga_handle hfpga, uint64_t delay) {    
    if (nullptr == hfpga)
        return -1;  
    
    if (delay != uint64_t(-1)) {
        // set start delay
        uint64_t cmd_delay = ((delay << 3) | 4);
        CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, cmd_delay));    
        std::cout << "scope start delay: " << delay << std::endl;
    }

    return 0;
}

int vx_scope_stop(fpga_handle hfpga, uint64_t delay) {    
    if (nullptr == hfpga)
        return -1;
    
    if (delay != uint64_t(-1)) {
        // stop recording
        uint64_t cmd_stop = ((delay << 3) | 5);
        CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, cmd_stop));
        std::cout << "scope stop delay: " << delay << std::endl;
    }

    std::ofstream ofs("vx_scope.vcd");

    ofs << "$timescale 1 ns $end" << std::endl;
    ofs << "$var reg 1 0 clk $end" << std::endl;

    ofs << "enddefinitions $end" << std::endl;

    uint64_t frame_width, max_frames, data_valid;    

    // wait for recording to terminate
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 0));
    do {        
        CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &data_valid));        
        if (data_valid)
            break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } while (true);

    std::cout << "scope trace dump begin..." << std::endl;    

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 2));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &frame_width));
    std::cout << "scope::frame_width=" << std::dec << frame_width << std::endl;

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 3));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &max_frames));
    std::cout << "scope::max_frames=" << std::dec << max_frames << std::endl;    

    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 1));

    if (fwidth != (int)frame_width) {   
        std::cerr << "invalid frame_width: expecting " << std::dec << fwidth << "!" << std::endl;
        std::abort();
    }
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
        fpga_result res = fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &delta);
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
            CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 0));
            CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &data_valid));  
            assert(data_valid == 1);
            CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 1));
        }

        uint64_t word;
        CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &word));
        
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

    std::cout << "scope trace dump done! - " << (timestamp/2) << " cycles" << std::endl;

    // verify data not valid
    CHECK_RES(fpgaWriteMMIO64(hfpga, 0, MMIO_SCOPE_WRITE, 0));
    CHECK_RES(fpgaReadMMIO64(hfpga, 0, MMIO_SCOPE_READ, &data_valid));  
    assert(data_valid == 0);

    return 0;
}