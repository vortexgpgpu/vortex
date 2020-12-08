#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <chrono>

#include <vortex.h>
#include <VX_config.h>
#include <ram.h>
#include <simulator.h>

#define CACHE_LINESIZE  64
#define ALLOC_BASE_ADDR 0x10000000
#define LOCAL_MEM_SIZE  0xffffffff

///////////////////////////////////////////////////////////////////////////////

inline size_t align_size(size_t size, size_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

///////////////////////////////////////////////////////////////////////////////

class vx_device;

class vx_buffer {
public:
    vx_buffer(size_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        auto aligned_asize = align_size(size, CACHE_LINESIZE);
        data_ = malloc(aligned_asize);
    }

    ~vx_buffer() {
        if (data_) {
            free(data_);
        }
    }

    void* data() const {
        return data_;
    }

    size_t size() const {
        return size_;
    }

    vx_device* device() const {
        return device_;
    }

private:
    size_t size_;
    vx_device* device_;
    void* data_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {    
public:
    vx_device() {        
        mem_allocation_ = ALLOC_BASE_ADDR;        
    } 

    ~vx_device() {    
        if (future_.valid()) {
            future_.wait();
        }
    }

    int alloc_local_mem(size_t size, size_t* dev_maddr) {
        auto dev_mem_size = LOCAL_MEM_SIZE;
        size_t asize = align_size(size, CACHE_LINESIZE);        
        if (mem_allocation_ + asize > dev_mem_size)
            return -1;
        *dev_maddr = mem_allocation_;
        mem_allocation_ += asize;
        return 0;
    }

    int upload(void* src, size_t dest_addr, size_t size, size_t src_offset) {
        size_t asize = align_size(size, CACHE_LINESIZE);
        if (dest_addr + asize > ram_.size())
            return -1;

        /*printf("VXDRV: upload %d bytes to 0x%x\n", size, dest_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-write: 0x%x <- 0x%x\n", uint32_t(dest_addr + i), *(uint32_t*)((uint8_t*)src + src_offset + i));
        }*/
        
        ram_.write(dest_addr, asize, (uint8_t*)src + src_offset);
        return 0;
    }

    int download(const void* dest, size_t src_addr, size_t size, size_t dest_offset) {
        size_t asize = align_size(size, CACHE_LINESIZE);
        if (src_addr + asize > ram_.size())
            return -1;

        ram_.read(src_addr, asize, (uint8_t*)dest + dest_offset);
        
        /*printf("VXDRV: download %d bytes from 0x%x\n", size, src_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", uint32_t(src_addr + i), *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int start() {   
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }
        simulator_.attach_ram(&ram_);
        future_ = std::async(std::launch::async, [&]{             
            simulator_.reset();        
            while (simulator_.is_busy()) {
                simulator_.step();
            }
        });
        return 0;
    }

    int wait(long long timeout) {
        if (!future_.valid())
            return 0;
        auto timeout_sec = (timeout < 0) ? timeout : (timeout / 1000);
        std::chrono::seconds wait_time(1);
        for (;;) {
            auto status = future_.wait_for(wait_time); // wait for 1 sec and check status
            if (status == std::future_status::ready 
             || 0 == timeout_sec--)
                break;
        }
        return 0;
    }

    int flush_caches(size_t dev_maddr, size_t size) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        simulator_.attach_ram(&ram_);
        simulator_.flush_caches(dev_maddr, size);        
        while (simulator_.snp_req_active()) {
            simulator_.step();
        };
        simulator_.attach_ram(NULL);
        return 0;
    }

    int set_csr(int core_id, int addr, unsigned value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        simulator_.set_csr(core_id, addr, value);        
        while (simulator_.csr_req_active()) {
            simulator_.step();
        };
        return 0;
    }

    int get_csr(int core_id, int addr, unsigned *value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        simulator_.get_csr(core_id, addr, value);        
        while (simulator_.csr_req_active()) {
            simulator_.step();
        };
        return 0;
    }

private:

    size_t mem_allocation_;     
    RAM ram_;
    Simulator simulator_;
    std::future<void> future_;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, unsigned caps_id, unsigned *value) {
   if (nullptr == hdevice)
        return  -1;

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = IMPLEMENTATION_ID;
        break;
    case VX_CAPS_MAX_CORES:
        *value = NUM_CORES;        
        break;
    case VX_CAPS_MAX_WARPS:
        *value = NUM_WARPS;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = NUM_THREADS;
        break;
    case VX_CAPS_CACHE_LINESIZE:
        *value = CACHE_LINESIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = 0xffffffff;
        break;
    case VX_CAPS_ALLOC_BASE_ADDR:
        *value = 0x10000000;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = STARTUP_ADDR;
        break;
    default:
        std::cout << "invalid caps id: " << caps_id << std::endl;
        std::abort();
        return -1;
    }

    return 0;
}

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return  -1;

    *hdevice = new vx_device();

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    
#ifdef DUMP_PERF_STATS
    unsigned num_cores;
    vx_csr_get(hdevice, 0, CSR_NC, &num_cores);
    if (num_cores > 1) {
        uint64_t total_instrs = 0, total_cycles = 0;
        // -------------------------
        #ifdef PERF_ENABLE
            // PERF: cache 
            uint64_t total_r = 0;
            uint64_t total_w = 0;
            uint64_t dram_st = 0;
            uint64_t dram_lat = 0;
            uint64_t dram_rsp = 0;
            uint64_t msrq_st = 0;
            uint64_t total_st = 0;
            uint64_t r_miss = 0;
            uint64_t w_miss = 0;
            uint64_t core_rsp_st = 0;
            uint64_t total_evict = 0;
            // PERF: pipeline stalls
            uint64_t lsu_stall = 0;
            uint64_t fpu_stall = 0;
            uint64_t mul_stall = 0;
            uint64_t csr_stall = 0;
            uint64_t alu_stall = 0;
            uint64_t gpu_stall = 0;
            uint64_t ibuffer_stall = 0;
            uint64_t scoreboard_stall = 0;
            uint64_t icache_stall = 0;
        #endif     
        // -------------------------
        for (unsigned core_id = 0; core_id < num_cores; ++core_id) {           
            uint64_t instrs, cycles;
            vx_get_perf(hdevice, core_id, &instrs, &cycles);
            float IPC = (float)(double(instrs) / double(cycles));
            fprintf(stdout, "PERF: core%d: instrs=%ld, cycles=%ld, IPC=%f\n", core_id, instrs, cycles, IPC);            
            total_instrs += instrs;
            total_cycles = std::max<uint64_t>(total_cycles, cycles);

            #ifdef PERF_ENABLE
                // PERF: cache
                // total_read
                uint64_t total_r_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_TOTAL_R, CSR_TOTAL_R_H, &total_r_per_core);
                fprintf(stdout, "PERF: \t\ttotal_reads_per_core=%ld\n", total_r_per_core);
                total_r += total_r_per_core;
                // total_write
                uint64_t total_w_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_TOTAL_W, CSR_TOTAL_W_H, &total_w_per_core);
                fprintf(stdout, "PERF: \t\ttotal_writes_per_core=%ld\n", total_w_per_core);
                total_w += total_w_per_core;
                // dram_stall
                uint64_t dram_st_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_DRAM_ST, CSR_DRAM_ST_H, &dram_st_per_core);
                fprintf(stdout, "PERF: \t\tdram_stalls_per_core=%ld\n", dram_st_per_core);
                dram_st += dram_st_per_core;
                // dram_latency
                uint64_t dram_lat_per_core, dram_rsp_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_DRAM_LAT, CSR_DRAM_LAT_H, &dram_lat_per_core);
                vx_csr_get_l(hdevice, core_id, CSR_DRAM_RSP, CSR_DRAM_RSP_H, &dram_rsp_per_core);                
                fprintf(stdout, "PERF: \t\tdram_latency_per_core=%ld\n", dram_lat_per_core);
                fprintf(stdout, "PERF: \t\tdram_response_per_core=%ld\n", dram_rsp_per_core);
                dram_lat += dram_lat_per_core;
                dram_rsp += dram_rsp_per_core;
                float dram_lat_per_rsp_per_core = (float)(double(dram_lat_per_core) / double(dram_rsp_per_core));
                fprintf(stdout, "PERF: \t\tdram_latency_per_response_per_core=%f\n", dram_lat_per_rsp_per_core);
                // miss_reserve_queue_stall
                uint64_t msrq_st_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_MSRQ_ST, CSR_MSRQ_ST_H, &msrq_st_per_core);
                fprintf(stdout, "PERF: \t\tmsrq_stalls_per_core=%ld\n", msrq_st_per_core);
                msrq_st += msrq_st_per_core;
                // total_stall
                uint64_t total_st_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_TOTAL_ST, CSR_TOTAL_ST_H, &total_st_per_core);
                fprintf(stdout, "PERF: \t\ttotal_stalls_per_core=%ld\n", total_st_per_core);
                total_st += total_st_per_core;
                // read_miss
                uint64_t r_miss_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_R_MISS, CSR_R_MISS_H, &r_miss_per_core);
                fprintf(stdout, "PERF: \t\tread_misses_per_core=%ld\n", r_miss_per_core);
                r_miss += r_miss_per_core;
                // write_miss
                uint64_t w_miss_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_W_MISS, CSR_W_MISS_H, &w_miss_per_core);
                fprintf(stdout, "PERF: \t\twrite_misses_per_core=%ld\n", w_miss_per_core);
                w_miss += w_miss_per_core;
                // core_rsp_stalls
                uint64_t core_rsp_st_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_CORE_RSP_ST, CSR_CORE_RSP_ST_H, &core_rsp_st_per_core);
                fprintf(stdout, "PERF: \t\tcore_rsp_stalls_per_core=%ld\n", core_rsp_st_per_core);
                core_rsp_st += core_rsp_st_per_core;
                // total_evictions
                uint64_t total_evict_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_TOTAL_EV, CSR_TOTAL_EV_H, &total_evict_per_core);
                fprintf(stdout, "PERF: \t\ttotal_evictions_per_core=%ld\n", total_evict_per_core);
                total_evict += total_evict_per_core;
                // PERF: pipeline stall
                // lsu_stall
                uint64_t lsu_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_LSU_ST, CSR_LSU_ST_H, &lsu_stall_per_core);
                fprintf(stdout, "PERF: \t\tlsu_stall=%ld\n", lsu_stall_per_core);
                lsu_stall += lsu_stall_per_core;
                // fpu_stall
                uint64_t fpu_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_FPU_ST, CSR_FPU_ST_H, &fpu_stall_per_core);
                fprintf(stdout, "PERF: \t\tfpu_stall=%ld\n", fpu_stall_per_core);
                fpu_stall += fpu_stall_per_core;
                // mul_stall
                uint64_t mul_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_MUL_ST, CSR_MUL_ST_H, &mul_stall_per_core);
                fprintf(stdout, "PERF: \t\tmul_stall=%ld\n", mul_stall_per_core);
                mul_stall += mul_stall_per_core;
                // csr_stall
                uint64_t csr_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_CSR_ST, CSR_CSR_ST_H, &csr_stall_per_core);
                fprintf(stdout, "PERF: \t\tcsr_stall=%ld\n", csr_stall_per_core);
                csr_stall += csr_stall_per_core;
                // alu_stall
                uint64_t alu_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_ALU_ST, CSR_ALU_ST_H, &alu_stall_per_core);
                fprintf(stdout, "PERF: \t\talu_stall=%ld\n", alu_stall_per_core);
                alu_stall += alu_stall_per_core;
                // gpu_stall
                uint64_t gpu_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_GPU_ST, CSR_GPU_ST_H, &gpu_stall_per_core);
                fprintf(stdout, "PERF: \t\tgpu_stall=%ld\n", gpu_stall_per_core);
                gpu_stall += gpu_stall_per_core;
                // ibuffer_stall
                uint64_t ibuffer_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_IBUF_ST, CSR_IBUF_ST_H, &ibuffer_stall_per_core);
                fprintf(stdout, "PERF: \t\tibuffer_stall=%ld\n", ibuffer_stall_per_core);
                ibuffer_stall += ibuffer_stall_per_core;
                // scoreboard_stall
                uint64_t scoreboard_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_SCRBRD_ST, CSR_SCRBRD_ST_H, &scoreboard_stall_per_core);
                fprintf(stdout, "PERF: \t\tscoreboard_stall=%ld\n", scoreboard_stall_per_core);
                scoreboard_stall += scoreboard_stall_per_core;
                // icache_stall
                uint64_t icache_stall_per_core;
                vx_csr_get_l(hdevice, core_id, CSR_ICACHE_ST, CSR_ICACHE_ST_H, &icache_stall_per_core);
                fprintf(stdout, "PERF: \t\ticache_stall=%ld\n", icache_stall_per_core);
                icache_stall += icache_stall_per_core;
            #endif
            // -------------------------
        }
        float IPC = (float)(double(total_instrs) / double(total_cycles));
        fprintf(stdout, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", total_instrs, total_cycles, IPC);    
        
        #ifdef PERF_ENABLE
            // PERF: cache 
            fprintf(stdout, "PERF: \t\ttotal_reads=%ld\n", total_r);
            fprintf(stdout, "PERF: \t\ttotal_writes=%ld\n", total_w);
            fprintf(stdout, "PERF: \t\tdram_stalls=%ld\n", dram_st);
            fprintf(stdout, "PERF: \t\tdram_latency=%ld\n", dram_lat);
            fprintf(stdout, "PERF: \t\tdram_response=%ld\n", dram_rsp);
            float dram_lat_per_rsp = (float)(double(dram_lat) / double(dram_rsp));
            fprintf(stdout, "PERF: \t\tdram_latency_per_response=%f\n", dram_lat_per_rsp);
            fprintf(stdout, "PERF: \t\tmsrq_stalls=%ld\n", msrq_st);
            fprintf(stdout, "PERF: \t\ttotal_stalls=%ld\n", total_st);
            fprintf(stdout, "PERF: \t\tread_misses=%ld\n", r_miss);
            fprintf(stdout, "PERF: \t\twrite_misses=%ld\n", w_miss);
            fprintf(stdout, "PERF: \t\tcore_rsp_stalls=%ld\n", core_rsp_st);
            fprintf(stdout, "PERF: \t\ttotal_evictions=%ld\n", total_evict);
            // PERF: pipeline stall
            fprintf(stdout, "PERF: \t\tlsu_stall=%ld\n", lsu_stall);
            fprintf(stdout, "PERF: \t\tfpu_stall=%ld\n", fpu_stall);
            fprintf(stdout, "PERF: \t\tmul_stall=%ld\n", mul_stall);
            fprintf(stdout, "PERF: \t\tcsr_stall=%ld\n", csr_stall);
            fprintf(stdout, "PERF: \t\talu_stall=%ld\n", alu_stall);
            fprintf(stdout, "PERF: \t\tgpu_stall=%ld\n", gpu_stall);
            fprintf(stdout, "PERF: \t\tibuffer_stall=%ld\n", ibuffer_stall);
            fprintf(stdout, "PERF: \t\tscoreboard_stall=%ld\n", scoreboard_stall);
            fprintf(stdout, "PERF: \t\ticache_stall=%ld\n", icache_stall);
        #endif
        // -------------------------
    } else {
        uint64_t instrs, cycles;
        vx_get_perf(hdevice, 0, &instrs, &cycles);
        float IPC = (float)(double(instrs) / double(cycles));
        fprintf(stdout, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", instrs, cycles, IPC);        
        
        #ifdef PERF_ENABLE
            // PERF: cache 
            // total_read
            uint64_t total_r;
            vx_csr_get_l(hdevice, 0, CSR_TOTAL_R, CSR_TOTAL_R_H, &total_r);
            fprintf(stdout, "PERF: \t\ttotal_reads=%ld\n", total_r);
            // total_write
            uint64_t total_w;
            vx_csr_get_l(hdevice, 0, CSR_TOTAL_W, CSR_TOTAL_W_H, &total_w);
            fprintf(stdout, "PERF: \t\ttotal_writes=%ld\n", total_w);
            // dram_stall
            uint64_t dram_st;
            vx_csr_get_l(hdevice, 0, CSR_DRAM_ST, CSR_DRAM_ST_H, &dram_st);
            fprintf(stdout, "PERF: \t\tdram_stalls=%ld\n", dram_st);
            // dram_latency
            uint64_t dram_lat, dram_rsp;
            vx_csr_get_l(hdevice, 0, CSR_DRAM_LAT, CSR_DRAM_LAT_H, &dram_lat);
            vx_csr_get_l(hdevice, 0, CSR_DRAM_RSP, CSR_DRAM_RSP_H, &dram_rsp);
            float dram_lat_per_rsp = (float)(double(dram_lat) / double(dram_rsp));
            fprintf(stdout, "PERF: \t\tdram_latency=%ld\n", dram_lat);
            fprintf(stdout, "PERF: \t\tdram_response=%ld\n", dram_rsp);
            fprintf(stdout, "PERF: \t\tdram_latency_per_response=%f\n", dram_lat_per_rsp);
            // miss_reserve_queue_stall
            uint64_t msrq_st;
            vx_csr_get_l(hdevice, 0, CSR_MSRQ_ST, CSR_MSRQ_ST_H, &msrq_st);
            fprintf(stdout, "PERF: \t\tmsrq_stalls=%ld\n", msrq_st);
            // total_stall
            uint64_t total_st;
            vx_csr_get_l(hdevice, 0, CSR_TOTAL_ST, CSR_TOTAL_ST_H, &total_st);
            fprintf(stdout, "PERF: \t\ttotal_stalls=%ld\n", total_st);
            // read_miss
            uint64_t r_miss;
            vx_csr_get_l(hdevice, 0, CSR_R_MISS, CSR_R_MISS_H, &r_miss);
            fprintf(stdout, "PERF: \t\tread_misses=%ld\n", r_miss);
            // write_miss
            uint64_t w_miss;
            vx_csr_get_l(hdevice, 0, CSR_W_MISS, CSR_W_MISS_H, &w_miss);
            fprintf(stdout, "PERF: \t\twrite_misses=%ld\n", w_miss);
            // core_rsp_stalls
            uint64_t core_rsp_st;
            vx_csr_get_l(hdevice, 0, CSR_CORE_RSP_ST, CSR_CORE_RSP_ST_H, &core_rsp_st);
            fprintf(stdout, "PERF: \t\ttotal_stalls=%ld\n", core_rsp_st);
            // total_evictions
            uint64_t total_evict;
            vx_csr_get_l(hdevice, 0, CSR_TOTAL_EV, CSR_TOTAL_EV_H, &total_evict);
            fprintf(stdout, "PERF: \t\ttotal_evictions=%ld\n", total_evict);
            // PERF: pipeline stalls
            // TODO:
            // lsu_stall
            uint64_t lsu_stall;
            vx_csr_get_l(hdevice, 0, CSR_LSU_ST, CSR_LSU_ST_H, &lsu_stall);
            fprintf(stdout, "PERF: \t\tlsu_stall=%ld\n", lsu_stall);
            // fpu_stall
            uint64_t fpu_stall;
            vx_csr_get_l(hdevice, 0, CSR_FPU_ST, CSR_FPU_ST_H, &fpu_stall);
            fprintf(stdout, "PERF: \t\tfpu_stall=%ld\n", fpu_stall);
            // mul_stall
            uint64_t mul_stall;
            vx_csr_get_l(hdevice, 0, CSR_MUL_ST, CSR_MUL_ST_H, &mul_stall);
            fprintf(stdout, "PERF: \t\tmul_stall=%ld\n", mul_stall);
            // csr_stall
            uint64_t csr_stall;
            vx_csr_get_l(hdevice, 0, CSR_CSR_ST, CSR_CSR_ST_H, &csr_stall);
            fprintf(stdout, "PERF: \t\tcsr_stall=%ld\n", csr_stall);
            // alu_stall
            uint64_t alu_stall;
            vx_csr_get_l(hdevice, 0, CSR_ALU_ST, CSR_ALU_ST_H, &alu_stall);
            fprintf(stdout, "PERF: \t\talu_stall=%ld\n", alu_stall);
            // gpu_stall
            uint64_t gpu_stall;
            vx_csr_get_l(hdevice, 0, CSR_GPU_ST, CSR_GPU_ST_H, &gpu_stall);
            fprintf(stdout, "PERF: \t\tgpu_stall=%ld\n", gpu_stall);
            // ibuffer_stall
            uint64_t ibuffer_stall;
            vx_csr_get_l(hdevice, 0, CSR_IBUF_ST, CSR_IBUF_ST_H, &ibuffer_stall);
            fprintf(stdout, "PERF: \t\tibuffer_stall=%ld\n", ibuffer_stall);
            // scoreboard_stall
            uint64_t scoreboard_stall;
            vx_csr_get_l(hdevice, 0, CSR_SCRBRD_ST, CSR_SCRBRD_ST_H, &scoreboard_stall);
            fprintf(stdout, "PERF: \t\tscoreboard_stall=%ld\n", scoreboard_stall);
            // icache_stall
            uint64_t icache_stall;
            vx_csr_get_l(hdevice, 0, CSR_ICACHE_ST, CSR_ICACHE_ST_H, &icache_stall);
            fprintf(stdout, "PERF: \t\ticache_stall=%ld\n", icache_stall);
        #endif
        // -------------------------
    }
#endif

    delete device;

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->alloc_local_mem(size, dev_maddr);
}

extern int vx_flush_caches(vx_device_h hdevice, size_t dev_maddr, size_t size) {
    if (nullptr == hdevice 
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->flush_caches(dev_maddr, size);
}


extern int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice 
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    auto buffer = new vx_buffer(size, device);
    if (nullptr == buffer->data()) {
        delete buffer;
        return -1;
    }

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    return buffer->data();
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    delete buffer;

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + src_offset > buffer->size())
        return -1;

    return buffer->device()->upload(buffer->data(), dev_maddr, size, src_offset);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dest_offset) {
     if (nullptr == hbuffer 
      || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + dest_offset > buffer->size())
        return -1;    

    return buffer->device()->download(buffer->data(), dev_maddr, size, dest_offset);
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->start();
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->wait(timeout);
}

extern int vx_csr_set(vx_device_h hdevice, int core_id, int addr, unsigned value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->set_csr(core_id, addr, value);
}

extern int vx_csr_get(vx_device_h hdevice, int core_id, int addr, unsigned* value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->get_csr(core_id, addr, value);
}

extern int vx_csr_get_l(vx_device_h hdevice, int core_id, int addr, int addr_h, uint64_t* value) {
    if (nullptr == hdevice)
        return -1;

    unsigned csr_value;
    vx_csr_get(hdevice, core_id, addr_h, &csr_value);
    *value = csr_value;
    vx_csr_get(hdevice, core_id, addr, &csr_value);
    *value = (*value << 32) | csr_value;
    return 0;
}