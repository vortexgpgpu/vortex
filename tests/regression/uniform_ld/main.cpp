// uniform_ld — warp-uniform loads must keep each thread's order with its own
// stores, and every lane must receive its own word.

#include <vortex2.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file   = "kernel.vxbin";
uint32_t    num_iters     = 64;
uint32_t    scatter_words = 65536;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:s:k:h")) != -1) {
        switch (c) {
            case 'n': num_iters     = std::atoi(optarg); break;
            case 's': scatter_words = std::atoi(optarg); break;
            case 'k': kernel_file   = optarg;            break;
            default:
                std::cout << "Usage: [-n iterations] [-s scattered words] [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    if (num_iters == 0) num_iters = 1;
    if (scatter_words == 0) scatter_words = 1;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    // Fill every warp of every core.
    uint64_t nt = 0, nw = 0, nc = 0;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc));
    const uint32_t W = static_cast<uint32_t>(nt);
    const uint32_t num_threads = static_cast<uint32_t>(nt * nw * nc);
    const uint32_t num_slots = (num_threads / W) * num_iters;
    std::cout << "uniform_ld: iters=" << num_iters << " threads=" << num_threads
              << " scatter=" << scatter_words << std::endl;

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    const uint32_t raw_words = num_slots * RAW_STRIDE;
    const uint32_t war_words = num_slots * WAR_STRIDE;
    const uint32_t ro_words  = num_slots * RO_LINE;
    vx_buffer_h raw_buf = nullptr, war_buf = nullptr, ro_buf = nullptr;
    vx_buffer_h scatter_buf = nullptr, err_buf = nullptr, sink_buf = nullptr;
    CHECK(vx_buffer_create(dev, raw_words * sizeof(uint32_t), VX_MEM_READ_WRITE, &raw_buf));
    CHECK(vx_buffer_create(dev, war_words * sizeof(uint32_t), VX_MEM_READ_WRITE, &war_buf));
    CHECK(vx_buffer_create(dev, ro_words * sizeof(uint32_t), VX_MEM_READ, &ro_buf));
    CHECK(vx_buffer_create(dev, scatter_words * sizeof(uint32_t), VX_MEM_READ, &scatter_buf));
    CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &err_buf));
    CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &sink_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_iters     = num_iters;
    kernel_arg.num_threads   = num_threads;
    kernel_arg.warp_size     = W;
    kernel_arg.scatter_words = scatter_words;
    CHECK(vx_buffer_address(raw_buf,     &kernel_arg.raw_addr));
    CHECK(vx_buffer_address(war_buf,     &kernel_arg.war_addr));
    CHECK(vx_buffer_address(ro_buf,      &kernel_arg.ro_addr));
    CHECK(vx_buffer_address(scatter_buf, &kernel_arg.scatter_addr));
    CHECK(vx_buffer_address(err_buf,     &kernel_arg.err_addr));
    CHECK(vx_buffer_address(sink_buf,    &kernel_arg.sink_addr));

    std::vector<uint32_t> h_raw(raw_words), h_war(war_words), h_ro(ro_words), h_scatter(scatter_words);
    for (uint32_t i = 0; i < raw_words; ++i) h_raw[i] = i * 7 + 1;
    for (uint32_t i = 0; i < war_words; ++i) h_war[i] = i * 7 + 1;
    for (uint32_t i = 0; i < ro_words; ++i)  h_ro[i]  = i * 3 + 2;
    for (uint32_t i = 0; i < scatter_words; ++i) h_scatter[i] = i;
    CHECK(vx_enqueue_write(q, raw_buf, 0, h_raw.data(), raw_words * sizeof(uint32_t), 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, war_buf, 0, h_war.data(), war_words * sizeof(uint32_t), 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, ro_buf, 0, h_ro.data(), ro_words * sizeof(uint32_t), 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, scatter_buf, 0, h_scatter.data(), scatter_words * sizeof(uint32_t), 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_threads, grid, block));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    std::vector<uint32_t> h_err(num_threads);
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_err.data(), err_buf, 0, num_threads * sizeof(uint32_t), 1, &launch_ev, nullptr));
    CHECK(vx_enqueue_read(q, h_raw.data(), raw_buf, 0, raw_words * sizeof(uint32_t), 1, &launch_ev, nullptr));
    CHECK(vx_enqueue_read(q, h_war.data(), war_buf, 0, war_words * sizeof(uint32_t), 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t t = 0; t < num_threads; ++t) {
        if (h_err[t] == 0) continue;
        if (errors < 16)
            std::printf("*** thread %u: %u wrong loads\n", t, h_err[t]);
        errors += h_err[t];
    }
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        uint32_t ri = slot * RAW_STRIDE, wi = slot * WAR_STRIDE;
        if (h_raw[ri] != UNIFORM_MARK + ri) {
            if (errors < 16) std::printf("*** raw[%u] expected=0x%x actual=0x%x\n", ri, UNIFORM_MARK + ri, h_raw[ri]);
            ++errors;
        }
        if (h_war[wi] != UNIFORM_MARK + wi) {
            if (errors < 16) std::printf("*** war[%u] expected=0x%x actual=0x%x\n", wi, UNIFORM_MARK + wi, h_war[wi]);
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(sink_buf);
    vx_buffer_release(err_buf);
    vx_buffer_release(scatter_buf);
    vx_buffer_release(ro_buf);
    vx_buffer_release(war_buf);
    vx_buffer_release(raw_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
