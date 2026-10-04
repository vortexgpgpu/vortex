// ldst_order — a thread's load must not observe its own later store to the
// same word, even while the line is still draining its miss chain.

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
const char* kernel_file = "kernel.vxbin";
uint32_t    num_words   = 1024;
uint32_t    delay_sweep = 4;
uint32_t    delay_step  = 1;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:s:d:k:h")) != -1) {
        switch (c) {
            case 'n': num_words   = std::atoi(optarg); break;
            case 's': delay_sweep = std::atoi(optarg); break;
            case 'd': delay_step  = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-n words] [-s delay sweep] [-d delay step] [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

uint32_t init_word(uint32_t i) { return i * 7 + 1; }
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    if (delay_sweep == 0) delay_sweep = 1;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    // Fill every warp of every core.
    uint64_t nt = 0, nw = 0, nc = 0;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc));
    const uint32_t num_threads = static_cast<uint32_t>(nt * nw * nc);
    std::cout << "ldst_order: words=" << num_words << " threads=" << num_threads
              << " delay=" << delay_sweep << "x" << delay_step << std::endl;

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    const uint64_t data_size = uint64_t(num_words) * sizeof(uint32_t);
    const uint64_t sink_size = uint64_t(num_threads) * sizeof(uint32_t);
    vx_buffer_h data_buf = nullptr, old_buf = nullptr, sink_buf = nullptr;
    CHECK(vx_buffer_create(dev, data_size, VX_MEM_READ_WRITE, &data_buf));
    CHECK(vx_buffer_create(dev, data_size, VX_MEM_WRITE,      &old_buf));
    CHECK(vx_buffer_create(dev, sink_size, VX_MEM_WRITE,      &sink_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_words   = num_words;
    kernel_arg.num_threads = num_threads;
    kernel_arg.warp_size   = static_cast<uint32_t>(nt);
    kernel_arg.delay_sweep = delay_sweep;
    kernel_arg.delay_step  = delay_step;
    CHECK(vx_buffer_address(data_buf, &kernel_arg.data_addr));
    CHECK(vx_buffer_address(old_buf,  &kernel_arg.old_addr));
    CHECK(vx_buffer_address(sink_buf, &kernel_arg.sink_addr));

    std::vector<uint32_t> h_data(num_words), h_old(num_words);
    for (uint32_t i = 0; i < num_words; ++i) h_data[i] = init_word(i);
    CHECK(vx_enqueue_write(q, data_buf, 0, h_data.data(), data_size, 0, nullptr, nullptr));

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

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_old.data(), old_buf, 0, data_size, 1, &launch_ev, nullptr));
    CHECK(vx_enqueue_read(q, h_data.data(), data_buf, 0, data_size, 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    // Each warp owns groups of four regions of warp_size words: region 0 is
    // loaded and then overwritten, region 3 is overwritten, regions 1-2 are
    // only read.
    const uint32_t W = static_cast<uint32_t>(nt);
    const uint32_t group_words = 4 * W;
    const uint32_t groups = num_words / group_words;
    if (groups < num_threads / W) {
        std::cout << "words (" << num_words << ") must cover one group per warp ("
                  << (num_threads / W) * group_words << ")" << std::endl;
        return 1;
    }
    int errors = 0;
    auto expect = [&](const char* what, uint32_t i, uint32_t actual, uint32_t ref) {
        if (actual == ref) return;
        if (errors < 16)
            std::printf("*** %s [%u] expected=0x%x actual=0x%x%s\n", what, i, ref, actual,
                        actual == LDST_MARK + i ? " (its own later store)" : "");
        ++errors;
    };
    for (uint32_t g = 0; g < groups; ++g) {
        for (uint32_t l = 0; l < W; ++l) {
            uint32_t i0 = g * group_words + l;
            expect("load r0", i0, h_old[i0], init_word(i0));
            expect("data r0", i0, h_data[i0], LDST_MARK + i0);
            expect("data r1", i0 + W, h_data[i0 + W], init_word(i0 + W));
            expect("data r2", i0 + 2 * W, h_data[i0 + 2 * W], init_word(i0 + 2 * W));
            expect("data r3", i0 + 3 * W, h_data[i0 + 3 * W], LDST_MARK + i0 + 3 * W);
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(sink_buf);
    vx_buffer_release(old_buf);
    vx_buffer_release(data_buf);
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
