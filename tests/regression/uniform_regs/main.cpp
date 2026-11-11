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
uint32_t    size        = 64;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n words] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    std::srand(50);

    const uint32_t n = size;
    const uint32_t scale = 3;
    const uint64_t buf_size = n * sizeof(uint32_t);

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    uint64_t nt = 0;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));

    vx_buffer_h src_buf = nullptr, dst_buf = nullptr, res_buf = nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ, &src_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));
    CHECK(vx_buffer_create(dev, RES_COUNT * sizeof(uint32_t), VX_MEM_READ_WRITE, &res_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t arg{};
    arg.num_points = n;
    arg.scale = scale;
    CHECK(vx_buffer_address(src_buf, &arg.src_addr));
    CHECK(vx_buffer_address(dst_buf, &arg.dst_addr));
    CHECK(vx_buffer_address(res_buf, &arg.res_addr));

    std::vector<uint32_t> h_src(n), h_dst(n), h_res(RES_COUNT, 0);
    for (uint32_t i = 0; i < n; ++i) {
        h_src[i] = std::rand() & 0xffff;
    }
    CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), buf_size, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, res_buf, 0, h_res.data(), RES_COUNT * sizeof(uint32_t), 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &n, grid, block));
    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &arg;
    li.args_size   = sizeof(arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    vx_event_h launch_ev = nullptr, ev1 = nullptr, ev2 = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &ev1));
    CHECK(vx_enqueue_read(q, h_res.data(), res_buf, 0, RES_COUNT * sizeof(uint32_t), 1, &ev1, &ev2));
    CHECK(vx_event_wait_value(ev2, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t ref = h_src[i] * scale + n;
        if (h_dst[i] != ref) {
            if (errors < 16)
                std::printf("*** dst[%u] expected=%u actual=%u\n", i, ref, h_dst[i]);
            ++errors;
        }
    }
    const uint32_t expect[RES_COUNT] = {
        0, 1u << 8,          // 1 << 40
        4,                   // (1 << 40) >> 38
        n,                   // (n * scale) / scale
        n,                   // the loop
        0xffffffffu,         // lui 0x80000, upper word
        0xffffffffu,         // addw 0x7fffffff + 1, upper word
        uint32_t(nt),        // NUM_THREADS
        n,                   // per-thread store of a uniform value
    };
    static const char* names[RES_COUNT] = {
        "shl40.lo", "shl40.hi", "srl", "div", "loop", "lui.hi", "addw.hi", "csr", "xstore"};
    for (uint32_t i = 0; i < RES_COUNT; ++i) {
        if (h_res[i] != expect[i]) {
            std::printf("*** %s expected=0x%x actual=0x%x\n", names[i], expect[i], h_res[i]);
            ++errors;
        }
    }

    vx_event_release(ev2);
    vx_event_release(ev1);
    vx_event_release(launch_ev);
    vx_buffer_release(res_buf);
    vx_buffer_release(dst_buf);
    vx_buffer_release(src_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
