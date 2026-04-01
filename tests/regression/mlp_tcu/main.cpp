#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <tensor_cfg.h>
#include <rvfloats.h>
#include <util.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP   4
#define MAX_ERRORS  100

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = (_expr);                                         \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

using namespace vortex;
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

using fp16_t = uint16_t;

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

static inline fp16_t f32_to_fp16(float f) {
    return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr);
}
static inline float fp16_to_f32(fp16_t h) {
    return bit_cast<float>(rv_htof_s(h, 0, nullptr));
}

// ─────────────────────────────────────────────────────────────────────────────
const char *kernel_file = "kernel.vxbin";

vx_device_h device          = nullptr;
vx_buffer_h input_buffer    = nullptr;
vx_buffer_h output_buffer   = nullptr;
vx_buffer_h buffer1         = nullptr;
vx_buffer_h buffer2         = nullptr;
vx_buffer_h krnl_buffer     = nullptr;
vx_buffer_h args_buffer     = nullptr;
vx_buffer_h layer_cfg_buf   = nullptr;

vx_buffer_h weights_buf[NUM_LAYERS];
vx_buffer_h bias_buf[NUM_LAYERS];

kernel_arg_t kernel_arg = {};

static const uint32_t layer_in_dims [NUM_LAYERS] = { INPUT_DIM,   HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM };
static const uint32_t layer_out_dims[NUM_LAYERS] = { HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, OUTPUT_DIM  };

// ─────────────────────────────────────────────────────────────────────────────
static void show_usage() {
    std::cout << "Vortex MLP-TCU Test." << std::endl;
    std::cout << "Usage: [-k kernel_file] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "k:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        vx_mem_free(input_buffer);
        vx_mem_free(output_buffer);
        vx_mem_free(buffer1);
        vx_mem_free(buffer2);
        vx_mem_free(layer_cfg_buf);
        for (int i = 0; i < NUM_LAYERS; ++i) {
            vx_mem_free(weights_buf[i]);
            vx_mem_free(bias_buf[i]);
        }
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

static void init_weights_xavier(std::vector<float> &w, uint32_t fan_in, uint32_t fan_out) {
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    for (auto &v : w)
        v = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
}

static void mlp_cpu_fp16_batch(
    const std::vector<fp16_t> &input_fp16,
    std::vector<fp16_t>       &output_fp16,
    const std::vector<std::vector<fp16_t>> &fp16_weights,
    const std::vector<std::vector<fp16_t>> &fp16_biases)
{
    std::vector<fp16_t> cur = input_fp16;
    std::vector<fp16_t> nxt;

    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim  = layer_in_dims[l];
        uint32_t out_dim = layer_out_dims[l];
        bool     relu    = (l < NUM_LAYERS - 1);
        nxt.assign(out_dim * BATCH_SIZE, f32_to_fp16(0.0f));

        for (uint32_t i = 0; i < out_dim; ++i) {
            fp16_t bias_h = fp16_biases[l][i];
            for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
                fp16_t sum = bias_h;
                for (uint32_t j = 0; j < in_dim; ++j) {
                    fp16_t w_val = fp16_weights[l][i * in_dim + j];
                    fp16_t x_val = cur[j * BATCH_SIZE + b];
                    sum = muladd_t<vt::fp16, vt::fp16>::eval(w_val, x_val, sum);
                }
                if (relu && fp16_to_f32(sum) < 0.0f) sum = f32_to_fp16(0.0f);
                nxt[i * BATCH_SIZE + b] = sum;
            }
        }
        cur = nxt;
    }

    for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
        float max_val = fp16_to_f32(cur[b]);
        for (uint32_t i = 1; i < OUTPUT_DIM; ++i)
            max_val = std::max(max_val, fp16_to_f32(cur[i * BATCH_SIZE + b]));

        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < OUTPUT_DIM; ++i) {
            float e = std::exp(fp16_to_f32(cur[i * BATCH_SIZE + b]) - max_val);
            e = fp16_to_f32(f32_to_fp16(e));
            cur[i * BATCH_SIZE + b] = f32_to_fp16(e);
            sum_exp += e;
        }
        for (uint32_t i = 0; i < OUTPUT_DIM; ++i)
            cur[i * BATCH_SIZE + b] = f32_to_fp16(fp16_to_f32(cur[i * BATCH_SIZE + b]) / sum_exp);
    }

    output_fp16 = cur;
}

static bool compare_fp16(fp16_t a_h, fp16_t b_h, int index, int errors) {
    auto ordered = [](uint16_t h) -> int32_t {
        int32_t v = (int32_t)h;
        return (v & 0x8000) ? -(v & 0x7FFF) : v;
    };
    int32_t d = std::abs(ordered(a_h) - ordered(b_h));
    if (d > FLOAT_ULP) {
        if (errors < MAX_ERRORS)
            printf("*** error [%d]: expected=%f (0x%04x), actual=%f (0x%04x) diff=%d fp16-ULP\n",
                   index, fp16_to_f32(b_h), (unsigned)b_h,
                   fp16_to_f32(a_h), (unsigned)a_h, d);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    std::srand(42);

    std::cout << "=== Vortex MLP-TCU Test ===" << std::endl;
    std::cout << "Network : " << INPUT_DIM << " -> " << HIDDEN1_DIM
              << " -> " << HIDDEN2_DIM << " -> " << HIDDEN3_DIM
              << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "Batch   : " << BATCH_SIZE << std::endl;
    std::cout << "WMMA tile: M=" << cfg::tileM << " N=" << cfg::tileN
              << " K=" << cfg::tileK << std::endl;
    std::cout << "Input/Output dtype: " << vt::ITYPE::name << std::endl;

    // ── Open device ──────────────────────────────────────────────────────────
    RT_CHECK(vx_dev_open(&device));

    // ── Check TCU support ─────────────────────────────────────────────────────
    uint64_t isa_flags;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if (!(isa_flags & VX_ISA_EXT_TCU)) {
        std::cout << "TCU extension not supported on this device!" << std::endl;
        cleanup();
        return -1;
    }

    uint64_t NT;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
    if (NT != NUM_THREADS) {
        std::cout << "Error: device warp size (" << NT
                  << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
        cleanup();
        return -1;
    }

    // ── Validate TCU tile alignment ───────────────────────────────────────────
    for (int l = 0; l < NUM_LAYERS; ++l) {
        if (layer_out_dims[l] % cfg::tileM || layer_in_dims[l] % cfg::tileK
                || BATCH_SIZE % cfg::tileN) {
            std::cout << "Error: layer " << l << " dims not multiples of TCU tiles ("
                      << cfg::tileM << "/" << cfg::tileN << "/" << cfg::tileK << ")!" << std::endl;
            cleanup();
            return -1;
        }
    }
    std::vector<std::vector<float>>  h_weights_f32(NUM_LAYERS);
    std::vector<std::vector<fp16_t>> h_weights(NUM_LAYERS);
    std::vector<std::vector<fp16_t>> h_biases(NUM_LAYERS);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim  = layer_in_dims[l];
        uint32_t out_dim = layer_out_dims[l];
        h_weights_f32[l].resize(out_dim * in_dim);
        init_weights_xavier(h_weights_f32[l], in_dim, out_dim);

        h_weights[l].resize(out_dim * in_dim);
        for (size_t k = 0; k < h_weights_f32[l].size(); ++k)
            h_weights[l][k] = f32_to_fp16(h_weights_f32[l][k]);

        h_biases[l].assign(out_dim, f32_to_fp16(0.0f));  // zero bias
    }

    std::vector<fp16_t> h_input(INPUT_DIM * BATCH_SIZE);
    for (auto &v : h_input)
        v = f32_to_fp16(static_cast<float>(rand()) / RAND_MAX);

    // ── Allocate device memory ────────────────────────────────────────────────
    std::cout << "Allocating device memory..." << std::endl;

    RT_CHECK(vx_mem_alloc(device, INPUT_DIM  * BATCH_SIZE * sizeof(fp16_t), VX_MEM_READ,       &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer,  &kernel_arg.input_addr));

    RT_CHECK(vx_mem_alloc(device, OUTPUT_DIM * BATCH_SIZE * sizeof(fp16_t), VX_MEM_READ_WRITE, &output_buffer));
    RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));

    uint32_t max_hidden = std::max({HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM});
    RT_CHECK(vx_mem_alloc(device, max_hidden * BATCH_SIZE * sizeof(fp16_t), VX_MEM_READ_WRITE, &buffer1));
    RT_CHECK(vx_mem_address(buffer1, &kernel_arg.buffer1_addr));

    RT_CHECK(vx_mem_alloc(device, max_hidden * BATCH_SIZE * sizeof(fp16_t), VX_MEM_READ_WRITE, &buffer2));
    RT_CHECK(vx_mem_address(buffer2, &kernel_arg.buffer2_addr));

    // ── Upload layer configs, weights, biases ─────────────────────────────────
    std::vector<layer_config_t> layer_cfgs(NUM_LAYERS);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim  = layer_in_dims[l];
        uint32_t out_dim = layer_out_dims[l];

        RT_CHECK(vx_mem_alloc(device, out_dim * in_dim * sizeof(fp16_t), VX_MEM_READ, &weights_buf[l]));
        RT_CHECK(vx_mem_address(weights_buf[l], &layer_cfgs[l].weights_addr));
        RT_CHECK(vx_copy_to_dev(weights_buf[l], h_weights[l].data(), 0, out_dim * in_dim * sizeof(fp16_t)));

        RT_CHECK(vx_mem_alloc(device, out_dim * sizeof(fp16_t), VX_MEM_READ, &bias_buf[l]));
        RT_CHECK(vx_mem_address(bias_buf[l], &layer_cfgs[l].bias_addr));
        RT_CHECK(vx_copy_to_dev(bias_buf[l], h_biases[l].data(), 0, out_dim * sizeof(fp16_t)));

        layer_cfgs[l].input_dim  = in_dim;
        layer_cfgs[l].output_dim = out_dim;
        std::cout << "  Layer " << l << ": " << in_dim << " -> " << out_dim << std::endl;
    }

    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_cfg_buf));
    RT_CHECK(vx_mem_address(layer_cfg_buf, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_cfg_buf, layer_cfgs.data(), 0, NUM_LAYERS * sizeof(layer_config_t)));

    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, INPUT_DIM * BATCH_SIZE * sizeof(fp16_t)));

    kernel_arg.num_layers = NUM_LAYERS;
    kernel_arg.batch_size = BATCH_SIZE;

    // ── Upload kernel ─────────────────────────────────────────────────────────
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // ── Run ───────────────────────────────────────────────────────────────────
    std::cout << "Starting MLP-TCU inference..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Elapsed time: %.3f ms\n",
           std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);

    // ── Download output ───────────────────────────────────────────────────────
    std::vector<fp16_t> h_output(OUTPUT_DIM * BATCH_SIZE);
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, OUTPUT_DIM * BATCH_SIZE * sizeof(fp16_t)));

    // ── CPU reference (fp16 precision) ───────────────────────────────────────
    std::vector<fp16_t> h_ref;
    mlp_cpu_fp16_batch(h_input, h_ref, h_weights, h_biases);

    // ── Verify ────────────────────────────────────────────────────────────────
    int errors = 0;
    for (uint32_t i = 0; i < OUTPUT_DIM * BATCH_SIZE; ++i) {
        if (!compare_fp16(h_output[i], h_ref[i], i, errors))
            ++errors;
    }

    cleanup();

    if (errors != 0) {
        std::cout << "Found " << errors << " / " << (OUTPUT_DIM * BATCH_SIZE) << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
