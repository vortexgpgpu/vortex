#include "common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <sparse_cfg.h>
#include <string.h>
#include <type_traits>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP  6
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = (_expr);                                         \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

using namespace vortex;
namespace vt = sparse;

static uint32_t g_sparsity_degree = 2; // default: 2:4

static size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

using cfg     = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

template <typename T>
struct data_accessor_t {
    using Type = typename T::dtype;
    static Type read(const Type *ptr, size_t offset) { return ptr[offset]; }
};

template <typename S, typename D>
struct muladd_t {
    using stype = typename S::dtype;
    using dtype = typename D::dtype;
    static dtype eval(stype a, stype b, dtype c) {
        return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
    }
};

template <>
struct muladd_t<vt::fp16, vt::fp32> {
    static float eval(uint16_t a, uint16_t b, float c) {
        auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
        auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
        return fa * fb + c;
    }
};

struct SparseMat {
    std::vector<itype_t> values;  
    std::vector<uint8_t> meta;    
    uint32_t rows, cols;          
};

static void matmul_cpu_sparseA(
    otype_t *C, const SparseMat &A, const itype_t *B,
    uint32_t N, uint32_t sparsity_degree)
{
    const uint32_t M = A.rows;
    const uint32_t K = A.cols;
    const uint32_t values_per_row = K * sparsity_degree / 4;
    const uint32_t meta_per_row   = K / 4;

    for (uint32_t m = 0; m < M; ++m) {
        const itype_t *row_vals = A.values.data() + (size_t)m * values_per_row;
        const uint8_t *row_meta = A.meta.data()   + (size_t)m * meta_per_row;
        otype_t       *crow     = C + (size_t)m * N;

        for (uint32_t n = 0; n < N; ++n) {
            size_t  v_idx = 0;
            otype_t sum   = otype_t(0);
            for (uint32_t blk = 0; blk < K; blk += 4) {
                uint8_t mask = row_meta[blk / 4];
                if (!mask) continue;
                for (uint32_t i = 0; i < 4; ++i) {
                    if (!(mask & (1u << i))) continue;
                    itype_t a_val = row_vals[v_idx++];
                    itype_t b_val = data_accessor_t<vt::ITYPE>::read(B, (size_t)(blk + i) * N + n);
                    sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a_val, b_val, sum);
                }
            }
            crow[n] = sum;
        }
    }
}

static SparseMat pruneAndPack(
    const std::vector<itype_t> &denseA,
    uint32_t M, uint32_t K, uint32_t sparsity_degree)
{
    SparseMat out;
    out.rows = M;
    out.cols = K;
    out.values.reserve((size_t)M * K * sparsity_degree / 4);
    out.meta.reserve((size_t)M * K / 4);

    const itype_t *src = denseA.data();
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < K; c += 4) {
            itype_t blk[4] = {
                src[r * K + c + 0], src[r * K + c + 1],
                src[r * K + c + 2], src[r * K + c + 3]
            };
            uint32_t idx[4] = {0, 1, 2, 3};
            for (uint32_t i = 3; i > 0; --i) {
                uint32_t j = rand() % (i + 1);
                std::swap(idx[i], idx[j]);
            }
            uint8_t kept[2]; // max sparsity_degree == 2
            uint8_t mask = 0;
            for (uint32_t i = 0; i < sparsity_degree; ++i) {
                kept[i] = static_cast<uint8_t>(idx[i]);
                mask |= (1u << idx[i]);
            }
            std::sort(kept, kept + sparsity_degree);
            for (uint32_t i = 0; i < sparsity_degree; ++i)
                out.values.push_back(blk[kept[i]]);
            out.meta.push_back(mask);
        }
    }
    return out;
}

static bool compare_float(otype_t a, otype_t b, int index, int errors) {
    // Sign-magnitude ordering gives correct ULP distance across zero.
    auto ordered = [](float f) -> int64_t {
        union { float f; uint32_t u; } tmp;
        tmp.f = f;
        return (tmp.u & 0x80000000u) ? -(int64_t)(tmp.u & 0x7FFFFFFFu) : (int64_t)tmp.u;
    };
    auto d = std::abs(ordered(a) - ordered(b));
    if (d > FLOAT_ULP) {
        if (errors < MAX_ERRORS)
            printf("*** error [%d]: expected=%f, actual=%f (diff=%d ULP)\n",
                   index, (double)b, (double)a, (int)d);
        return false;
    }
    return true;
}

const char *kernel_file  = "kernel.vxbin";
vx_device_h device       = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h buffer1      = nullptr;
vx_buffer_h buffer2      = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
vx_buffer_h layer_cfg_buf = nullptr;
vx_buffer_h weights_buf[NUM_LAYERS];
vx_buffer_h bias_buf[NUM_LAYERS];
kernel_arg_t kernel_arg = {};

static const uint32_t layer_in_dims [NUM_LAYERS] = { INPUT_DIM,   HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM };
static const uint32_t layer_out_dims[NUM_LAYERS] = { HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, OUTPUT_DIM  };

// ─────────────────────────────────────────────────────────────────────────────
static void show_usage() {
    std::cout << "Vortex MLP-Sparse Test." << std::endl;
    std::cout << "Usage: [-k kernel_file] [-p sparsity] [-h]" << std::endl;
    std::cout << "  -p  Sparsity degree: 1 (1:4) or 2 (2:4) [default: 2]" << std::endl;
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "k:p:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'p':
            g_sparsity_degree = atoi(optarg);
            if (g_sparsity_degree != 1 && g_sparsity_degree != 2) {
                std::cerr << "Error: sparsity must be 1 or 2" << std::endl;
                show_usage(); exit(-1);
            }
            break;
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

static void init_weights_xavier(std::vector<itype_t> &w, uint32_t fan_in, uint32_t fan_out) {
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    for (auto &v : w)
        v = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
}


static void mlp_cpu_sparse_batch(
    const std::vector<itype_t>   &input_batch,       
    std::vector<otype_t>         &output_batch,       
    const std::vector<SparseMat> &sparse_weights,     
    const std::vector<std::vector<otype_t>> &biases)
{
    std::vector<otype_t> cur(input_batch.begin(), input_batch.end());
    std::vector<otype_t> nxt;

    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t out_dim = layer_out_dims[l];
        bool     relu    = (l < NUM_LAYERS - 1);

        nxt.assign((size_t)out_dim * BATCH_SIZE, otype_t(0));
        matmul_cpu_sparseA(nxt.data(), sparse_weights[l],
                           reinterpret_cast<const itype_t *>(cur.data()),
                           BATCH_SIZE, g_sparsity_degree);

        for (uint32_t i = 0; i < out_dim; ++i) {
            for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
                otype_t v = nxt[i * BATCH_SIZE + b] + biases[l][i];
                if (relu && v < otype_t(0)) v = otype_t(0);
                nxt[i * BATCH_SIZE + b] = v;
            }
        }
        cur = nxt;
    }

    // Softmax per sample (column)
    for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
        otype_t max_val = cur[b];
        for (uint32_t i = 1; i < OUTPUT_DIM; ++i)
            max_val = std::max(max_val, cur[i * BATCH_SIZE + b]);

        otype_t sum_exp = 0;
        for (uint32_t i = 0; i < OUTPUT_DIM; ++i) {
            otype_t e = std::exp(cur[i * BATCH_SIZE + b] - max_val);
            cur[i * BATCH_SIZE + b] = e;
            sum_exp += e;
        }
        for (uint32_t i = 0; i < OUTPUT_DIM; ++i)
            cur[i * BATCH_SIZE + b] /= sum_exp;
    }

    output_batch.assign(cur.begin(), cur.end());
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    std::srand(42);

    std::cout << "=== Vortex MLP-Sparse Test ===" << std::endl;
    std::cout << "Network  : " << INPUT_DIM << " -> " << HIDDEN1_DIM
              << " -> " << HIDDEN2_DIM << " -> " << HIDDEN3_DIM
              << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "Batch    : " << BATCH_SIZE << std::endl;
    std::cout << "Sparsity : " << g_sparsity_degree << ":4" << std::endl;
    std::cout << "WMMA tile: M=" << cfg::tileM << " N=" << cfg::tileN
              << " K=" << cfg::tileK << std::endl;
    std::cout << "dtype    : " << vt::ITYPE::name << std::endl;

    RT_CHECK(vx_dev_open(&device));

    uint64_t NT;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
    if (NT != NUM_THREADS) {
        std::cout << "Error: device warp size (" << NT
                  << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
        cleanup();
        return -1;
    }

    // Validate tile and sparsity alignment for every layer
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t M = layer_out_dims[l], K = layer_in_dims[l];
        if (M % cfg::tileM || K % cfg::tileK || BATCH_SIZE % cfg::tileN || K % 4) {
            std::cout << "Error: layer " << l << " dims (" << M << "x" << K
                      << ") or BATCH_SIZE=" << BATCH_SIZE
                      << " misaligned with tile (" << cfg::tileM << "/"
                      << cfg::tileN << "/" << cfg::tileK << ") or K not multiple of 4!"
                      << std::endl;
            cleanup();
            return -1;
        }
    }

    // ── Dense weight init → sparsify → keep sparse copy for CPU reference ────
    std::vector<std::vector<itype_t>> h_weights(NUM_LAYERS);
    std::vector<SparseMat>            sparse_weights(NUM_LAYERS);
    std::vector<std::vector<otype_t>> h_biases(NUM_LAYERS);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim  = layer_in_dims[l];
        uint32_t out_dim = layer_out_dims[l];
        h_weights[l].resize((size_t)out_dim * in_dim);
        init_weights_xavier(h_weights[l], in_dim, out_dim);
        sparse_weights[l] = pruneAndPack(h_weights[l], out_dim, in_dim, g_sparsity_degree);
        h_biases[l].assign(out_dim, otype_t(0));
    }

    // ── Input: [INPUT_DIM × BATCH_SIZE], values in [0,1) ─────────────────────
    std::vector<itype_t> h_input((size_t)INPUT_DIM * BATCH_SIZE);
    for (auto &v : h_input)
        v = static_cast<itype_t>(rand()) / RAND_MAX;

    // ── Allocate device memory ────────────────────────────────────────────────
    std::cout << "Allocating device memory..." << std::endl;

    RT_CHECK(vx_mem_alloc(device, INPUT_DIM  * BATCH_SIZE * sizeof(itype_t), VX_MEM_READ,       &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer,  &kernel_arg.input_addr));

    RT_CHECK(vx_mem_alloc(device, OUTPUT_DIM * BATCH_SIZE * sizeof(otype_t), VX_MEM_READ_WRITE, &output_buffer));
    RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));

    uint32_t max_hidden = std::max({HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM});
    RT_CHECK(vx_mem_alloc(device, (size_t)max_hidden * BATCH_SIZE * sizeof(otype_t), VX_MEM_READ_WRITE, &buffer1));
    RT_CHECK(vx_mem_address(buffer1, &kernel_arg.buffer1_addr));

    RT_CHECK(vx_mem_alloc(device, (size_t)max_hidden * BATCH_SIZE * sizeof(otype_t), VX_MEM_READ_WRITE, &buffer2));
    RT_CHECK(vx_mem_address(buffer2, &kernel_arg.buffer2_addr));

    // ── Pack, allocate and upload sparse weights + biases ─────────────────────
    std::vector<layer_config_t> layer_cfgs(NUM_LAYERS);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim  = layer_in_dims[l];
        uint32_t out_dim = layer_out_dims[l];
        const auto &sw   = sparse_weights[l];

        // Layout: [ values | align-pad | metadata (uint32_t per entry) ]
        size_t values_size       = sw.values.size() * sizeof(itype_t);
        constexpr size_t meta_align = sizeof(uint32_t);
        size_t meta_offset       = align_up(values_size, meta_align);
        size_t meta_size         = sw.meta.size() * sizeof(uint32_t);
        size_t sparse_bytes      = meta_offset + meta_size;

        std::vector<uint8_t> packed(sparse_bytes, 0);
        memcpy(packed.data(), sw.values.data(), values_size);
        auto meta_words = reinterpret_cast<uint32_t *>(packed.data() + meta_offset);
        for (size_t i = 0; i < sw.meta.size(); ++i)
            meta_words[i] = static_cast<uint32_t>(sw.meta[i]);

        RT_CHECK(vx_mem_alloc(device, sparse_bytes, VX_MEM_READ, &weights_buf[l]));
        RT_CHECK(vx_mem_address(weights_buf[l], &layer_cfgs[l].weights_addr));
        RT_CHECK(vx_copy_to_dev(weights_buf[l], packed.data(), 0, sparse_bytes));

        RT_CHECK(vx_mem_alloc(device, out_dim * sizeof(otype_t), VX_MEM_READ, &bias_buf[l]));
        RT_CHECK(vx_mem_address(bias_buf[l], &layer_cfgs[l].bias_addr));
        RT_CHECK(vx_copy_to_dev(bias_buf[l], h_biases[l].data(), 0, out_dim * sizeof(otype_t)));

        layer_cfgs[l].input_dim  = in_dim;
        layer_cfgs[l].output_dim = out_dim;
        std::cout << "  Layer " << l << ": " << in_dim << " -> " << out_dim
                  << " (sparse " << g_sparsity_degree << ":4, "
                  << sparse_bytes << " bytes)" << std::endl;
    }

    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_cfg_buf));
    RT_CHECK(vx_mem_address(layer_cfg_buf, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_cfg_buf, layer_cfgs.data(), 0, NUM_LAYERS * sizeof(layer_config_t)));

    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, INPUT_DIM * BATCH_SIZE * sizeof(itype_t)));

    kernel_arg.num_layers      = NUM_LAYERS;
    kernel_arg.batch_size      = BATCH_SIZE;
    kernel_arg.sparsity_degree = g_sparsity_degree;
    kernel_arg._pad            = 0;

    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    std::cout << "Starting MLP-Sparse inference..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Elapsed time: %.3f ms\n",
           std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);

    std::vector<otype_t> h_output((size_t)OUTPUT_DIM * BATCH_SIZE);
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0,
                              OUTPUT_DIM * BATCH_SIZE * sizeof(otype_t)));

    // CPU reference uses the same sparse matrices as uploaded to the device
    std::vector<otype_t> h_ref;
    mlp_cpu_sparse_batch(h_input, h_ref, sparse_weights, h_biases);

    int errors = 0;
    for (uint32_t i = 0; i < OUTPUT_DIM * BATCH_SIZE; ++i) {
        if (!compare_float(h_output[i], h_ref[i], i, errors))
            ++errors;
    }

    cleanup();

    if (errors != 0) {
        std::cout << "Found " << errors << " / " << (OUTPUT_DIM * BATCH_SIZE)
                  << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
