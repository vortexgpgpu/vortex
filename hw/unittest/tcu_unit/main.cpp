// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_tcu_unit end-to-end testbench.
// Drives WMMA operations through VX_dispatch_if and verifies the result
// returned on VX_commit_if against a software floating-point reference.

#include "VVX_tcu_unit_top.h"
#include <verilated.h>
#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif
#ifdef SAIF_OUTPUT
#include <verilated_saif_c.h>
#endif

#if defined(VCD_OUTPUT) && defined(SAIF_OUTPUT)
#error "VCD_OUTPUT and SAIF_OUTPUT cannot both be defined"
#endif

#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "VX_config.h"
#include "softfloat_ext.h"
#include <bitmanip.h>

bool sim_trace_enabled() { return true; }

static uint64_t timestamp = 0;
double sc_time_stamp() { return timestamp; }

// ---- TCU block dimensions (derived from NUM_THREADS) ----------------------
// TCU_BLOCK_CAP = NUM_THREADS, TCU_LG_BLOCK_CAP = log2(NUM_THREADS)
// TCU_BLOCK_EM  = LG / 2 (rounded up), TCU_BLOCK_EN = LG - EM
// TCU_TC_M = 2^EM, TCU_TC_N = 2^EN, TCU_TC_K = NT / max(TC_M, TC_N)

static_assert((NUM_THREADS & (NUM_THREADS - 1)) == 0, "NUM_THREADS must be power of 2");

constexpr int log2_floor(int n) {
    return (n <= 1) ? 0 : 1 + log2_floor(n / 2);
}

constexpr int TCU_LG = log2_floor(NUM_THREADS);
constexpr int TCU_EN = TCU_LG / 2;
constexpr int TCU_EM = TCU_LG - TCU_EN;
constexpr int TCU_TC_M = 1 << TCU_EM;
constexpr int TCU_TC_N = 1 << TCU_EN;
constexpr int TCU_TC_K = NUM_THREADS / ((TCU_TC_M > TCU_TC_N) ? TCU_TC_M : TCU_TC_N);

static_assert(TCU_TC_M * TCU_TC_N == NUM_THREADS, "TC dimension check");

// Words per lane in the Verilator signal (XLEN/32: 1 for 32-bit, 2 for 64-bit)
constexpr int LANE_WORDS = XLEN / 32;

// TCU format IDs (must match VX_tcu_pkg.sv)
constexpr uint32_t TCU_FP32_ID = 0;
constexpr uint32_t TCU_FP16_ID = 1;
constexpr uint32_t TCU_BF16_ID = 2;
constexpr uint32_t TCU_TF32_ID = 5;

// WMMA op_type (INST_TCU_WMMA = 4'h0 from VX_gpu_pkg.sv)
constexpr uint32_t INST_TCU_WMMA = 0;

// Wide-signal helpers work on any Verilator type (IData/QData/VlWide<N>).
// Each lane occupies LANE_WORDS 32-bit words; the value lives in word [0] of the lane.
static void write_lane(void *arr_raw, int t, uint32_t val) {
    auto *arr = static_cast<uint32_t *>(arr_raw);
    arr[t * LANE_WORDS] = val;
    for (int w = 1; w < LANE_WORDS; w++)
        arr[t * LANE_WORDS + w] = 0;
}

static uint32_t read_lane(const void *arr_raw, int t) {
    const auto *arr = static_cast<const uint32_t *>(arr_raw);
    return arr[t * LANE_WORDS];
}

// ---- Helper: bit_cast -------------------------------------------------------
template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) &&
                     std::is_trivially_copyable_v<From> &&
                     std::is_trivially_copyable_v<To>,
                 To>
bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>);
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

// ---- Format helpers ---------------------------------------------------------
struct FmtInfo {
    uint32_t id;
    int exp_bits;
    int sig_bits;
    int width_bits; // 16, 19 (tf32), or 32
};

static FmtInfo get_fmt_info(uint32_t fmt_id) {
    switch (fmt_id) {
    case TCU_FP32_ID: return {TCU_FP32_ID, 8, 23, 32};
    case TCU_FP16_ID: return {TCU_FP16_ID, 5, 10, 16};
    case TCU_BF16_ID: return {TCU_BF16_ID, 8,  7, 16};
    case TCU_TF32_ID: return {TCU_TF32_ID, 8, 10, 19};
    default:
        std::cerr << "Unsupported format id: " << fmt_id << "\n";
        std::abort();
    }
}

// Convert a packed source-format element to float (for software reference).
static float elem_to_float(uint32_t raw, const FmtInfo &fi) {
    return cvt_custom_to_f32(raw, fi.exp_bits, fi.sig_bits, 0 /*RNE*/, nullptr);
}

// ---- Software WMMA reference -----------------------------------------------
// Computes D[t] = sum_k(A[i][k] * B[k][j]) + C[t]
// where t = i*TC_N + j, A in row-major, B in col-major (each column consecutive).
//
// rs1_data layout: lane t -> A[i][k] for t = i*TC_K + k (A in row-major)
//   BUT each OUTPUT thread t=(i,j) shares the same A row.
//   a_row[k] = rs1_data[i * TC_K + k] (all TC_N output threads in the same row share this)
// rs2_data layout: lane t -> B[k][j] for t = j*TC_K + k (B^T in row-major, i.e. B columns packed)
//   b_col[k] = rs2_data[j * TC_K + k]
// rs3_data layout: lane t -> C[i][j] where t = i*TC_N + j
//
static std::vector<float> wmma_ref(
    const std::vector<uint32_t> &rs1,  // NUM_THREADS elements (A, row-major)
    const std::vector<uint32_t> &rs2,  // NUM_THREADS elements (B columns, col-major)
    const std::vector<uint32_t> &rs3,  // NUM_THREADS elements (C, row-major)
    const FmtInfo &fi_s)               // source format (for A and B)
{
    std::vector<float> result(NUM_THREADS);

    for (int i = 0; i < TCU_TC_M; i++) {
        for (int j = 0; j < TCU_TC_N; j++) {
            int t = i * TCU_TC_N + j;
            // Match DPI: accumulate with fp32 rounding at each multiply and add step.
            float prod = 0.0f;
            for (int k = 0; k < TCU_TC_K; k++) {
                float a = elem_to_float(rs1[i * TCU_TC_K + k], fi_s);
                float b = elem_to_float(rs2[j * TCU_TC_K + k], fi_s);
                prod += a * b;
            }
            float c = bit_cast<float>(rs3[t]);  // C is always fp32
            result[t] = prod + c;
        }
    }
    return result;
}

// ---- Approximate equality (ULP-based) --------------------------------------
static int approx_equal(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return 0;
    if (std::isinf(a) && std::isinf(b))
        return (std::signbit(a) == std::signbit(b)) ? 0 : 1;
    uint32_t xa = bit_cast<uint32_t>(a) ^ 0x80000000u;
    uint32_t xb = bit_cast<uint32_t>(b) ^ 0x80000000u;
    return (int)(xa - xb);
}

// ---- Float printing --------------------------------------------------------
static void print_float(const std::string &prefix, float v, bool nl = true) {
    std::cout << prefix << v << " (0x" << std::hex << bit_cast<uint32_t>(v) << ")" << std::dec;
    if (nl) std::cout << "\n";
}

// ---- RoundingMode ----------------------------------------------------------
enum class RoundingMode { RNE = 0, RTZ = 1, RDN = 2, RUP = 3, RMM = 4 };

static RoundingMode frm_from_string(const std::string &s) {
    if (s == "RNE") return RoundingMode::RNE;
    if (s == "RTZ") return RoundingMode::RTZ;
    if (s == "RDN") return RoundingMode::RDN;
    if (s == "RUP") return RoundingMode::RUP;
    if (s == "RMM") return RoundingMode::RMM;
    throw std::invalid_argument("Unknown rounding mode: " + s);
}

// ---- Test configuration ---------------------------------------------------
struct TestConfig {
    uint64_t    max_cycles  = 2000;
    bool        enable_trace = true;
    uint64_t    trace_start = 0;
    uint64_t    trace_end   = ~0ULL;
    unsigned    random_seed = 12345;
    uint32_t    fmt_s       = TCU_FP16_ID; // source format (A, B)
    RoundingMode frm        = RoundingMode::RNE;
    uint32_t    num_tests   = 1000;
    int         ulp         = 1;
    int32_t     test_id     = -1;
    std::vector<std::string> features = {"infinities","nans","subnormals","normals","zeros"};
};

// ---- Test value generation (same helpers as tcu_fedp) ----------------------
static uint32_t pack_fp(uint32_t sign, uint32_t exp, uint32_t frac,
                        uint32_t eb, uint32_t sb) {
    return (sign << (eb + sb)) | (exp << sb) | frac;
}

static uint32_t gen_fp_value(const std::string &feat, uint32_t eb, uint32_t sb,
                              uint32_t test_id, std::mt19937 &rng) {
    uint32_t all_exp = (eb == 32) ? 0xFFFFFFFFu : ((1u << eb) - 1u);
    uint32_t max_frac = (sb == 32) ? 0xFFFFFFFFu : ((1u << sb) - 1u);
    std::uniform_int_distribution<uint32_t> sign_d(0, 1);
    std::uniform_int_distribution<uint32_t> exp_d(1, (all_exp > 0) ? all_exp - 1 : 0);
    std::uniform_int_distribution<uint32_t> frac_d(0, max_frac);
    std::uniform_int_distribution<uint32_t> frac_nz_d(1, std::max(max_frac, 1u));

    if (feat == "zeros") {
        return pack_fp((test_id & 1) ? 1 : 0, 0, 0, eb, sb);
    } else if (feat == "normals") {
        uint32_t s = sign_d(rng);
        uint32_t e = exp_d(rng);
        uint32_t f = frac_d(rng);
        return pack_fp(s, e, f, eb, sb);
    } else if (feat == "subnormals") {
        if (sb == 0) return pack_fp(0, 0, 0, eb, sb);
        return pack_fp(sign_d(rng), 0, frac_nz_d(rng), eb, sb);
    } else if (feat == "infinities") {
        return pack_fp((test_id & 1), all_exp, 0, eb, sb);
    } else if (feat == "nans") {
        if (sb == 0) return pack_fp(sign_d(rng), all_exp, 0, eb, sb);
        uint32_t qbit = 1u << (sb - 1);
        uint32_t payload = frac_d(rng) & (qbit - 1);
        if (!payload) payload = 1;
        return pack_fp(sign_d(rng), all_exp, qbit | payload, eb, sb);
    }
    std::cerr << "Unknown feature: " << feat << "\n";
    std::abort();
}

// ---- Testbench class -------------------------------------------------------
class Testbench {
public:
    explicit Testbench(const TestConfig &cfg)
        : cfg_(cfg), cycle_(0), rng_(cfg.random_seed) {
        Verilated::traceEverOn(cfg_.enable_trace);
        dut_ = std::make_unique<VVX_tcu_unit_top>();
#ifdef VCD_OUTPUT
        if (cfg_.enable_trace) {
            trace_ = std::make_unique<VerilatedVcdC>();
            dut_->trace(trace_.get(), 99);
            const char* vcd_file = std::getenv("VCD_FILE");
            trace_->open(vcd_file ? vcd_file : "trace.vcd");
        }
#endif
#ifdef SAIF_OUTPUT
        if (cfg_.enable_trace) {
            saif_ = std::make_unique<VerilatedSaifC>();
            dut_->trace(saif_.get(), 99);
            const char* saif_file = std::getenv("SAIF_FILE");
            saif_->open(saif_file ? saif_file : "trace.saif");
        }
#endif
        dut_->clk = 0;
        dut_->reset = 0;
        idle_dispatch();
        dut_->commit_ready = 1;
    }

    ~Testbench() {
#ifdef VCD_OUTPUT
        if (trace_) trace_->close();
#endif
#ifdef SAIF_OUTPUT
        if (saif_) saif_->close();
#endif
    }

    bool run_tests() {
        reset();

        const FmtInfo fi = get_fmt_info(cfg_.fmt_s);
        std::cout << "Testing WMMA fmt_s=" << fi.id
                  << " (" << fi.exp_bits << "e" << fi.sig_bits << "m)"
                  << " TCU_TC_M=" << TCU_TC_M
                  << " TCU_TC_N=" << TCU_TC_N
                  << " TCU_TC_K=" << TCU_TC_K
                  << " NUM_THREADS=" << NUM_THREADS
                  << "\n";

        uint32_t tests_done = 0;
        for (uint32_t test_id = 0; test_id < cfg_.num_tests; test_id++) {
            if (cfg_.test_id >= 0 && (int)test_id != cfg_.test_id)
                continue;

            // Select feature to test
            size_t feat_idx = test_id % cfg_.features.size();
            const std::string &feat = cfg_.features[feat_idx];

            // Generate source elements for A and B (NUM_THREADS values each)
            std::vector<uint32_t> rs1(NUM_THREADS), rs2(NUM_THREADS), rs3(NUM_THREADS);
            for (int t = 0; t < NUM_THREADS; t++) {
                rs1[t] = gen_fp_value(feat, fi.exp_bits, fi.sig_bits, test_id * NUM_THREADS + t, rng_);
                rs2[t] = gen_fp_value("normals", fi.exp_bits, fi.sig_bits, test_id * NUM_THREADS + t, rng_);
                // C accumulator is always fp32
                float c = bit_cast<float>(gen_fp_value("normals", 8, 23, test_id * NUM_THREADS + t + 1, rng_));
                rs3[t] = bit_cast<uint32_t>(c);
            }

            // Compute software reference
            std::vector<float> expected = wmma_ref(rs1, rs2, rs3, fi);

            // Drive DUT
            std::vector<uint32_t> result = drive_and_collect(rs1, rs2, rs3, fi);

            // Verify
            bool test_pass = true;
            for (int t = 0; t < NUM_THREADS; t++) {
                float act = bit_cast<float>(result[t]);
                int delta = approx_equal(act, expected[t]);
                if (std::abs(delta) > cfg_.ulp) {
                    if (test_pass) {
                        // First failure for this test
                        std::cout << "Test #" << test_id << " (" << feat << ") FAILED:\n";
                    }
                    std::cout << "  thread=" << t
                              << "  expected="; print_float("", expected[t], false);
                    std::cout << "  actual=";  print_float("", act, false);
                    std::cout << "  delta=" << delta << "\n";
                    test_pass = false;
                }
            }
            if (!test_pass) return false;

            tests_done++;
        }

        if (cfg_.test_id >= 0) {
            std::cout << "Test #" << cfg_.test_id << " PASSED!\n";
        } else {
            std::cout << tests_done << " test(s) PASSED!\n";
        }
        std::cout << "Simulation completed in " << cycle_ << " cycles\n";
        return true;
    }

private:
    TestConfig cfg_;
    uint64_t cycle_;
    std::mt19937 rng_;
    std::unique_ptr<VVX_tcu_unit_top> dut_;
#ifdef VCD_OUTPUT
    std::unique_ptr<VerilatedVcdC> trace_;
#endif
#ifdef SAIF_OUTPUT
    std::unique_ptr<VerilatedSaifC> saif_;
#endif

    void tick() {
        dut_->clk = 0;
        dut_->eval();
#ifdef VCD_OUTPUT
        if (cfg_.enable_trace && timestamp >= cfg_.trace_start && timestamp < cfg_.trace_end)
            trace_->dump(timestamp);
#endif
#ifdef SAIF_OUTPUT
        if (cfg_.enable_trace && timestamp >= cfg_.trace_start && timestamp < cfg_.trace_end)
            saif_->dump(timestamp);
#endif
        timestamp++;
        dut_->clk = 1;
        dut_->eval();
#ifdef VCD_OUTPUT
        if (cfg_.enable_trace && timestamp >= cfg_.trace_start && timestamp < cfg_.trace_end)
            trace_->dump(timestamp);
#endif
#ifdef SAIF_OUTPUT
        if (cfg_.enable_trace && timestamp >= cfg_.trace_start && timestamp < cfg_.trace_end)
            saif_->dump(timestamp);
#endif
        timestamp++;
        cycle_++;
    }

    void reset(int cycles = 4) {
        dut_->reset = 1;
        for (int i = 0; i < cycles; i++) tick();
        dut_->reset = 0;
    }

    void idle_dispatch() {
        dut_->dispatch_valid     = 0;
        dut_->dispatch_uuid      = 0;
        dut_->dispatch_wis       = 0;
        dut_->dispatch_sid       = 0;
        dut_->dispatch_tmask     = 0;
        dut_->dispatch_PC        = 0;
        dut_->dispatch_wb        = 0;
        dut_->dispatch_wr_xregs  = 0;
        dut_->dispatch_rd        = 0;
        dut_->dispatch_bytesel   = 0;
        dut_->dispatch_op_type   = 0;
        dut_->dispatch_fmt_s     = 0;
        dut_->dispatch_fmt_d     = 0;
        dut_->dispatch_step_m    = 0;
        dut_->dispatch_step_n    = 0;
        dut_->dispatch_step_k    = 0;
        dut_->dispatch_sop       = 0;
        dut_->dispatch_eop       = 0;
        std::memset(&dut_->dispatch_rs1_data, 0, sizeof(dut_->dispatch_rs1_data));
        std::memset(&dut_->dispatch_rs2_data, 0, sizeof(dut_->dispatch_rs2_data));
        std::memset(&dut_->dispatch_rs3_data, 0, sizeof(dut_->dispatch_rs3_data));
    }

    // Drive one WMMA dispatch packet and wait for commit result.
    std::vector<uint32_t> drive_and_collect(
        const std::vector<uint32_t> &rs1,
        const std::vector<uint32_t> &rs2,
        const std::vector<uint32_t> &rs3,
        const FmtInfo &fi)
    {
        // Set dispatch fields
        dut_->dispatch_valid    = 1;
        dut_->dispatch_uuid     = 0;
        dut_->dispatch_wis      = 0;
        dut_->dispatch_sid      = 0;
        dut_->dispatch_tmask    = (1u << NUM_THREADS) - 1;  // all threads active
        dut_->dispatch_PC       = 0;
        dut_->dispatch_wb       = 1;
        dut_->dispatch_wr_xregs = 0;
        dut_->dispatch_rd       = 1;  // arbitrary destination register
        dut_->dispatch_bytesel  = 0;
        dut_->dispatch_op_type  = INST_TCU_WMMA;
        dut_->dispatch_fmt_s    = fi.id;
        dut_->dispatch_fmt_d    = TCU_FP32_ID;  // output always fp32
        dut_->dispatch_step_m   = 0;
        dut_->dispatch_step_n   = 0;
        dut_->dispatch_step_k   = 0;
        dut_->dispatch_sop      = 1;
        dut_->dispatch_eop      = 1;

        for (int t = 0; t < NUM_THREADS; t++) {
            write_lane(&dut_->dispatch_rs1_data, t, rs1[t]);
            write_lane(&dut_->dispatch_rs2_data, t, rs2[t]);
            write_lane(&dut_->dispatch_rs3_data, t, rs3[t]);
        }

        // Clock until dispatch fires (ready goes high)
        dut_->commit_ready = 1;
        uint64_t start_cycle = cycle_;
        while (true) {
            if (cycle_ - start_cycle > cfg_.max_cycles) {
                std::cerr << "TIMEOUT waiting for dispatch_ready\n";
                std::abort();
            }
            tick();
            if (dut_->dispatch_ready) break;
        }
        // Dispatch fired: de-assert valid
        idle_dispatch();

        // Wait for commit result
        start_cycle = cycle_;
        while (!dut_->commit_valid) {
            if (cycle_ - start_cycle > cfg_.max_cycles) {
                std::cerr << "TIMEOUT waiting for commit_valid\n";
                std::abort();
            }
            tick();
        }

        // Collect result from commit data (one cycle with commit_ready=1)
        std::vector<uint32_t> result(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; t++) {
            result[t] = read_lane(&dut_->commit_data, t);
        }
        tick();  // consume the commit beat

        return result;
    }
};

// ---- Command-line parsing --------------------------------------------------
static TestConfig parse_args(int argc, char **argv) {
    TestConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-trace") {
            cfg.enable_trace = false;
        } else if (arg.substr(0, 6) == "--fmt=") {
            cfg.fmt_s = std::stoi(arg.substr(6));
        } else if (arg.substr(0, 6) == "--frm=") {
            cfg.frm = frm_from_string(arg.substr(6));
        } else if (arg.substr(0, 8) == "--tests=") {
            cfg.num_tests = std::stoi(arg.substr(8));
        } else if (arg.substr(0, 7) == "--test=") {
            cfg.test_id = std::stoi(arg.substr(7));
        } else if (arg.substr(0, 7) == "--seed=") {
            cfg.random_seed = std::stoi(arg.substr(7));
        } else if (arg.substr(0, 6) == "--ulp=") {
            cfg.ulp = std::stoi(arg.substr(6));
        } else if (arg.substr(0, 14) == "--trace-start=") {
            cfg.trace_start = std::stoull(arg.substr(14));
        } else if (arg.substr(0, 12) == "--trace-end=") {
            cfg.trace_end = std::stoull(arg.substr(12));
        } else if (arg.substr(0, 11) == "--features=") {
            cfg.features.clear();
            std::stringstream ss(arg.substr(11));
            std::string item;
            while (std::getline(ss, item, ',')) cfg.features.push_back(item);
        } else {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --fmt=<id>        Source format (0=fp32,1=fp16,2=bf16,5=tf32)\n"
                      << "  --frm=<mode>      Rounding mode (RNE,RTZ,RDN,RUP,RMM)\n"
                      << "  --tests=<N>       Number of test cases (default 1000)\n"
                      << "  --test=<id>       Run a single test by id\n"
                      << "  --seed=<N>        Random seed\n"
                      << "  --ulp=<N>         ULP tolerance (default 1)\n"
                      << "  --no-trace        Disable VCD output\n"
                      << "  --features=LIST   Comma-separated list of features to test\n"
                      << "  --trace-start=N   Start VCD trace at timestamp N\n"
                      << "  --trace-end=N     Stop VCD trace at timestamp N\n";
            std::exit(0);
        }
    }
    return cfg;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    TestConfig cfg = parse_args(argc, argv);

    Testbench tb(cfg);
    if (!tb.run_tests()) {
        return 1;
    }
    return 0;
}
