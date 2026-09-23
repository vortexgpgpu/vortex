// llama_dtcu host program.
//
//   --probe -m <mode> -M <M> -N <N> -K <K> [--rep <n>]
//       One D = C + A*B with the microbenchmark's synthetic operands on one mode, verified
//       against the CPU, run `rep` times in the SAME device session (cold, then warm).
//
//   --run <model.bin> -z <tokenizer.bin> [-i "<prompt>"] [-T <rows>] [-m <mode>]
//         [--modemap <file>] [--fallback <mode>] [--noverify] [--notext]
//       Prefill forward of the prompt (BOS prepended, padded to T rows) with every GEMM on
//       mode <mode> (or the per-site modes in <file>: lines "<site> <mode>", sites qkv qk pv
//       wo w13 w2 attn). Prints [LLAMA] lines per site and in total; with verification on,
//       [LLAMA-VERIFY] lines against the host fp16-emulating reference.
//
//   --decode <model.bin> -z <tokenizer.bin> [-i "<prompt>"] [-B <seqs>] [-m <mode>]
//         [--modemap <file>] [--fallback <mode>] [--noverify] [--notext]
//       One batched decode step: B sequences, each with the prompt in its KV cache, each
//       generating the token after it. GEMM sites qkv wo w13 w2 cls (modemap also accepts
//       "lin" for the four linear layers); attention is a SIMT pass. Same output format.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <VX_types.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

using namespace vortex;   // bit_cast, as cgo27_motivation/main.cpp does before its host headers

#include "common.h"
#include "host_types.h"
#include "host_modes.h"
#include "llama_dev.h"
#include "llama_forward.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                                   \
  do {                                                                    \
    int _ret = (int)(_expr);                                              \
    if (0 != _ret) {                                                      \
      std::cerr << "llama_dtcu: " << #_expr << " returned " << _ret       \
                << std::endl;                                             \
      return _ret;                                                        \
    }                                                                     \
  } while (false)

static uint32_t parse_u32(const char* s, const char* flag) {
  char* end = nullptr;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0') {
    std::cerr << "llama_dtcu: invalid " << flag << " '" << s << "'" << std::endl;
    exit(-1);
  }
  return (uint32_t)v;
}

static int run_probe(uint32_t mode, uint32_t M, uint32_t N, uint32_t K, uint32_t reps) {
  LlamaDevice dev;
  RT_CHECK(dev.open());
  const std::string why = dev.mode_illegal_reason(mode, M, N, K);
  if (!why.empty()) {
    std::cerr << "llama_dtcu: " << why << std::endl;
    return -1;
  }

  // The microbenchmark's operand formulas (main.cpp), so a [PROBE] cell and a [MOTI] cell
  // of the same shape run the same numbers through the same program.
  std::vector<itype_t> hA((size_t)M * K), hB((size_t)K * N);
  std::vector<otype_t> hC((size_t)M * N);
  std::vector<float>   hRef((size_t)M * N);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t k = 0; k < K; ++k)
      hA[(size_t)i * K + k] = Convert<vt::ITYPE>::from_float(float((i * 13 + k * 7) % 11) - 5.0f);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j)
      hB[(size_t)j * K + k] = Convert<vt::ITYPE>::from_float(float((k * 5 + j * 17) % 9) - 4.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      hC[(size_t)i * N + j] = Convert<vt::OTYPE>::from_float(float((i * 9 + j * 11) % 13) - 6.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float acc = Convert<vt::OTYPE>::to_float(hC[(size_t)i * N + j]);
      for (uint32_t k = 0; k < K; ++k)
        acc += Convert<vt::ITYPE>::to_float(hA[(size_t)i * K + k])
             * Convert<vt::ITYPE>::to_float(hB[(size_t)j * K + k]);
      hRef[(size_t)i * N + j] = acc;
    }

  vx_buffer_h A_buf, B_buf, C_buf, D_buf;
  uint64_t A_addr, B_addr, C_addr, D_addr;
  RT_CHECK(dev.alloc(hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf, &A_addr));
  RT_CHECK(dev.alloc(hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf, &B_addr));
  RT_CHECK(dev.alloc(hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf, &C_addr));
  RT_CHECK(dev.alloc(hRef.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf, &D_addr));
  RT_CHECK(dev.upload(A_buf, 0, hA.data(), hA.size() * sizeof(itype_t)));
  RT_CHECK(dev.upload(B_buf, 0, hB.data(), hB.size() * sizeof(itype_t)));
  RT_CHECK(dev.upload(C_buf, 0, hC.data(), hC.size() * sizeof(otype_t)));

  std::vector<otype_t> out((size_t)M * N), zeros((size_t)M * N, otype_t{});
  for (uint32_t rep = 1; rep <= reps; ++rep) {
    RT_CHECK(dev.upload(D_buf, 0, zeros.data(), zeros.size() * sizeof(otype_t)));
    LaunchStats st{};
    RT_CHECK(dev.gemm(mode, M, N, K, A_addr, B_addr, C_addr, D_addr, &st));
    RT_CHECK(dev.download(out.data(), D_buf, 0, out.size() * sizeof(otype_t)));
    int errors = 0;
    for (size_t idx = 0; idx < out.size(); ++idx) {
      const float got = Convert<vt::OTYPE>::to_float(out[idx]);
      if (ulp_diff(got, hRef[idx]) > FLOAT_ULP) {
        if (errors < 3)
          std::cerr << "  mismatch D[" << idx << "]: got=" << got << " exp=" << hRef[idx] << "\n";
        ++errors;
      }
    }
    std::printf("[PROBE] mode=%u name=%s M=%u N=%u K=%u rep=%u launches=%u cycles=%llu "
                "instrs=%llu errors=%d host_ms=%.1f\n",
                mode, kShortNames[mode], M, N, K, rep, st.launches,
                (unsigned long long)st.cycles, (unsigned long long)st.instrs, errors, st.host_ms);
    if (errors) return errors;
  }
  vx_buffer_release(A_buf); vx_buffer_release(B_buf);
  vx_buffer_release(C_buf); vx_buffer_release(D_buf);
  std::printf("PASSED!\n");
  return 0;
}

static void usage() {
  std::cerr << "usage: llama_dtcu --probe -m <mode> -M <M> -N <N> -K <K> [--rep <n>]\n"
               "       llama_dtcu --run <model.bin> -z <tokenizer.bin> [-i <prompt>] [-T <rows>] [-m <mode>]\n"
               "                  [--modemap <file>] [--fallback <mode>] [--noverify] [--notext]\n"
               "       llama_dtcu --decode <model.bin> -z <tokenizer.bin> [-i <prompt>] [-B <seqs>] [-m <mode>]\n"
               "                  [--modemap <file>] [--fallback <mode>] [--noverify] [--notext]" << std::endl;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--probe") == 0) {
    uint32_t mode = 1, M = 128, N = 288, K = 288, reps = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
      if      (!std::strcmp(argv[i], "-m"))    mode = parse_u32(argv[i + 1], "-m");
      else if (!std::strcmp(argv[i], "-M"))    M    = parse_u32(argv[i + 1], "-M");
      else if (!std::strcmp(argv[i], "-N"))    N    = parse_u32(argv[i + 1], "-N");
      else if (!std::strcmp(argv[i], "-K"))    K    = parse_u32(argv[i + 1], "-K");
      else if (!std::strcmp(argv[i], "--rep")) reps = parse_u32(argv[i + 1], "--rep");
      else { std::cerr << "llama_dtcu: unknown option " << argv[i] << std::endl; return -1; }
    }
    return run_probe(mode, M, N, K, reps);
  }
  const bool is_run = argc >= 3 && std::strcmp(argv[1], "--run") == 0;
  const bool is_decode = argc >= 3 && std::strcmp(argv[1], "--decode") == 0;
  if (is_run || is_decode) {
    RunOptions opt;
    opt.model = argv[2];
    opt.prompt = "Once upon a time";
    for (int i = 3; i < argc; ++i) {
      auto need = [&](const char* flag) -> const char* {
        if (i + 1 >= argc) { std::cerr << "llama_dtcu: " << flag << " needs a value" << std::endl; exit(-1); }
        return argv[++i];
      };
      if      (!std::strcmp(argv[i], "-z"))         opt.tokenizer = need("-z");
      else if (!std::strcmp(argv[i], "-i"))         opt.prompt    = need("-i");
      else if (!std::strcmp(argv[i], "-T"))         opt.T         = parse_u32(need("-T"), "-T");
      else if (!std::strcmp(argv[i], "-B"))         opt.B         = parse_u32(need("-B"), "-B");
      else if (!std::strcmp(argv[i], "-m"))         opt.mode      = (int)parse_u32(need("-m"), "-m");
      else if (!std::strcmp(argv[i], "--modemap"))  opt.modemap   = need("--modemap");
      else if (!std::strcmp(argv[i], "--fallback")) opt.fallback  = (int)parse_u32(need("--fallback"), "--fallback");
      else if (!std::strcmp(argv[i], "--noverify")) opt.verify    = false;
      else if (!std::strcmp(argv[i], "--notext"))   opt.text      = false;
      else { std::cerr << "llama_dtcu: unknown option " << argv[i] << std::endl; usage(); return -1; }
    }
    if (opt.tokenizer.empty()) { usage(); return -1; }
    return is_decode ? run_decode(opt) : run_llama(opt);
  }
  usage();
  return -1;
}
