// dtcu_xcore: does a core that never submitted the descriptor observe its completion,
// and is the output actually visible to that core when it does?
//
// This is the property the whole completion design rests on. dtensor_*_start returns a
// ticket in a register, and a register never reaches another core, so completion has to
// live in memory -- in the descriptor, which is the only object both the submitter and
// an arbitrary consumer can address. Every other DTCU test submits and checks from the
// same warp, which exercises none of that: the submitting core already has the line.
//
// It is a separate test rather than a mode of dtcu_basic because it needs a multi-core
// build, and making dtcu_basic multi-core would move the cycle baseline that test is
// held against.

#include "common.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <dtcu_cfg.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>
#include <vortex2.h>

#define FLOAT_ULP 6
#define MAX_ERRORS 20

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 != _ret) {                                         \
      std::cerr << "Runtime Error: " << #_expr               \
                << " returned " << _ret << std::endl;        \
      return _ret;                                           \
    }                                                        \
  } while (false)

using namespace vortex;
namespace vt = vortex::tensor;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

template <typename T> struct Convert;
template <> struct Convert<vt::fp32> {
  using dtype = float;
  static inline dtype from_float(float f) { return f; }
  static inline float to_float(dtype x) { return x; }
};
template <> struct Convert<vt::fp16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_htof_s(x, 0, nullptr)); }
};
template <> struct Convert<vt::bf16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftob_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_btof_s(x, 0, nullptr)); }
};

static inline int ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isinf(a) || std::isinf(b)) return (a == b) ? 0 : 0x7fffffff;
  int ia, ib;
  std::memcpy(&ia, &a, sizeof(int));
  std::memcpy(&ib, &b, sizeof(int));
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  return std::abs(ia - ib);
}

#define FAIL(msg)                                                        \
  do {                                                                   \
    std::cerr << tag << ": " << msg << std::endl;                        \
    ++errors;                                                            \
  } while (false)

static int run_engine(int engine, const char* tag, bool* skipped, int* errors_out) {
  *skipped = false;
  *errors_out = 0;
  int errors = 0;

  // Deliberately MULTI-TILE and RAGGED on every axis, for both engines. Multi-tile so
  // completion is not instantaneous and the consumers really do spin across the 0 -> 1
  // transition; ragged so the masked trailing-tile store is part of what "done" has to
  // be ordered after.
  const uint32_t M = 100, N = 96, K = 40;

  std::vector<itype_t> hA(M * K), hB(K * N);
  std::vector<otype_t> hD(M * N);
  std::vector<float>   hRef(M * N);

  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t k = 0; k < K; ++k)
      hA[i * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(float((i * 13 + k * 7) % 11) - 5.0f);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j)
      hB[j * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(float((k * 5 + j * 17) % 9) - 4.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        acc += Convert<vt::ITYPE>::to_float(hA[i * K + k])
             * Convert<vt::ITYPE>::to_float(hB[j * K + k]);
      hRef[i * N + j] = acc;
    }

  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));

  uint64_t num_cores = 0, socket_size = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_SOCKET_SIZE, &socket_size));
  if (socket_size == 0) socket_size = 1;

  {
    const uint64_t need = (engine == DTCU_ENGINE_CLUSTER) ? VX_ISA_EXT_DTCU_CLUSTER
                                                          : VX_ISA_EXT_DTCU_SOCKET;
    uint64_t isa_flags = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & need) == 0) {
      std::cout << tag << ": skipped (engine not present in this build)" << std::endl;
      *skipped = true;
      vx_device_release(device);
      return 0;
    }
  }
  if (num_cores < 2) {
    // Not a skip: this test's only reason to exist is the second core.
    std::cerr << tag << ": build has " << num_cores
              << " core(s); the cross-core property cannot be tested" << std::endl;
    vx_device_release(device);
    *errors_out = 1;
    return 0;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  kernel_arg_t karg{};
  karg.engine = (uint32_t)engine;
  karg.M = M; karg.N = N; karg.K = K;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, D_buf = nullptr,
              desc_buf = nullptr, ctl_buf = nullptr;
  RT_CHECK(vx_buffer_create(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &karg.B_addr));
  RT_CHECK(vx_buffer_create(device, hD.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &karg.D_addr));

  const uint32_t tile_n = (engine == DTCU_ENGINE_CLUSTER) ? 32u : dtcu_tile_n_max_of(engine);
  if (!dtcu_tile_n_valid_of(engine, tile_n)) {
    std::cerr << tag << ": tile_n=" << tile_n << " illegal for this engine" << std::endl;
    return -1;
  }

  dtensor_desc_t desc{};
  desc.ptrA = karg.A_addr; desc.ptrB = karg.B_addr; desc.ptrC = 0; desc.ptrD = karg.D_addr;
  desc.ldmA = K; desc.ldmB = K; desc.ldmC = 0; desc.ldmD = N;
  desc.M = (uint16_t)M; desc.N = (uint16_t)N; desc.K = (uint16_t)K;
  desc.fmt_s = vt::ITYPE::id; desc.fmt_d = vt::OTYPE::id;
  desc.flags = DTENSOR_FLAG_ZERO_ACC;
  desc.shape_n_size = dtcu_shape_n_size(tile_n);
  desc.shape_policy = 0;
  desc.done = 0;
  RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ_WRITE, &desc_buf));
  RT_CHECK(vx_buffer_address(desc_buf, &karg.desc_addr));

  xcore_ctl_t ctl{};
  RT_CHECK(vx_buffer_create(device, sizeof(xcore_ctl_t), VX_MEM_READ_WRITE, &ctl_buf));
  RT_CHECK(vx_buffer_address(ctl_buf, &karg.ctl_addr));

  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, &desc, sizeof(desc), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, ctl_buf, 0, &ctl, sizeof(ctl), 0, nullptr, nullptr));

  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // One CTA per core, one thread per CTA: a CTA then occupies exactly one warp slot, so
  // the dispatcher's one-CTA-per-core-per-cycle throttle spreads them across the cores.
  // That is a tendency, not a guarantee -- which is why the host asserts the spread
  // below rather than assuming it.
  std::cout << tag << ": launch " << num_cores << " CTAs (M=" << M << " N=" << N
            << " K=" << K << ", tile_n=" << tile_n << ")" << std::endl;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &karg;
  li.args_size    = sizeof(karg);
  li.ndim         = 1;
  li.grid_dim[0]  = (uint32_t)num_cores;
  li.block_dim[0] = 1;
  vx_event_h launch_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  vx_event_h dread_ev = nullptr, cread_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, hD.data(), D_buf, 0, hD.size() * sizeof(otype_t), 1, &launch_ev, &dread_ev));
  RT_CHECK(vx_enqueue_read(queue, &ctl, ctl_buf, 0, sizeof(ctl), 1, &launch_ev, &cread_ev));
  RT_CHECK(vx_event_wait_value(dread_ev, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(cread_ev, 1, VX_TIMEOUT_INFINITE));

  // ---- the assertions this test exists for ----

  // (1) A submitter ran and is identifiable.
  if (ctl.core_of_ticket[0] == 0) {
    FAIL("ticket 0 never ran -- no core submitted the descriptor");
  }
  const uint32_t submitter = ctl.core_of_ticket[0] - 1;

  // (2) THE assertion. At least one core that did NOT submit observed completion. A run
  // where dispatch happened to put every CTA on the submitter's core is a FAILURE, not a
  // pass: it means the property went untested and a regression here would slip through.
  uint32_t xcore = 0, xsocket = 0, stalled = 0;
  for (uint32_t t = 1; t < num_cores && t < XCORE_MAX_CTAS; ++t) {
    if (ctl.core_of_ticket[t] == 0) continue;
    const uint32_t c = ctl.core_of_ticket[t] - 1;
    if (ctl.done_seen[t] & 2u) {
      std::cerr << tag << ": ticket " << t << " on core " << c
                << " hit the spin limit -- completion never became visible" << std::endl;
      ++stalled;
      continue;
    }
    if (!(ctl.done_seen[t] & 1u)) continue;
    if (c != submitter) {
      ++xcore;
      if (c / socket_size != submitter / socket_size) ++xsocket;
    }
  }
  if (stalled) FAIL(stalled << " consumer(s) never saw the completion flag");
  if (xcore == 0)
    FAIL("no cross-core observer: every CTA that saw completion ran on core "
         << submitter << ", so the cross-core path was NOT exercised");
  // (3) With more than one socket, a different-socket observer must also have run. That
  // is the case DTCU_socket's output placement actually stresses: D landed in the
  // submitter's socket dcache, and this consumer is not behind it.
  if (num_cores > socket_size && xsocket == 0)
    FAIL("no different-socket observer ran");

  // (4) Every consumer's witness of D agrees with the host's own read-back. This is what
  // turns "the flag said done" into "the OUTPUT was visible when the flag said done".
  const uint32_t expect = xcore_witness_of(bit_cast<uint32_t>(hD[0]),
                                           bit_cast<uint32_t>(hD[M * N - 1]));
  for (uint32_t t = 1; t < num_cores && t < XCORE_MAX_CTAS; ++t) {
    if (!(ctl.done_seen[t] & 1u)) continue;
    if (ctl.witness[t] != expect)
      FAIL("consumer on core " << (ctl.core_of_ticket[t] - 1)
           << " read a stale or partial D (witness=0x" << std::hex << ctl.witness[t]
           << " expected 0x" << expect << std::dec << ")");
  }

  // (5) And D itself is correct.
  int mismatches = 0;
  for (uint32_t i = 0; i < M * N; ++i) {
    float got = Convert<vt::OTYPE>::to_float(hD[i]);
    if (ulp_diff(got, hRef[i]) > FLOAT_ULP) {
      if (mismatches < MAX_ERRORS)
        std::cerr << tag << ": mismatch D[" << i << "]: got=" << got
                  << " exp=" << hRef[i] << std::endl;
      ++mismatches;
    }
  }
  if (mismatches) FAIL(mismatches << " / " << (M * N) << " D elements wrong");

  std::cout << tag << ": submitter=core" << submitter
            << " observers=" << ctl.observers
            << " cross_core=" << xcore << " cross_socket=" << xsocket << std::endl;

  vx_event_release(dread_ev); vx_event_release(cread_ev); vx_event_release(launch_ev);
  vx_buffer_release(A_buf); vx_buffer_release(B_buf); vx_buffer_release(D_buf);
  vx_buffer_release(desc_buf); vx_buffer_release(ctl_buf);
  vx_kernel_release(kernel); vx_module_release(module_);
  vx_queue_release(queue);
  vx_device_dump_perf(device, stdout);
  vx_device_release(device);

  *errors_out = errors;
  return 0;
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;

  struct { int engine; const char* tag; } kEngines[] = {
    { DTCU_ENGINE_CLUSTER, "dtcu_xcore[cluster]" },
    { DTCU_ENGINE_SOCKET,  "dtcu_xcore[socket]"  },
  };

  int total_errors = 0, ran = 0;
  for (auto& e : kEngines) {
    bool skipped = false;
    int errors = 0;
    int rc = run_engine(e.engine, e.tag, &skipped, &errors);
    if (rc != 0) return rc;
    total_errors += errors;
    if (!skipped) ++ran;
  }

  if (ran == 0) {
    std::cout << "dtcu_xcore: no DTCU engine present in this build" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  if (total_errors) {
    std::cout << "Found " << total_errors << " error(s) across " << ran
              << " engine(s)!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return total_errors;
  }
  std::cout << "dtcu_xcore: " << ran << " engine(s) verified cross-core" << std::endl;
  std::cout << "PASSED!" << std::endl;
  return 0;
}
