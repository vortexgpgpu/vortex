#include "llama_dev.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

#include <VX_types.h>
#include <dtcu_cfg.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>
#include <dxa.h>

// The microbenchmark's own host headers: mode ids and names, the per-mode launch spec,
// the WMMA/WGMMA tile geometry and the shared kernel_arg_t ABI. Included, not copied.
using namespace vortex;   // bit_cast, as cgo27_motivation/main.cpp does before its host headers

#include "common.h"
#include "host_types.h"
#include "host_modes.h"
#include "run_modes.h"

#define LD_CHECK(_expr)                                                    \
  do {                                                                     \
    int _ret = (int)(_expr);                                               \
    if (0 != _ret) {                                                       \
      std::cerr << "llama_dtcu: " << #_expr << " returned " << _ret        \
                << std::endl;                                              \
      return _ret;                                                         \
    }                                                                      \
  } while (false)

static int read_file(const char* path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "llama_dtcu: cannot read " << path << std::endl; return -1; }
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return out.empty() ? -1 : 0;
}

int LlamaDevice::open() {
  LD_CHECK(vx_device_open(0, &dev_));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  LD_CHECK(vx_queue_create(dev_, &qi, &queue_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_NUM_CORES,      &num_cores_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_SOCKET_SIZE,    &socket_size_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_NUM_WARPS,      &num_warps_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_ISSUE_WIDTH,    &issue_width_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_ISA_FLAGS,      &isa_flags_));
  LD_CHECK(vx_dev_caps(dev_, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size_));
  if (socket_size_ == 0) socket_size_ = 1;

  const uint64_t slots = num_cores_ * MOTI_PIPE_TILES;   // mode 14 needs the most
  desc_zeros_.assign((size_t)slots * sizeof(dtensor_desc_t), 0);
  LD_CHECK(vx_buffer_create(dev_, desc_zeros_.size(), VX_MEM_READ_WRITE, &desc_buf_));
  LD_CHECK(vx_buffer_address(desc_buf_, &desc_addr_));
  return 0;
}

void LlamaDevice::close() {
  for (int k = 0; k < kMaxPassKernels; ++k)
    if (active_kern_[k]) { vx_kernel_release(active_kern_[k]); active_kern_[k] = nullptr; }
  if (active_mod_) { vx_module_release(active_mod_); active_mod_ = nullptr; }
  active_slot_ = -1;
  if (desc_buf_) { vx_buffer_release(desc_buf_); desc_buf_ = nullptr; }
  if (queue_)    { vx_queue_release(queue_);     queue_    = nullptr; }
  if (dev_)      { vx_device_release(dev_);      dev_      = nullptr; }
}

int LlamaDevice::alloc(uint64_t bytes, int flags, vx_buffer_h* buf, uint64_t* addr) {
  LD_CHECK(vx_buffer_create(dev_, bytes, flags, buf));
  LD_CHECK(vx_buffer_address(*buf, addr));
  return 0;
}

int LlamaDevice::upload(vx_buffer_h buf, uint64_t off, const void* src, uint64_t bytes) {
  vx_event_h ev = nullptr;
  LD_CHECK(vx_enqueue_write(queue_, buf, off, src, bytes, 0, nullptr, &ev));
  LD_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  return 0;
}

int LlamaDevice::download(void* dst, vx_buffer_h buf, uint64_t off, uint64_t bytes) {
  vx_event_h ev = nullptr;
  LD_CHECK(vx_enqueue_read(queue_, dst, buf, off, bytes, 0, nullptr, &ev));
  LD_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  return 0;
}

int LlamaDevice::register_mode(uint32_t mode) {
  if (mode >= 16 || mode_state(mode) != ModeState::Implemented) return -1;
  if (!prog_bytes_[mode].empty()) return 0;
  const ModeSpec spec = moti_mode_spec(mode);
  if (spec.kentry == nullptr) return -1;
  const char* dir = std::getenv("LLAMA_VXBIN_DIR");
  char path[512];
  std::snprintf(path, sizeof(path), "%s/kernel_m%u.vxbin", dir ? dir : ".", mode);
  LD_CHECK(read_file(path, prog_bytes_[mode]));
  prog_entries_[mode] = { spec.kentry };
  return 0;
}

int LlamaDevice::register_passes(const char* filename, const std::vector<std::string>& entries) {
  if ((int)entries.size() > kMaxPassKernels) return -1;
  const char* dir = std::getenv("LLAMA_VXBIN_DIR");
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s", dir ? dir : ".", filename);
  LD_CHECK(read_file(path, prog_bytes_[kPassSlot]));
  prog_entries_[kPassSlot] = entries;
  return 0;
}

// Make `slot`'s program the resident one. Host time only; nothing here is measured.
int LlamaDevice::activate(int slot) {
  if (active_slot_ == slot) return 0;
  if (prog_bytes_[slot].empty()) return -1;
  for (int k = 0; k < kMaxPassKernels; ++k)
    if (active_kern_[k]) { vx_kernel_release(active_kern_[k]); active_kern_[k] = nullptr; }
  if (active_mod_) { vx_module_release(active_mod_); active_mod_ = nullptr; }
  active_slot_ = -1;
  LD_CHECK(vx_module_load_bytes(dev_, prog_bytes_[slot].data(), prog_bytes_[slot].size(), &active_mod_));
  for (size_t k = 0; k < prog_entries_[slot].size(); ++k)
    LD_CHECK(vx_module_get_kernel(active_mod_, prog_entries_[slot][k].c_str(), &active_kern_[k]));
  active_slot_ = slot;
  ++swaps_;
  return 0;
}

bool LlamaDevice::mode_runnable(uint32_t mode) const {
  if (mode >= 16 || mode_state(mode) != ModeState::Implemented) return false;
  const ModeSpec spec = moti_mode_spec(mode);
  return spec.kentry != nullptr && (isa_flags_ & spec.isa_need) == spec.isa_need;
}

std::string LlamaDevice::mode_illegal_reason(uint32_t mode, uint32_t M, uint32_t N,
                                             uint32_t K) const {
  char buf[256];
  if (!mode_runnable(mode)) return "mode not runnable on this device";
  const ModeSpec spec = moti_mode_spec(mode);
  const uint32_t tileM = cfg::tileM, tileN = cfg::tileN;
  const uint32_t tileK = cfg::tileK * (4 / sizeof(itype_t));
  const uint32_t cta_M = (uint32_t)issue_width_ * wgcfg::xtileM;
  const uint32_t wg_stK = MOTI_WG_KSTEPS * wgcfg::tileK;
  auto mult = [&](uint32_t v, uint32_t m, const char* dim) -> bool {
    if (v % m) {
      std::snprintf(buf, sizeof(buf), "%s=%u is not a multiple of %u (mode %u)", dim, v, m, mode);
      return false;
    }
    return true;
  };
  switch (spec.geom) {
  case ModeSpec::GEOM_SIMT:
    if (!mult(N, NUM_THREADS, "N")) return buf;
    break;
  case ModeSpec::GEOM_WMMA:
    if (!mult(M, tileM, "M") || !mult(N, tileN, "N") || !mult(K, tileK, "K")) return buf;
    break;
  case ModeSpec::GEOM_WMMA_WG:
  case ModeSpec::GEOM_WMMA_WG_ACOL: {
    if (!mult(K, wg_stK, "K") || !mult(M, cta_M, "M") || !mult(N, wgcfg::xtileN, "N")) return buf;
    if (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL) {
      if (!mult(N, MOTI_WG_NCOLS * wgcfg::xtileN, "N")) return buf;
      const uint64_t lmem = (uint64_t)(cta_M * K + spec.lmem_stages * wgcfg::xtileN * wg_stK) * sizeof(itype_t);
      if (lmem_size_ && lmem > lmem_size_) {
        std::snprintf(buf, sizeof(buf), "Local Memory %llu B exceeds %llu B (mode %u)",
                      (unsigned long long)lmem, (unsigned long long)lmem_size_, mode);
        return buf;
      }
    }
    break;
  }
  case ModeSpec::GEOM_PER_CORE:
  case ModeSpec::GEOM_PER_CORE_PIPE:
    if (M > 0xFFFFu || N > 0xFFFFu || K > 0xFFFFu) return "M/N/K exceed the descriptor's uint16 fields";
    if (M == 0 || N == 0 || K == 0) return "empty GEMM";
    break;
  }
  return "";
}

int LlamaDevice::read_cycles(LaunchStats* st) {
  uint64_t cyc = 0, ins = 0;
  LD_CHECK(vx_mpm_query(dev_, VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE,   0, &cyc));
  LD_CHECK(vx_mpm_query(dev_, VX_DCR_MPM_CLASS_BASE, VX_CSR_MINSTRET, 0, &ins));
  st->cycles += cyc;
  st->instrs += ins;
  st->launches += 1;
  return 0;
}

int LlamaDevice::launch_and_wait(vx_launch_info_t& li, LaunchStats* st) {
  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h ev = nullptr;
  LD_CHECK(vx_enqueue_launch(queue_, &li, 0, nullptr, &ev));
  LD_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  auto t1 = std::chrono::high_resolution_clock::now();
  vx_event_release(ev);
  // MCYCLE is reset by every launch, so it is read here, before anything else is launched.
  LD_CHECK(read_cycles(st));
  st->host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  return 0;
}

int LlamaDevice::launch_pass(int idx, const void* arg, uint32_t arg_size, uint32_t blocks_x,
                             uint32_t blocks_y, uint32_t lmem, LaunchStats* st) {
  LD_CHECK(activate(kPassSlot));
  if (idx < 0 || idx >= (int)prog_entries_[kPassSlot].size()) return -1;
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = active_kern_[idx]; li.args_host = const_cast<void*>(arg); li.args_size = arg_size;
  li.ndim = (blocks_y > 1) ? 2 : 1;
  li.grid_dim[0] = blocks_x; li.grid_dim[1] = blocks_y;
  li.block_dim[0] = NUM_THREADS; li.block_dim[1] = 1;
  li.lmem_size = lmem;
  return launch_and_wait(li, st);
}

int LlamaDevice::gemm(uint32_t mode, uint32_t M, uint32_t N, uint32_t K,
                      uint64_t A_addr, uint64_t B_addr, uint64_t C_addr, uint64_t D_addr,
                      LaunchStats* st) {
  const ModeSpec spec = moti_mode_spec(mode);
  if (spec.kentry == nullptr) return -1;
  LD_CHECK(register_mode(mode));
  LD_CHECK(activate((int)mode));

  kernel_arg_t karg{};
  karg.mode = mode; karg.app = MOTI_APP;
  karg.M = M; karg.N = N; karg.K = K;
  karg.A_addr = A_addr; karg.B_addr = B_addr; karg.C_addr = C_addr; karg.D_addr = D_addr;
  karg.desc_addr = uses_engine(mode) ? desc_addr_ : 0;

  const uint32_t tcu_tileM = cfg::tileM, tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * (4 / sizeof(itype_t));
  const uint32_t wg_warps = (uint32_t)issue_width_;
  const uint32_t cta_M    = wg_warps * wgcfg::xtileM;
  const uint32_t wg_stK   = MOTI_WG_KSTEPS * wgcfg::tileK;

  // Engine modes: the kernel builds its descriptors in this array; slots whose slice is
  // empty are never written, so the array is re-zeroed before every launch (in-order on
  // the queue, ahead of the launch; host-side, no measured cycles).
  if (uses_engine(mode)) {
    LD_CHECK(vx_enqueue_write(queue_, desc_buf_, 0, desc_zeros_.data(), desc_zeros_.size(),
                              0, nullptr, nullptr));
  }

  // DXA-fed modes: the copy engine's 2D descriptors are device-global registers and carry
  // the operand addresses, so they are reprogrammed for every GEMM.
  if (spec.dxa_desc) {
    const bool wg   = (spec.geom == ModeSpec::GEOM_WMMA_WG) || (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL);
    const bool acol = (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL);
    const uint32_t dK = wg ? wg_stK        : tcu_tileK;
    const uint32_t dM = wg ? cta_M         : tcu_tileM;
    const uint32_t dN = wg ? wgcfg::xtileN : tcu_tileN;
    LD_CHECK(vortex::dxa::program_2d(dev_, DESC_A, A_addr,
      /*size0=*/K, /*size1=*/M, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/(acol ? K : dK), /*tile1=*/dM, /*elem_bytes=*/sizeof(itype_t)));
    LD_CHECK(vortex::dxa::program_2d(dev_, DESC_B, B_addr,
      /*size0=*/K, /*size1=*/N, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/dK, /*tile1=*/dN, /*elem_bytes=*/sizeof(itype_t)));
  }

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = active_kern_[0]; li.args_host = &karg; li.args_size = sizeof(karg);
  switch (spec.geom) {
  case ModeSpec::GEOM_SIMT:
    li.ndim = 2;
    li.grid_dim[0]  = N / NUM_THREADS; li.grid_dim[1]  = M;
    li.block_dim[0] = NUM_THREADS;     li.block_dim[1] = 1;
    break;
  case ModeSpec::GEOM_WMMA: {
    li.ndim = 2;
    li.grid_dim[0]  = N / tcu_tileN; li.grid_dim[1]  = M / tcu_tileM;
    li.block_dim[0] = NUM_THREADS;   li.block_dim[1] = 1;
    const uint32_t stage_bytes = (tcu_tileM * tcu_tileK + tcu_tileN * tcu_tileK) * sizeof(itype_t);
    li.lmem_size = spec.lmem_stages * stage_bytes;
    break;
  }
  case ModeSpec::GEOM_WMMA_WG:
    li.ndim = 2;
    li.grid_dim[0]  = N / wgcfg::xtileN;      li.grid_dim[1]  = M / cta_M;
    li.block_dim[0] = wg_warps * NUM_THREADS; li.block_dim[1] = 1;
    li.lmem_size = spec.lmem_stages * (cta_M + wgcfg::xtileN) * wg_stK * sizeof(itype_t);
    break;
  case ModeSpec::GEOM_WMMA_WG_ACOL:
    li.ndim = 2;
    li.grid_dim[0]  = N / (MOTI_WG_NCOLS * wgcfg::xtileN);
    li.grid_dim[1]  = M / cta_M;
    li.block_dim[0] = wg_warps * NUM_THREADS; li.block_dim[1] = 1;
    li.lmem_size = (cta_M * K + spec.lmem_stages * wgcfg::xtileN * wg_stK) * sizeof(itype_t);
    break;
  case ModeSpec::GEOM_PER_CORE:
    li.ndim = 1;
    li.grid_dim[0] = (uint32_t)num_cores_; li.block_dim[0] = 1;
    break;
  case ModeSpec::GEOM_PER_CORE_PIPE:
    li.ndim = 1;
    li.grid_dim[0]  = (uint32_t)num_cores_;
    li.block_dim[0] = (uint32_t)num_warps_ * NUM_THREADS;
    break;
  }
  return launch_and_wait(li, st);
}
