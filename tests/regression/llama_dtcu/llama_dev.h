#ifndef _LLAMA_DTCU_DEV_H_
#define _LLAMA_DTCU_DEV_H_

// The device side of llama_dtcu: one Vortex device, one queue, the microbenchmark's
// per-mode GEMM programs plus the llama pass program, and blocking launches that read the
// launch's cycle count.
//
// Everything mode-specific (kernel entry name, launch geometry, DXA descriptors, DTCU
// descriptor slots) follows tests/regression/cgo27_motivation/host/host_run.h exactly, so
// a GEMM launched from here and the same GEMM launched by the microbenchmark are the same
// launch.
//
// ONE PROGRAM RESIDENT AT A TIME. Every device program is linked at the same fixed address
// (STARTUP_ADDR), which is what gives the microbenchmark its property that every mode's
// code starts at 0x180000034 regardless of what else exists. The loader therefore refuses a
// second program while one is loaded. This class keeps the program bytes in host memory
// and swaps the resident program (release + load from bytes, host time only) whenever the
// next launch needs a different one -- so each mode runs at the microbenchmark's address.

#include <cstdint>
#include <string>
#include <vector>

#include <vortex.h>

struct LaunchStats {
  uint64_t cycles = 0;     // MCYCLE summed over the launches of this call
  uint64_t instrs = 0;     // MINSTRET, core 0, summed
  uint32_t launches = 0;
  double   host_ms = 0.0;  // wall time of enqueue -> event, simulator time included
};

class LlamaDevice {
 public:
  static const int kPassSlot = 16;   // slots 0..15 are modes, 16 is the pass program
  static const int kMaxPassKernels = 8;

  ~LlamaDevice() { close(); }

  int  open();
  void close();

  // Device memory. upload()/download() block until the copy is done.
  int alloc(uint64_t bytes, int flags, vx_buffer_h* buf, uint64_t* addr);
  int upload(vx_buffer_h buf, uint64_t off, const void* src, uint64_t bytes);
  int download(void* dst, vx_buffer_h buf, uint64_t off, uint64_t bytes);

  // Programs. Mode programs are kernel_m<mode>.vxbin in LLAMA_VXBIN_DIR (default ".") and
  // register lazily; the pass program must be registered once with its entry names.
  int  register_passes(const char* filename, const std::vector<std::string>& entries);
  bool mode_runnable(uint32_t mode) const;   // implemented and its ISA extension present
  // "" when the (mode, M, N, K) launch is legal; otherwise why not (tile divisibility,
  // Local Memory footprint, descriptor field width). Same rules as the microbenchmark.
  std::string mode_illegal_reason(uint32_t mode, uint32_t M, uint32_t N, uint32_t K) const;

  // D = C + A*B.  A: fp16 row-major [M x K].  B: fp16 col-major [K x N] (== a row-major
  // weight matrix W[N x K]).  C, D: fp32 row-major [M x N].  Blocking.
  int gemm(uint32_t mode, uint32_t M, uint32_t N, uint32_t K,
           uint64_t A_addr, uint64_t B_addr, uint64_t C_addr, uint64_t D_addr,
           LaunchStats* st);

  // One pass kernel (index into the registered entry list), a blocks_x x blocks_y grid of
  // one-warp blocks.
  int launch_pass(int idx, const void* arg, uint32_t arg_size, uint32_t blocks_x, uint32_t blocks_y,
                  uint32_t lmem, LaunchStats* st);

  vx_device_h device() const { return dev_; }
  vx_queue_h  queue()  const { return queue_; }
  uint64_t num_cores()   const { return num_cores_; }
  uint64_t socket_size() const { return socket_size_; }
  uint64_t num_warps()   const { return num_warps_; }
  uint64_t issue_width() const { return issue_width_; }
  uint64_t lmem_size()   const { return lmem_size_; }
  uint32_t program_swaps() const { return swaps_; }

 private:
  int read_cycles(LaunchStats* st);
  int register_mode(uint32_t mode);
  int activate(int slot);
  int launch_and_wait(vx_launch_info_t& li, LaunchStats* st);

  vx_device_h dev_   = nullptr;
  vx_queue_h  queue_ = nullptr;
  uint64_t num_cores_ = 0, socket_size_ = 1, num_warps_ = 0, issue_width_ = 1;
  uint64_t isa_flags_ = 0, lmem_size_ = 0;

  std::vector<uint8_t>     prog_bytes_[17];
  std::vector<std::string> prog_entries_[17];
  int          active_slot_ = -1;
  vx_module_h  active_mod_  = nullptr;
  vx_kernel_h  active_kern_[kMaxPassKernels] = {};
  uint32_t     swaps_ = 0;

  // DTCU descriptor array, sized for the largest submitter count any engine mode needs
  // (num_cores * MOTI_PIPE_TILES) and re-zeroed before every engine launch, because a
  // zeroed descriptor is a valid, instantly-done one to the engine.
  vx_buffer_h desc_buf_  = nullptr;
  uint64_t    desc_addr_ = 0;
  std::vector<uint8_t> desc_zeros_;
};

#endif // _LLAMA_DTCU_DEV_H_
