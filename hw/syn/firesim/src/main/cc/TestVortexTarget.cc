// See LICENSE for license details.

// Bring-up harness for the Vortex FireSim target.
//
// Exercises the control surface only: reset, the DCR request/response path, and
// start/busy. Running an actual kernel additionally needs a program image in
// target DRAM, which the runtime transport supplies.

#include "TestHarness.h"

#include "bridges/loadmem.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

// The launch the runtime performs, reduced to the parts the target can see.
//
// Captured from a traced `demo` run so it is the real sequence, not a guess:
// program image at 0x80000000, entry 0x24 into it, arguments at 0x13000, and a
// 4x1 grid of 4x1 blocks. Replaying it here removes the runtime, the command
// processor and a ~16 minute upload from the picture, which is what makes the
// question "does the core fetch at all" answerable in seconds instead of an
// afternoon.
struct dcr_write_t {
  uint32_t addr;
  uint32_t data;
};

constexpr dcr_write_t kLaunchDCRs[] = {
    {0x10, 0x80000000}, {0x11, 0x0}, {0x12, 0x80000024}, {0x13, 0x0},
    {0x14, 0x00013000}, {0x15, 0x0}, {0x16, 0x4},        {0x17, 0x1},
    {0x18, 0x1},        {0x19, 0x4}, {0x1a, 0x1},        {0x1b, 0x1},
    {0x1c, 0x0},        {0x1d, 0x4}, {0x1e, 0x0},        {0x1f, 0x0},
    {0x20, 0x0},        {0x21, 0x1}, {0x22, 0x1},        {0x23, 0x1},
};

} // namespace

class TestVortexTarget final : public TestHarness {
public:
  using TestHarness::TestHarness;

  void run_test() override {
    // Markers on stderr, which is unbuffered. The harness's own logging goes to
    // stdout, and stdout is block-buffered when redirected to a file, so a run
    // that is merely slow looks identical to one that is wedged until it exits.
    // That cost 45 minutes of watching an empty log.
    mark("run_test entered");

    // Vortex samples reset for VX_CFG_RESET_DELAY cycles before it will accept
    // any request, so hold it well past that.
    poke("ctrl_start", 0);
    poke("ctrl_dcr_req_valid", 0);
    mark("initial pokes done");
    target_reset(32);
    mark("reset released");

    // Out of reset and idle: nothing has been started, so the core must not
    // claim to be busy.
    expect(std::string_view("ctrl_busy"), uint32_t(0));

    // A DCR write is the smallest transaction that proves the control path is
    // live end to end. It is accepted combinationally, so one step suffices.
    poke("ctrl_dcr_req_valid", 1);
    poke("ctrl_dcr_req_rw", 1);
    poke("ctrl_dcr_req_addr", 0x1);
    poke("ctrl_dcr_req_data", 0xdeadbeef);
    step(1);
    poke("ctrl_dcr_req_valid", 0);
    step(1);

    mark("control-path checks done");

    // Still idle: a DCR write configures the device, it does not launch work.
    expect(std::string_view("ctrl_busy"), uint32_t(0));

    // Advance far enough for the reported rate to reflect steady state rather
    // than start-up. The target is idle here, so this measures how fast the
    // simulator runs when nothing is contending for the memory model.
    const char *soak = getenv("VORTEX_SOAK_CYCLES");
    if (soak != nullptr) {
      const uint32_t n = static_cast<uint32_t>(strtoul(soak, nullptr, 0));
      if (n > 0) {
        step(n);
      }
    }

    // Regions load before the image so that the image, which carries the load
    // address in its own header, always wins if the two overlap.
    const char *regions = getenv("VORTEX_KERNEL_REGIONS");
    if (regions != nullptr) {
      load_regions(regions);
    }

    const char *kernel = getenv("VORTEX_KERNEL_BIN");
    if (kernel != nullptr) {
      run_kernel(kernel);
    }
  }

  // Reads one Vortex performance counter over the DCR response path.
  //
  // The point of having this here as well as in the driver is the differential:
  // this harness completes the same kernel the driver hangs on, so comparing the
  // two counter dumps says where the two runs diverge inside the core, which no
  // amount of bus-level observation can.
  //
  // Requires RTL built with PERF_ENABLE; without it every read returns zero, and
  // zero is indistinguishable from a genuinely idle core -- so an all-zero dump
  // is reported as "instrument absent", not as a measurement.
  uint32_t read_mpm(uint32_t mpm_class, uint32_t csr_addr, uint32_t core_id) {
    constexpr uint32_t kMpmBase = 0xB00;     // VX_CSR_MPM_BASE
    constexpr uint32_t kDcrMpmValue = 0x001; // VX_DCR_BASE_MPM_VALUE
    const uint32_t tag =
        (mpm_class << 22) | ((csr_addr - kMpmBase) << 16) | core_id;
    poke("ctrl_dcr_req_valid", 1);
    poke("ctrl_dcr_req_rw", 0);
    poke("ctrl_dcr_req_addr", kDcrMpmValue);
    poke("ctrl_dcr_req_data", tag);
    step(1);
    poke("ctrl_dcr_req_valid", 0);
    for (int i = 0; i < 64; ++i) {
      step(1);
      if (peek("ctrl_dcr_rsp_valid") != 0) {
        return peek("ctrl_dcr_rsp_data");
      }
    }
    return 0;
  }

  void report_core_counters() {
    struct counter_t {
      const char *name;
      uint32_t csr;
    };
    static constexpr counter_t kCore[] = {
        {"sched_idle", 0xB03},   {"active_warps", 0xB04},
        {"stalled_warps", 0xB05}, {"issued_warps", 0xB06},
        {"stall_fetch", 0xB08},  {"stall_ibuf", 0xB09},
        {"stall_scrb", 0xB0A},   {"stall_opds", 0xB0B},
        {"stall_alu", 0xB0C},    {"stall_lsu", 0xB0E},
        {"stall_sfu", 0xB0F},   {"stall_tcu", 0xB10},
    };
    bool all_zero = true;
    fprintf(stderr, "[mpm] core counters:");
    for (const auto &c : kCore) {
      const uint32_t v = read_mpm(1 /* VX_DCR_MPM_CLASS_CORE */, c.csr, 0);
      all_zero = all_zero && (v == 0);
      fprintf(stderr, " %s=%u", c.name, v);
    }
    fprintf(stderr, "%s\n",
            all_zero ? "  (all zero -- was the RTL built with PERF_ENABLE?)" : "");
  }

  bool read_file(const char *path, std::vector<uint8_t> &out) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
      return false;
    }
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
      fclose(f);
      return false;
    }
    out.resize(static_cast<size_t>(len));
    const size_t got = fread(out.data(), 1, out.size(), f);
    fclose(f);
    return got == out.size();
  }

  void write_region(uint64_t addr, const std::vector<uint8_t> &data) {
    auto &loadmem = get_bridge<loadmem_t>();
    const size_t chunk = loadmem.get_mem_data_chunk() * sizeof(uint32_t);
    // Refuse rather than silently mis-place. An unaligned replay would inject a
    // corruption the driver never had, and the resulting failure would look
    // exactly like the defect under investigation.
    if ((addr % chunk) != 0 || (data.size() % chunk) != 0) {
      fprintf(stderr,
              "[kernel] region 0x%llx size %zu is not beat-aligned (%zu)\n",
              (unsigned long long)addr, data.size(), chunk);
      pass = false;
      return;
    }
    mpz_t word;
    mpz_init(word);
    for (size_t off = 0; off < data.size(); off += chunk) {
      mpz_import(word, chunk, -1, 1, 0, 0, data.data() + off);
      loadmem.write_mem(addr + off, word);
    }
    mpz_clear(word);
  }

  // Dumps a region of target memory once the kernel has finished, so the run
  // can be checked against the answer the host would have computed. Without
  // this the harness proves the kernel *completes*, which is a weaker claim
  // than the one under investigation: hw_emu already completes and returns
  // wrong data. The comparison is deliberately left outside the harness --
  // dumping bytes keeps it generic, and the reference is regenerated from the
  // test's own `srand(50)` sequence rather than restated here.
  //
  // VORTEX_KERNEL_READBACK=0x<addr>:<bytes>:<path>
  // The kernel's stores sit in the dcache until something writes them back. The
  // driver gets that for free -- its cache-flush DCR read is what produced
  // `aw_count 0 -> 64` on hw_emu -- so a readback taken without it would report
  // memory the results never reached and read as a total corruption. Address 0
  // is VX_DCR_BASE_STATE, whose read the flush gate answers.
  void flush_cache() {
    poke("ctrl_dcr_req_valid", 1);
    poke("ctrl_dcr_req_rw", 0);
    poke("ctrl_dcr_req_addr", 0x0);
    poke("ctrl_dcr_req_data", 0);
    step(1);
    poke("ctrl_dcr_req_valid", 0);
    for (int i = 0; i < 100000; ++i) {
      step(1);
      if (peek("ctrl_dcr_rsp_valid") != 0) {
        fprintf(stderr, "[kernel] cache flush acknowledged after %d cycles\n", i);
        return;
      }
    }
    fprintf(stderr, "[kernel] cache flush never acknowledged -- readback would "
                    "describe memory the stores never reached\n");
    pass = false;
  }

  void dump_readback() {
    const char *spec = getenv("VORTEX_KERNEL_READBACK");
    if (spec == nullptr) {
      return;
    }
    flush_cache();
    unsigned long long addr = 0;
    unsigned long long bytes = 0;
    char path[512] = {0};
    if (sscanf(spec, "0x%llx:%llu:%511s", &addr, &bytes, path) != 3) {
      fprintf(stderr, "[kernel] cannot parse VORTEX_KERNEL_READBACK=%s\n", spec);
      pass = false;
      return;
    }
    auto &loadmem = get_bridge<loadmem_t>();
    const size_t chunk = loadmem.get_mem_data_chunk() * sizeof(uint32_t);
    if ((addr % chunk) != 0 || (bytes % chunk) != 0) {
      fprintf(stderr, "[kernel] readback 0x%llx/%llu is not beat-aligned (%zu)\n",
              addr, bytes, chunk);
      pass = false;
      return;
    }
    std::vector<uint8_t> out(static_cast<size_t>(bytes), 0);
    mpz_t word;
    mpz_init(word);
    for (size_t off = 0; off < out.size(); off += chunk) {
      loadmem.read_mem(addr + off, word);
      std::vector<uint8_t> beat(chunk, 0);
      size_t count = 0;
      mpz_export(beat.data(), &count, -1, 1, 0, 0, word);
      std::memcpy(out.data() + off, beat.data(), chunk);
    }
    mpz_clear(word);
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
      fprintf(stderr, "[kernel] cannot write %s\n", path);
      pass = false;
      return;
    }
    const size_t put = fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    fprintf(stderr, "[kernel] readback 0x%llx %zu bytes -> %s\n", addr, put, path);
  }

  // Replays the uploads a real driver run performed, so the kernel executes on
  // the data it was actually given. Loading only the program image leaves the
  // argument struct and both input buffers as zeros, and the kernel then exits
  // in a few hundred cycles -- which reproduces the launch but not the
  // workload, and so cannot reproduce a stall that needs real data to reach.
  //
  // Files come from VORTEX_FIRESIM_DUMP_UPLOADS on the driver side and are
  // named upload_<seq>_0x<addr>_<size>.bin; the address and length are in the
  // name so the two sides cannot disagree about where a region belongs.
  void load_regions(const char *dir) {
    std::vector<std::string> names;
    DIR *d = opendir(dir);
    if (d == nullptr) {
      fprintf(stderr, "[kernel] cannot open region dir %s\n", dir);
      pass = false;
      return;
    }
    while (const dirent *e = readdir(d)) {
      const std::string name(e->d_name);
      if (name.rfind("upload_", 0) == 0) {
        names.push_back(name);
      }
    }
    closedir(d);
    // Sequence order matters: a later upload may legitimately overwrite an
    // earlier one, and the driver's order is the only correct one.
    std::sort(names.begin(), names.end());

    for (const auto &name : names) {
      unsigned long long addr = 0;
      unsigned long long size = 0;
      const char *p = std::strchr(name.c_str(), '_');
      p = (p != nullptr) ? std::strchr(p + 1, '_') : nullptr;
      if (p == nullptr || sscanf(p, "_0x%llx_%llu.bin", &addr, &size) != 2) {
        fprintf(stderr, "[kernel] cannot parse region name %s\n", name.c_str());
        pass = false;
        return;
      }
      const std::string path = std::string(dir) + "/" + name;
      std::vector<uint8_t> data;
      if (!read_file(path.c_str(), data) || data.size() != size) {
        fprintf(stderr, "[kernel] region %s did not read back as %llu bytes\n",
                path.c_str(), size);
        pass = false;
        return;
      }
      write_region(addr, data);
      fprintf(stderr, "[kernel] region 0x%llx <- %llu bytes (%s)\n", addr, size,
              name.c_str());
    }
  }

private:

  static void mark(const char *what) {
    fprintf(stderr, "[vortex-harness] %s\n", what);
    fflush(stderr);
  }

  // Loads a program image into target DRAM and launches it, reporting whether
  // the core ever fetches. Deliberately does not check a result: the question
  // this answers is whether the target's own read path works at all, and a
  // wrong answer is far more informative than no answer.
  void run_kernel(const char *path) {
    auto &loadmem = get_bridge<loadmem_t>();

    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
      fprintf(stderr, "[kernel] cannot open %s\n", path);
      pass = false;
      return;
    }
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> image(static_cast<size_t>(len), 0);
    if (fread(image.data(), 1, static_cast<size_t>(len), f) != static_cast<size_t>(len)) {
      fprintf(stderr, "[kernel] short read on %s\n", path);
      fclose(f);
      pass = false;
      return;
    }
    fclose(f);

    // A .vxbin is not a raw image: 16 bytes of header (min_vma, max_vma), then
    // the program, then an optional VXSYMTAB footer. Loading the file verbatim
    // shifts every instruction by 16 bytes and the core executes rubbish --
    // which looks exactly like the defect under investigation. The load address
    // and length both come from the header, so they cannot disagree with what
    // the runtime does.
    if (image.size() < 16) {
      fprintf(stderr, "[kernel] %s is too small to be a .vxbin\n", path);
      pass = false;
      return;
    }
    uint64_t min_vma = 0;
    uint64_t max_vma = 0;
    std::memcpy(&min_vma, image.data() + 0, 8);
    std::memcpy(&max_vma, image.data() + 8, 8);
    if (max_vma <= min_vma || (max_vma - min_vma) > image.size() - 16) {
      fprintf(stderr, "[kernel] bad .vxbin header: min=0x%llx max=0x%llx\n",
              (unsigned long long)min_vma, (unsigned long long)max_vma);
      pass = false;
      return;
    }
    const size_t payload = static_cast<size_t>(max_vma - min_vma);
    image.erase(image.begin(), image.begin() + 16);
    image.resize(payload);

    // The widget's chunk counts 32-bit words, not bytes. Using it as a byte
    // count uploads two live bytes per eight-byte beat and steps the address
    // by two, so the core fetches an instruction stream that is mostly zeros.
    const size_t chunk = loadmem.get_mem_data_chunk() * sizeof(uint32_t);
    image.resize(((payload + chunk - 1) / chunk) * chunk, 0);
    fprintf(stderr, "[kernel] file %ld bytes -> loading %zu bytes at 0x%llx\n",
            len, payload, (unsigned long long)min_vma);

    // Fault injection for the round-trip check below. A check is worth nothing
    // until it has been shown to fail: the single-beat version of this check
    // printed OK on every run of the defect it was supposed to catch. Setting
    // VORTEX_KERNEL_INJECT_STRIDE_BUG reproduces that defect exactly -- the
    // chunk taken as bytes rather than words -- so the check can be watched
    // going red before any green result from it is believed.
    const bool inject = (getenv("VORTEX_KERNEL_INJECT_STRIDE_BUG") != nullptr);
    const size_t upload_stride = inject ? loadmem.get_mem_data_chunk() : chunk;
    if (inject) {
      fprintf(stderr, "[kernel] INJECTING stride bug: %zu-byte stride\n",
              upload_stride);
    }

    mpz_t word;
    mpz_init(word);
    for (size_t off = 0; off < image.size(); off += upload_stride) {
      mpz_import(word, upload_stride, -1, 1, 0, 0, image.data() + off);
      loadmem.write_mem(min_vma + off, word);
    }

    // Read back a whole cache line, not one beat. A single-beat check passes
    // even when every beat after the first is misplaced, because the first
    // beat is the one case where a wrong stride still lands correctly. It says
    // nothing about what the target sees either, which is the distinction that
    // made this bug hard to find.
    const size_t verify_bytes = std::min<size_t>(64, image.size());
    std::vector<uint8_t> check(verify_bytes, 0);
    for (size_t off = 0; off < verify_bytes; off += chunk) {
      loadmem.read_mem(min_vma + off, word);
      std::vector<uint8_t> beat(chunk, 0);
      size_t count = 0;
      mpz_export(beat.data(), &count, -1, 1, 0, 0, word);
      std::memcpy(check.data() + off, beat.data(),
                  std::min(chunk, verify_bytes - off));
    }
    const bool image_ok =
        (std::memcmp(check.data(), image.data(), verify_bytes) == 0);
    fprintf(stderr, "[kernel] loadmem round-trip of first %zu bytes: %s\n",
            verify_bytes, image_ok ? "OK" : "MISMATCH");
    mpz_clear(word);

    for (const auto &d : kLaunchDCRs) {
      poke("ctrl_dcr_req_valid", 1);
      poke("ctrl_dcr_req_rw", 1);
      poke("ctrl_dcr_req_addr", d.addr);
      poke("ctrl_dcr_req_data", d.data);
      step(1);
      poke("ctrl_dcr_req_valid", 0);
      step(1);
    }

    poke("ctrl_start", 1);
    step(1);
    poke("ctrl_start", 0);
    step(1);

    // Poll in slices so the cycle at which `busy` rises and falls is visible,
    // rather than only whether it happened to be set at one sampling point.
    const uint32_t slice = 256;
    uint32_t max_cycles = 200000;
    if (const char *env = getenv("VORTEX_KERNEL_MAX_CYCLES")) {
      const unsigned long v = strtoul(env, nullptr, 0);
      if (v > 0) {
        max_cycles = static_cast<uint32_t>(v);
      }
    }
    bool saw_busy = false;
    uint32_t elapsed = 0;
    while (elapsed < max_cycles) {
      step(slice);
      elapsed += slice;
      const uint32_t busy = peek("ctrl_busy");
      if (busy != 0 && !saw_busy) {
        saw_busy = true;
        fprintf(stderr, "[kernel] busy asserted by cycle %u\n", elapsed);
      }
      if (saw_busy && busy == 0) {
        fprintf(stderr, "[kernel] COMPLETED after ~%u cycles\n", elapsed);
        report_core_counters();
        dump_readback();
        return;
      }
    }
    // Counters before the AXI dump: on a stall this is the one that localizes
    // the wedge, and the run may be killed before a long dump finishes.
    report_core_counters();
    fprintf(stderr, "[kernel] DID NOT COMPLETE in %u cycles (busy=%u, ever busy=%d)\n",
            elapsed, peek("ctrl_busy"), saw_busy ? 1 : 0);
    pass = false;
  }
};

TEST_MAIN(TestVortexTarget)
