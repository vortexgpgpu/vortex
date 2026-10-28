#include <iostream>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <vortex2.h>
#include "common.h"

// A sparse reservation (VX_MEM_SPARSE) bound page by page onto two memory
// buffers. Unbound pages must read 0 and discard stores and atomics, bound
// pages must reach exactly the memory page behind them, and a rebind must be
// seen by the next launch and by the command processor's copies.

#define PAGE_SIZE      4096u
#define WORDS_PER_PAGE (PAGE_SIZE / sizeof(uint32_t))
#define MEM_PAGES      (SPARSE_PAGES / 2)
#define NUM_THREADS    (SPARSE_PAGES * SLOTS)

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h sparse_buffer = nullptr;
vx_buffer_h mem_buffer[2] = { nullptr, nullptr };
vx_buffer_h read_buffer = nullptr;
vx_buffer_h amo_buffer = nullptr;

void cleanup() {
  if (device) {
    for (auto b : { sparse_buffer, mem_buffer[0], mem_buffer[1], read_buffer, amo_buffer }) {
      if (b) {
        vx_buffer_release(b);
      }
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void wait(vx_event_h ev) {
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

static void write_buffer(vx_buffer_h buf, const std::vector<uint32_t>& data) {
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_write(queue, buf, 0, data.data(), data.size() * sizeof(uint32_t), 0, nullptr, &ev));
  wait(ev);
}

static std::vector<uint32_t> read_buffer_words(vx_buffer_h buf, uint32_t words) {
  std::vector<uint32_t> data(words);
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, data.data(), buf, 0, words * sizeof(uint32_t), 0, nullptr, &ev));
  wait(ev);
  return data;
}

// The binding of a phase: mem[p] = which memory buffer page p of the
// reservation maps (-1: unbound), at memory page page_of(p).
struct Binding {
  int mem[SPARSE_PAGES];
};

static uint32_t page_of(uint32_t p) {
  return p / 2;
}

static void apply(const Binding& b) {
  for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
    vx_buffer_h m = b.mem[p] < 0 ? nullptr : mem_buffer[b.mem[p]];
    RT_CHECK(vx_buffer_bind(sparse_buffer, uint64_t(p) * PAGE_SIZE, m,
                            uint64_t(page_of(p)) * PAGE_SIZE, PAGE_SIZE));
  }
}

static uint32_t pattern(int m, uint32_t w) {
  return (m == 0 ? 0xA0000000u : 0xB0000000u) | w;
}

static void launch(uint32_t tag) {
  kernel_arg_t arg = {};
  RT_CHECK(vx_buffer_address(sparse_buffer, &arg.sparse_addr));
  RT_CHECK(vx_buffer_address(read_buffer, &arg.read_addr));
  RT_CHECK(vx_buffer_address(amo_buffer, &arg.amo_addr));
  arg.tag = tag;
  arg.num_threads = NUM_THREADS;
  uint32_t max_bx = 1, max_by = 1, max_bz = 1;
  RT_CHECK(vx_kernel_get_max_block_size(kernel, &max_bx, &max_by, &max_bz));
  uint32_t block_x = std::min(NUM_THREADS, max_bx * max_by * max_bz);
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = (NUM_THREADS + block_x - 1) / block_x;
  li.block_dim[0] = block_x;
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
  wait(ev);
}

static int errors = 0;

static void expect(const char* what, uint32_t i, uint32_t expected, uint32_t actual) {
  if (expected == actual) {
    return;
  }
  if (errors < 16) {
    printf("*** error: %s[%u] expected=0x%x, actual=0x%x\n", what, i, expected, actual);
  }
  ++errors;
}

// Runs one launch under `b` and checks it against `mem`, the host's copy of
// both memory buffers, which it updates with the launch's effects.
static void run_phase(const char* name, const Binding& b, uint32_t tag,
                      std::vector<uint32_t> mem[2]) {
  apply(b);
  launch(tag);
  auto reads = read_buffer_words(read_buffer, NUM_THREADS);
  auto amos = read_buffer_words(amo_buffer, NUM_THREADS);
  for (uint32_t t = 0; t < NUM_THREADS; ++t) {
    uint32_t p = t / SLOTS;
    uint32_t w = page_of(p) * WORDS_PER_PAGE + (t % SLOTS) * SLOT_STRIDE;
    int m = b.mem[p];
    if (m < 0) {
      expect("unbound read", t, 0u, reads[t]);
      expect("unbound atomic", t, 0u, amos[t]);
      continue;
    }
    expect("bound read", t, mem[m][w], reads[t]);
    expect("bound atomic", t, mem[m][w + 2], amos[t]);
    mem[m][w + 1] = tag | t;
    mem[m][w + 2] += 1;
  }
  for (int m = 0; m < 2; ++m) {
    auto data = read_buffer_words(mem_buffer[m], MEM_PAGES * WORDS_PER_PAGE);
    for (uint32_t w = 0; w < data.size(); ++w) {
      expect(m == 0 ? "memory A" : "memory B", w, mem[m][w], data[w]);
    }
  }
  std::cout << name << ": " << (errors ? "errors" : "ok") << std::endl;
}

int main() {
  std::cout << "vm_sparse: " << SPARSE_PAGES << " sparse pages over two "
            << MEM_PAGES << "-page buffers" << std::endl;

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t vm_enabled = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_VM_SUPPORT, &vm_enabled));
  if (0 == vm_enabled) {
    // Sparse reservations are page-table mappings: without VM there are none.
    std::cout << "vm_sparse: device has no VM support, skipping" << std::endl;
    std::cout << "PASSED!" << std::endl;
    cleanup();
    return 0;
  }

  RT_CHECK(vx_buffer_create(device, SPARSE_PAGES * PAGE_SIZE, VX_MEM_READ_WRITE | VX_MEM_SPARSE,
                            &sparse_buffer));
  for (int m = 0; m < 2; ++m) {
    RT_CHECK(vx_buffer_create(device, MEM_PAGES * PAGE_SIZE, VX_MEM_READ_WRITE, &mem_buffer[m]));
  }
  RT_CHECK(vx_buffer_create(device, NUM_THREADS * sizeof(uint32_t), VX_MEM_READ_WRITE, &read_buffer));
  RT_CHECK(vx_buffer_create(device, NUM_THREADS * sizeof(uint32_t), VX_MEM_READ_WRITE, &amo_buffer));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<uint32_t> mem[2];
  for (int m = 0; m < 2; ++m) {
    mem[m].resize(MEM_PAGES * WORDS_PER_PAGE);
    for (uint32_t w = 0; w < mem[m].size(); ++w) {
      mem[m][w] = pattern(m, w);
    }
    write_buffer(mem_buffer[m], mem[m]);
  }

  // Phase 1: even pages on A, odd pages unbound.
  Binding b1;
  for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
    b1.mem[p] = (p % 2 == 0) ? 0 : -1;
  }
  run_phase("phase 1 (even pages bound)", b1, 0x10000000u, mem);

  // Phase 2: the even pages unbound, the odd ones on B. Translations of the
  // first launch must not survive into this one.
  Binding b2;
  for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
    b2.mem[p] = (p % 2 == 1) ? 1 : -1;
  }
  run_phase("phase 2 (rebound)", b2, 0x20000000u, mem);

  // Phase 3: two reservation pages on one memory page, the rest unbound.
  // Pages 4 and 5 both map B's page 2: the second page's slots see the
  // first page's stores and atomics.
  Binding b3;
  for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
    b3.mem[p] = (p == 4 || p == 5) ? 1 : -1;
  }
  apply(b3);
  launch(0x30000000u);
  {
    auto reads = read_buffer_words(read_buffer, NUM_THREADS);
    auto amos = read_buffer_words(amo_buffer, NUM_THREADS);
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      uint32_t p = t / SLOTS;
      uint32_t w = page_of(p) * WORDS_PER_PAGE + (t % SLOTS) * SLOT_STRIDE;
      if (p == 4 || p == 5) {
        expect("aliased read", t, mem[1][w], reads[t]);
        // Both aliases increment the one word; either may come first.
        if (amos[t] != mem[1][w + 2] && amos[t] != mem[1][w + 2] + 1) {
          expect("aliased atomic", t, mem[1][w + 2], amos[t]);
        }
      } else {
        expect("unbound read", t, 0u, reads[t]);
        expect("unbound atomic", t, 0u, amos[t]);
      }
    }
    auto stored = read_buffer_words(mem_buffer[1], MEM_PAGES * WORDS_PER_PAGE);
    for (uint32_t s = 0; s < SLOTS; ++s) {
      uint32_t w = page_of(4) * WORDS_PER_PAGE + s * SLOT_STRIDE;
      mem[1][w + 2] += 2;
      // The two aliases store to the same word: either store may land last.
      uint32_t t4 = 4 * SLOTS + s, t5 = 5 * SLOTS + s;
      uint32_t v = stored[w + 1];
      if (v != (0x30000000u | t4) && v != (0x30000000u | t5)) {
        expect("aliased store", w + 1, 0x30000000u | t5, v);
      }
      mem[1][w + 1] = v;
    }
    for (int m = 0; m < 2; ++m) {
      auto data = read_buffer_words(mem_buffer[m], MEM_PAGES * WORDS_PER_PAGE);
      for (uint32_t w = 0; w < data.size(); ++w) {
        expect(m == 0 ? "memory A" : "memory B", w, mem[m][w], data[w]);
      }
    }
    std::cout << "phase 3 (aliased): " << (errors ? "errors" : "ok") << std::endl;
  }

  // Phase 4: command-processor copies through the phase-2 binding. A write
  // across the whole reservation lands only in the bound pages; a read of it
  // returns 0 for the unbound ones.
  apply(b2);
  {
    std::vector<uint32_t> fill(SPARSE_PAGES * WORDS_PER_PAGE);
    for (uint32_t w = 0; w < fill.size(); ++w) {
      fill[w] = 0xC0000000u | w;
    }
    write_buffer(sparse_buffer, fill);
    auto back = read_buffer_words(sparse_buffer, SPARSE_PAGES * WORDS_PER_PAGE);
    for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
      for (uint32_t i = 0; i < WORDS_PER_PAGE; ++i) {
        uint32_t w = p * WORDS_PER_PAGE + i;
        uint32_t expected = (b2.mem[p] < 0) ? 0u : fill[w];
        expect("copy readback", w, expected, back[w]);
        if (b2.mem[p] >= 0) {
          mem[b2.mem[p]][page_of(p) * WORDS_PER_PAGE + i] = fill[w];
        }
      }
    }
    for (int m = 0; m < 2; ++m) {
      auto data = read_buffer_words(mem_buffer[m], MEM_PAGES * WORDS_PER_PAGE);
      for (uint32_t w = 0; w < data.size(); ++w) {
        expect(m == 0 ? "memory A" : "memory B", w, mem[m][w], data[w]);
      }
    }
    std::cout << "phase 4 (copies): " << (errors ? "errors" : "ok") << std::endl;
  }

  // Phase 5: indirect launches whose record and predicate lie in the
  // reservation. The record straddles two pages bound to unrelated memory
  // pages: x comes from A's page 3, y from B's page 3, not from A's page 4
  // that follows A's page 3 physically.
  {
    Binding b5;
    for (uint32_t p = 0; p < SPARSE_PAGES; ++p) {
      b5.mem[p] = (p == 6) ? 0 : (p == 7) ? 1 : -1;
    }
    apply(b5);
    const uint32_t x_groups = 4, y_groups = 1, decoy_y = 7;
    auto put = [&](vx_buffer_h buf, uint64_t off, uint32_t v) {
      vx_event_h ev = nullptr;
      RT_CHECK(vx_enqueue_write(queue, buf, off, &v, sizeof(v), 0, nullptr, &ev));
      wait(ev);
    };
    put(mem_buffer[0], 3 * PAGE_SIZE + PAGE_SIZE - 8, x_groups);  // record word 0
    put(mem_buffer[0], 4 * PAGE_SIZE + 4, decoy_y);               // where a contiguous read finds word 3
    put(mem_buffer[1], 3 * PAGE_SIZE + 4, y_groups);              // record word 3
    vx_buffer_h count_buffer = nullptr;
    RT_CHECK(vx_buffer_create(device, sizeof(uint32_t), VX_MEM_READ_WRITE, &count_buffer));
    put(count_buffer, 0, 0u);
    kernel_arg_t arg = {};
    RT_CHECK(vx_buffer_address(count_buffer, &arg.count_addr));
    uint64_t sparse_addr = 0;
    RT_CHECK(vx_buffer_address(sparse_buffer, &sparse_addr));
    vx_launch_indirect_t ind = {};
    ind.struct_size = sizeof(ind);
    ind.type = VX_STRUCT_TYPE_LAUNCH_INDIRECT;
    ind.record_addr = sparse_addr + 7 * uint64_t(PAGE_SIZE) - 8;
    ind.grid_word[0] = 0;
    ind.grid_word[1] = 3;
    ind.grid_word[2] = VX_INDIRECT_ONE;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.next         = &ind;
    li.kernel       = kernel;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 2;
    li.block_dim[0] = 1;
    li.block_dim[1] = 1;
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
    wait(ev);
    auto count = read_buffer_words(count_buffer, 1);
    expect("indirect grid threads", 0, x_groups * y_groups, count[0]);

    // An unbound page reads 0 for the command processor too: a predicate
    // there (page 0, unbound under b5) is 0, so the guarded launch is skipped.
    put(count_buffer, 0, 0u);
    vx_launch_indirect_t pred = {};
    pred.struct_size = sizeof(pred);
    pred.type = VX_STRUCT_TYPE_LAUNCH_INDIRECT;
    pred.pred_addr = sparse_addr;
    pred.pred_ref = 0;
    li.next = &pred;
    li.ndim = 1;
    li.grid_dim[0] = 4;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
    wait(ev);
    count = read_buffer_words(count_buffer, 1);
    expect("threads under an unbound predicate", 0, 0u, count[0]);
    RT_CHECK(vx_buffer_release(count_buffer));
    std::cout << "phase 5 (indirect record and predicate in the reservation): " << (errors ? "errors" : "ok") << std::endl;
  }

  // Phase 6: freeing the memory behind a bound page unbinds the page. A
  // buffer allocated on the freed pages afterwards must not show through the
  // reservation.
  {
    vx_buffer_h m = nullptr, n = nullptr;
    RT_CHECK(vx_buffer_create(device, PAGE_SIZE, VX_MEM_READ_WRITE, &m));
    RT_CHECK(vx_buffer_bind(sparse_buffer, 8 * uint64_t(PAGE_SIZE), m, 0, PAGE_SIZE));
    RT_CHECK(vx_buffer_release(m));
    RT_CHECK(vx_buffer_create(device, PAGE_SIZE, VX_MEM_READ_WRITE, &n));
    std::vector<uint32_t> fill(WORDS_PER_PAGE, 0xD3000000u);
    write_buffer(n, fill);
    std::vector<uint32_t> page(WORDS_PER_PAGE);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, page.data(), sparse_buffer, 8 * uint64_t(PAGE_SIZE),
                             PAGE_SIZE, 0, nullptr, &ev));
    wait(ev);
    for (uint32_t i = 0; i < WORDS_PER_PAGE; ++i) {
      expect("page whose memory was freed", i, 0u, page[i]);
    }
    RT_CHECK(vx_buffer_release(n));
    std::cout << "phase 6 (memory freed under a bound page): " << (errors ? "errors" : "ok") << std::endl;
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED! - " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
