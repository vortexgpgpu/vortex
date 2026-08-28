// Post-mortem device-memory probe. Reserves the exact device addresses a
// prior (failing) demo run used, downloads them through the CP DMA path, and
// prints what device memory actually holds — discriminating "the upload write
// never landed" (buffer holds stale bytes) from "the upload landed but the
// kernel read it stale" (buffer holds the source data).
//
// Usage: probe [-a addr] [-s bytes]   (defaults probe demo's 3 buffers)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vortex2.h>

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret) break;                                     \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    exit(-1);                                                 \
  } while (false)

static void probe(vx_device_h dev, uint64_t addr, uint64_t size) {
  vx_buffer_h buf = nullptr;
  RT_CHECK(vx_buffer_reserve(dev, addr, size, VX_MEM_READ, &buf));
  void* p = nullptr;
  RT_CHECK(vx_buffer_map(buf, 0, size, VX_MEM_READ, &p));
  const int32_t* w = static_cast<const int32_t*>(p);
  printf("[probe] 0x%lx:", addr);
  for (int i = 0; i < 8; ++i) printf(" %d", w[i]);
  printf("\n");
  vx_buffer_unmap(buf, p);
  vx_buffer_release(buf);
}

int main(int argc, char** argv) {
  uint64_t addr = 0, size = 1024;
  int c;
  while ((c = getopt(argc, argv, "a:s:")) != -1) {
    if (c == 'a') addr = strtoull(optarg, nullptr, 0);
    if (c == 's') size = strtoull(optarg, nullptr, 0);
  }
  vx_device_h dev = nullptr;
  RT_CHECK(vx_device_open(0, &dev));
  if (addr) {
    probe(dev, addr, size);
  } else {
    // demo's buffers: src0, src1, dst
    probe(dev, 0x10000, 1024);
    probe(dev, 0x10400, 1024);
    probe(dev, 0x10800, 1024);
  }
  vx_device_release(dev);
  printf("PASSED!\n");
  return 0;
}
