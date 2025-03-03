#include <stdio.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

////////////////////////////////////////////////////////////////////////////////
// Very Simple Malloc implementation
// We'll replace this with a more sohpisticated one later taht uses a free list
// and a memory pool that's defined in the linker script.
// For now, lets assume that we have a memory pool that is defined as a global
// array of bytes.

#define HEAP_SZ 1024 * 1024

char __data_pool[HEAP_SZ];  // pool
int __data_pool_offset = 0; // Tracks how much memory has been used

// We'll hook this function to the vx_mem_alloc function in the future
void *vx_malloc(int sz) {
  if (__data_pool_offset + sz > HEAP_SZ) {
    vx_printf("Out of memory\n");
    return nullptr;
  }

  // return a pointer at current offset, then move offset based on allocated size
  void *ptr = &__data_pool[__data_pool_offset];
  __data_pool_offset += sz;
  return ptr;
}

// we'll hook this function to the vx_mem_free function in the future
void vx_free(void *ptr) {
  // Do nothing (for now)
}

////////////////////////////////////////////////////////////////////////////////
// Kernel

typedef struct {
  int *src0;
  int *src1;
  int *dst;
  int num_elements;
} vecadd_args_t;

// Basic vector add kernel where each thread in a block will add its index of the vector
void vecadd_kernel(vecadd_args_t *__UNIFORM__ args) {
  auto src0_ptr = reinterpret_cast<int *>(args->src0);
  auto src1_ptr = reinterpret_cast<int *>(args->src1);
  auto dst_ptr = reinterpret_cast<int *>(args->dst);

  // Print addresses
  dst_ptr[blockIdx.x] = src0_ptr[blockIdx.x] + src1_ptr[blockIdx.x];
  vx_printf("[+] I am thread %d, I am adding %d and %d\n", blockIdx.x, src0_ptr[blockIdx.x], src1_ptr[blockIdx.x]);
}

////////////////////////////////////////////////////////////////////////////////
// Host code -> this will actually be run on vortex itself!

int main() {
  // We start in single threaded mode
  // core0, thread0 runs the host part of the program and initializes
  // the buffers and the kernel arguments.
  vx_printf(">> Starting host part of the code in hostless mode (coreid=%d, warpid=%d. threadid=%d)\n",
            vx_core_id(), vx_warp_id(), vx_thread_id());

  vx_printf(">> Malloc Pool address: %p\n", __data_pool);
  vx_printf(">> Malloc Pool size: %d\n", HEAP_SZ);

  // Allocate buffers
  vx_printf(">> Allocating buffers\n");
  int N = 16;
  int *src0 = (int *)vx_malloc(N * sizeof(int));
  int *src1 = (int *)vx_malloc(N * sizeof(int));
  int *dst = (int *)vx_malloc(N * sizeof(int));

  // Initialize buffers
  vx_printf(">> Initializing buffers\n");
  for (int i = 0; i < N; i++) {
    src0[i] = i * 2;
    src1[i] = i * 3;
  }

  // Print
  vx_printf(">> src0 buffer [0x%p]: ", src0);
  for (int i = 0; i < N; i++) {
    vx_printf("%02d ", src0[i]);
  }
  vx_printf("\n");

  vx_printf(">> src1 buffer [0x%p]: ", src1);
  for (int i = 0; i < N; i++) {
    vx_printf("%02d ", src1[i]);
  }
  vx_printf("\n");

  // Set kernel arguments (allocated on heap? what about thread local stack)
  vecadd_args_t args;
  args.src0 = src0;
  args.src1 = src1;
  args.dst = dst;
  args.num_elements = N;

  vx_printf(">> kernel_arg.src0: %p\n", args.src0);
  vx_printf(">> kernel_arg.src1: %p\n", args.src1);
  vx_printf(">> kernel_arg.dst: %p\n", args.dst);
  vx_printf(">> kernel_arg.num_elements: %d\n", args.num_elements);

  // Launch kernel
  vx_printf(">> Launching kernel (spawning lots of threads)\n");

  uint32_t total_threads = args.num_elements;
  // We spawn threads to execute the kernel
  // This will be replaced by vx_start in the future
  vx_spawn_threads(1, &total_threads, nullptr, (vx_kernel_func_cb)vecadd_kernel, &args);

  vx_printf(">> Kernel finished executing\n");

  // At this point we're back to single threaded execution
  vx_printf(">> we're back to single threaded execution (coreid=%d, warpid=%d. threadid=%d)\n",
            vx_core_id(), vx_warp_id(), vx_thread_id());

  // Print results:
  // NOTE: This should be printed by only 1 thread
  vx_printf("dst: ");
  int error_idx = -1;
  for (int i = 0; i < N; i++) {
    if (dst[i] != src0[i] + src1[i]) {
      error_idx = i;
    }
    vx_printf("%d ", dst[i]);
  }
  vx_printf("\n");

  if (error_idx == -1) {
    vx_printf("*** Vortex successfully ran in hostless mode! ***\n");
  } else {
    vx_printf("*** Something went wrong (idx=%d) ***\n", error_idx);
  }
  return 0;
}
