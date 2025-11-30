#include <mem_alloc.h>
#include <stdio.h>

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     return -1;                                                 \
   } while (false)

static uint64_t minAddress = 0;
static uint64_t maxAddress = 0xffffffff;
static uint32_t pageAlign  = 4096;
static uint32_t blockAlign = 64;

int main() {

    auto allocator = new vortex::MemoryAllocator(
        minAddress, maxAddress, pageAlign, blockAlign
    );

    uint64_t a0, a1, a2, a3;

    RT_CHECK(allocator->allocate(128, &a0));
    RT_CHECK(allocator->release(a0));

    RT_CHECK(allocator->allocate(1, &a0));
    RT_CHECK(allocator->allocate(1, &a1));
    RT_CHECK(allocator->allocate(1, &a2));
    RT_CHECK(allocator->release(a1));
    RT_CHECK(allocator->allocate(1, &a3));
    RT_CHECK(allocator->release(a0));
    RT_CHECK(allocator->release(a2));
    RT_CHECK(allocator->release(a3));

    RT_CHECK(allocator->allocate(5878, &a0));
    RT_CHECK(allocator->allocate(4095, &a1));
    RT_CHECK(allocator->allocate(1, &a2));
    RT_CHECK(allocator->allocate(1, &a3));
    RT_CHECK(allocator->release(a0));
    RT_CHECK(allocator->release(a1));
    RT_CHECK(allocator->release(a2));
    RT_CHECK(allocator->release(a3));

    delete allocator;

    printf("PASSED!\n");

    return 0;
}