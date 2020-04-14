#ifdef __MXPA__

#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <tbb/tbb.h>

namespace {
tbb::task_scheduler_init init;

class Foo {
public:
  Foo() {}
  void operator() (const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i != r.end(); i++) {
      printf ("");
    }
  }
};

}

extern "C"
void
mxpa_scheduler_init() {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, 100), Foo());
#if 0
  char cmd[32];
  int pid = getpid();
  printf("----------\n");
  sprintf(cmd, "ls -1 /proc/%d/task", pid);
  system(cmd);
  printf("----------\n");
#endif
}

#else

extern "C"
void
mxpa_scheduler_init() {
}

#endif

