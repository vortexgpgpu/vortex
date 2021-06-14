#ifndef MXPA_RUNTIME_PERF_MONITOR
#define MXPA_RUNTIME_PERF_MONITOR

#include "perf_util.h"

#ifdef __cplusplus
extern "C" {
#endif

void perf_init();
void perf_start(const char* kname);
void perf_end(const char* kname);
void perf_fini();

#ifdef __cplusplus
};
#endif

#endif

