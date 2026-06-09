#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>
#include "perfmon.h"

static const char* mxpa_profile_log = "mxpa_profile_%d.log";
static int enabled = 0;

// quartet's CPU supports these. Check check_events at libpfm4
// to make this adaptable.
static const char *gen_events_all[]={
  "snb_ep::L3_LAT_CACHE:MISS",
  "snb_ep::L3_LAT_CACHE:REFERENCE",

  "snb_ep::L2_RQSTS:ALL_DEMAND_DATA_RD",
  "snb_ep::L2_RQSTS:ALL_DEMAND_RD_HIT",

  "perf::PERF_COUNT_HW_CACHE_L1D:ACCESS",
  "perf::PERF_COUNT_HW_CACHE_L1D:MISS",

  "perf::PERF_COUNT_HW_CACHE_L1D:PREFETCH",
  "perf::L1-DCACHE-PREFETCH-MISSES",

  "perf::PERF_COUNT_HW_CACHE_L1I:READ",
  "perf::PERF_COUNT_HW_CACHE_L1I:MISS",

  "perf::ITLB-LOADS",
  "perf::ITLB-LOAD-MISSES",
  "perf::DTLB-LOADS",
  "perf::DTLB-LOAD-MISSES",
  "perf::CONTEXT-SWITCHES",
  "perf::CPU-MIGRATIONS",
  "perf::CYCLES",
  "snb_ep::RESOURCE_STALLS:ANY",

  "perf::INSTRUCTIONS",
  "perf::BRANCH-INSTRUCTIONS",
  "perf::BRANCHES",
  "perf::BRANCH-MISSES",
  NULL
};

#define NUM_MAX_THREAD    256

static perf_event_desc_t *g_fds[NUM_MAX_THREAD];
static int g_nthreads;
static int num_fds = 0;

/* note: unsafe for multithreading */
static uint64_t* begins;

static void
fetch_counts(perf_event_desc_t *fds, int num_fds)
{
  if (begins == 0) {
    begins = (uint64_t*) malloc(num_fds * sizeof(uint64_t));
    memset(begins, 0, num_fds * sizeof(uint64_t));
  }

	uint64_t val;
	uint64_t values[3];
	double ratio;
	int i;
	ssize_t ret;

	/*
	 * now read the results. We use pfp_event_count because
	 * libpfm guarantees that counters for the events always
	 * come first.
	 */
	memset(values, 0, sizeof(values));

	for (i = 0; i < num_fds; i++) {
		ret = read(fds[i].fd, values, sizeof(values));
		if (ret < (ssize_t)sizeof(values)) {
			if (ret == -1)
				fprintf(stderr, "cannot read results: %s", strerror(errno));
			else
				warnx("could not read event%d", i);
		}
		/*
		 * scaling is systematic because we may be sharing the PMU and
		 * thus may be multiplexed
		 */
    int valid = 0;
		val = perf_scale_valid(values, &valid);
    if (valid == 0) printf ("@i=%d, v0=%llu, v1=%llu, v2=%llu, val=%llu\n", i, values[0], values[1], values[2], val);
		ratio = perf_scale_ratio(values);

    begins[i] = val;
  }
}

static void
print_counts(perf_event_desc_t *fds, int num_fds, const char *msg, FILE* fp)
{
	uint64_t val;
	uint64_t values[3];
	double ratio;
	int i;
	ssize_t ret;

#if 0
  fprintf(fp, "%s ------------------------------------\n", msg);
#else
  fprintf(fp, "method=%s", msg);
#endif

	/*
	 * now read the results. We use pfp_event_count because
	 * libpfm guarantees that counters for the events always
	 * come first.
	 */
	memset(values, 0, sizeof(values));

	for (i = 0; i < num_fds; i++) {

		ret = read(fds[i].fd, values, sizeof(values));
		if (ret < (ssize_t)sizeof(values)) {
			if (ret == -1)
				fprintf(stderr, "cannot read results: %s", strerror(errno));
			else
				warnx("could not read event%d", i);
		}
		/*
		 * scaling is systematic because we may be sharing the PMU and
		 * thus may be multiplexed
		 */
    int valid;
		val = perf_scale_valid(values, &valid);
    if (valid == 0) printf ("!i=%d, v0=%llu, v1=%llu, v2=%llu, val=%llu\n", i, values[0], values[1], values[2], val);
		ratio = perf_scale_ratio(values);

#if 0
		fprintf(fp, "%s %'20"PRIu64" %s (%.2f%% scaling, raw=%'"PRIu64", ena=%'"PRIu64", run=%'"PRIu64")\n",
			"-", // msg,
			val,
			fds[i].name,
			(1.0-ratio)*100.0,
		        values[0],
			values[1],
			values[2]);
#else
    fprintf (fp, " %s=%llu", fds[i].name, val); // valid ? val : 0);
#endif
	}
  fprintf (fp, "\n");
}

FILE* open_log_file(char* fname) {
  FILE* fp;
  if (fp = fopen(fname, "r")) {
    fclose(fp);
    return fopen(fname, "a");
  }
  fp = fopen(fname, "a");
  return fp;
}

void perf_init() {
  static int init = 0;
 
  if (init) return;
  init = 1;

  char* prof_envvar = getenv("MXPA_PROFILE");
  if (prof_envvar) {
    enabled = 1;
  } else {
    return;
  }

  pfm_initialize();
}

static void get_tids(int* tids, int* number) {
  char path[32];
  int pid = getpid();
  sprintf (path, "/proc/%d/task", pid);
  struct dirent *de=NULL;
  DIR *d=NULL;
  d=opendir(path);
  assert(d != NULL && "Null for opendir");
  // Loop while not NULL
  char pid_str[8];
  char last[8];
  sprintf (pid_str, "%d", pid);
  int n = 0;
  while(de = readdir(d)) {
    if (!strcmp(de->d_name, ".")) continue;
    if (!strcmp(de->d_name, "..")) continue;
    if (!strcmp(de->d_name, pid_str)) continue;
    *tids++ = atoi(de->d_name);
    n++;
  }
  *number = n;
  // printf ("Sampling thread %d\n", tid);
  closedir(d);
}

void perf_start(const char* kname) {
  if (!enabled) return;

  char* prof_envvar = getenv("MXPA_PROFILE");

  int tids[32];
  int ntid;
  get_tids(tids, &ntid);
  g_nthreads = ntid;

  int n;
  for (n = 0; n < ntid; n++) {
    int ret;
    ret = perf_setup_list_events(prof_envvar, &(g_fds[n]), &num_fds);
    perf_event_desc_t *fds = g_fds[n];
    int cpu = -1;
    int group_fd = -1;
    int pid = tids[n];
    fds[0].fd = -1;
    int i;
    for(i=0; i < num_fds; i++) {
      fds[i].hw.read_format = PERF_FORMAT_SCALE;
      fds[i].hw.disabled = 1; /* do not start now */
      fds[i].hw.inherit = 1;  /* XXX child process will inherit, when forked only? */

      /* each event is in an independent group (multiplexing likely) */
      fds[i].fd = perf_event_open(&fds[i].hw, pid, cpu, group_fd, 0);
      if (fds[i].fd == -1) {
        fprintf(stderr, "cannot open event %d\n", i);
        exit(2);
      }
    }
  }
  prctl(PR_TASK_PERF_EVENTS_ENABLE);
}

void perf_end(const char* kname) {
  if (!enabled) return;
  int i, n;
  prctl(PR_TASK_PERF_EVENTS_DISABLE);

  static int first_time = 1;
  if (first_time) {
    first_time = 0;
    char name[128];
    for (n = 0; n < g_nthreads; n++) {
      sprintf (name, mxpa_profile_log, n);
      FILE* fp = fopen(name, "w");
      fclose(fp);
    }
  }

  char name[128];
  for (n = 0; n < g_nthreads; n++) {
    sprintf (name, mxpa_profile_log, n);
    FILE* fp = open_log_file(name);
    perf_event_desc_t *fds = g_fds[n];
    print_counts(fds, num_fds, kname, fp);
    for (i = 0; i < num_fds; i++) close(fds[i].fd);
    perf_free_fds(fds, num_fds);
    g_fds[n] = fds = NULL;
    fclose(fp);
  }
}

void pin_trace_enable(char* n) {
  perf_start((const char*)n);
}

void pin_trace_disable(char* n) {
  perf_end((const char*)n);
}


