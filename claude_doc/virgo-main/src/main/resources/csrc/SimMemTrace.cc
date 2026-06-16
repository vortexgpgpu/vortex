#ifndef NO_VPI
#include <vpi_user.h>
#include <svdpi.h>
#endif
#include <string>
#include <string.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <unistd.h>
#include "SimMemTrace.h"

// Single driver trace file used across the simulation.
// Currently doesn't support multiple driver traces.
std::string tracefilename{};
// Global singleton reader instance
static std::unique_ptr<MemTraceReader> reader;

MemTraceReader::MemTraceReader(const std::string &filename)
    : filename(filename) {
  char cwd[4096];
  if (getcwd(cwd, sizeof(cwd))) {
    printf("MemTraceReader: current working dir: %s\n", cwd);
  }

  infile.open(filename);
  if (infile.fail()) {
    fprintf(stderr, "MemTraceReader: error: failed to open file %s\n",
            filename.c_str());
    exit(EXIT_FAILURE);
  }
}

MemTraceReader::~MemTraceReader() {
  infile.close();
  printf("MemTraceReader destroyed\n");
}

void MemTraceReader::error(long fileline, const std::string &msg) {
  fprintf(stderr, "parse error at %s:%ld: %s\n", filename.c_str(), fileline,
          msg.c_str());
  exit(EXIT_FAILURE);
}

// Parse trace file in its entirety and store it into an internal structure.
// If `has_source` is true, assumes the trace has an additional column after
// core and lane_id for source id and tries to parse that.
// TODO: might block for a long time when the trace gets big, check if need to
// be broken down
void MemTraceReader::parse(const bool has_source) {
  MemTraceLine line;

  printf("MemTraceReader: started parsing\n");

  long size = 0;
  long source = 0;
  std::string loadstore; // slow?
  for (long fileline = 1;; fileline++) {
    if (infile.peek() == '\n') {
      infile.get();
      continue;
    }
    if (infile.eof()) {
      break;
    }

    if (!(infile >> line.cycle >> loadstore >> line.core_id >> line.lane_id)) {
      error(fileline, "failed parsing cycle..lane_id");
    }
    if (has_source && !(infile >> source)) {
      error(fileline, "failed parsing source");
    }
    if (!(infile >> std::hex >> line.address >> line.data >> std::dec >>
          size)) {
      error(fileline, "failed parsing address..size");
    }
    if (infile.get() != '\n') {
      error(fileline, "trailing characters at the end of the line");
    }

    line.valid = true;
    line.is_store = (loadstore == "STORE");
    if (size <= 0) {
      error(fileline, "invalid size in trace");
    }
    int lgsize = static_cast<int>(log2(size));
    if ((size & ~(~0lu << lgsize)) != 0) {
      error(fileline, "non-power-of-2 size detected in trace");
    }
    line.log_data_size = lgsize;

    trace_buf.push_back(line);
  }
  read_pos = trace_buf.cbegin();

  printf("MemTraceReader: finished parsing\n");
}

// Try to read a memory request that might have happened at a given cycle, on a
// given SIMD lane (= "thread").  In case no request happened at that point,
// return an empty line with .valid = false.
MemTraceLine MemTraceReader::read_trace_at(const long cycle, const int lane_id,
                                           unsigned char trace_read_ready) {
  MemTraceLine line;
  line.valid = false;

  if (finished()) {
    return line;
  }

  line = *read_pos;
  // It should always be guaranteed that we consumed all of the past lines, and
  // the next line is in the future.
  if (line.cycle < cycle) {
    long fileline = read_pos - std::cbegin(trace_buf) + 1;
    error(fileline, "some trace lines are left unread in the past. "
                    "Tried cycle=" +
                        std::to_string(cycle) +
                        ", found line.cycle=" + std::to_string(line.cycle) +
                        ". Is NUM_LANES set correctly?");
    return MemTraceLine{};
  }

  if (line.cycle > cycle) {
    // We haven't reached the cycle mark specified in this line yet, so we don't
    // read it right now.
    return MemTraceLine{};
  } else if (line.lane_id != lane_id) {
    return MemTraceLine{};
  } else if (line.cycle == cycle && line.lane_id == lane_id) {
    if (trace_read_ready) {
      printf("Fire! cycle=%ld, valid=%d, %s addr=%lx, size=%d \n", cycle,
             line.valid, (line.is_store ? "STORE" : "LOAD"), line.address,
             line.log_data_size);

      // NOTE: Currently lane_id is assumed to be in always-increasing order,
      // e.g. 0->1->2->3->0->..., both in the trace file and the order the
      // caller calls this function.  If this is not true, we cannot simply
      // monotonically increment read_pos.  lane_id need not be contiguous, e.g.
      // 0->1->3 is fine.
      ++read_pos;
    } else {
      // For debugging purposes, instead of early-returning on
      // !trace_read_ready, print something to notify we are blocking a valid
      // trace line.
      printf("All Lanes Blocked on this cycle! cycle=%ld \n", cycle);
    }
    // We want to return valid line regardless of `trace_read_ready` or not,
    // because we want to let the driver know that it missed a valid line at the
    // given cycle, so that it holds its cycle counter and safely reads back the
    // line in the future.
    return line;
  }

  assert(!"unreachable");
}

extern "C" void memtrace_init(const char *filename, bool has_source) {
#ifndef NO_VPI
  // If VPI option is given, override filename set from Chisel/Verilog.
  s_vpi_vlog_info info;
  if (!vpi_get_vlog_info(&info)) {
    fprintf(stderr, "fatal: failed to get plusargs from VCS\n");
    exit(1);
  }
  const char *TRACEFILENAME_PLUSARG = "+memtracefile=";
  for (int i = 0; i < info.argc; i++) {
    char *input_arg = info.argv[i];
    if (strncmp(input_arg, TRACEFILENAME_PLUSARG,
                strlen(TRACEFILENAME_PLUSARG)) == 0) {
      printf(
          "memtrace_init: +memtracefile given, overriding Verilog parameter\n");
      filename = input_arg + strlen(TRACEFILENAME_PLUSARG);
      break;
    }
  }
#endif

  printf("memtrace_init: filename=[%s]\n", filename);

  tracefilename = filename;

  reader = std::make_unique<MemTraceReader>(filename);
  // parse file upfront
  // driver trace file is assumed to not have source id
  reader->parse(has_source);
}

// TODO: accept core_id as well
extern "C" void memtrace_query(unsigned char trace_read_ready,
                               unsigned long trace_read_cycle,
                               int           trace_read_lane_id,
                               unsigned char *trace_read_valid,
                               unsigned long *trace_read_address,
                               unsigned char *trace_read_is_store,
                               unsigned char *trace_read_size, // logsize, don't need full int
                               unsigned long *trace_read_data,
                               unsigned char *trace_read_finished) {
  // printf("memtrace_query(cycle=%ld, tid=%d)\n", trace_read_cycle,
  //        trace_read_lane_id);

  auto line = reader->read_trace_at(trace_read_cycle, trace_read_lane_id, trace_read_ready);
  *trace_read_valid = line.valid;
  *trace_read_address = line.address;
  *trace_read_is_store = line.is_store;
  *trace_read_size = line.log_data_size;
  *trace_read_data = line.data;
  // This means finished and valid will go up at the same cycle.  Need to
  // handle this without skipping the last line.
  *trace_read_finished = reader->finished();

  return;
}
