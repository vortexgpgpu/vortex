#ifndef NO_VPI
#include <svdpi.h>
#include <vpi_user.h>
#endif
#include "SimMemTrace.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>

// Contains handle for every logger that is instantiated per Verilog module
// instance
static std::vector<std::unique_ptr<MemTraceWriter>> loggers;

MemTraceWriter::MemTraceWriter(const bool is_response,
                               const std::string &filename) {
  this->is_response = is_response;

  char cwd[4096];
  if (getcwd(cwd, sizeof(cwd))) {
    printf("MemTraceWriter: current working dir: %s\n", cwd);
  }

  outfile = fopen(filename.c_str(), "w");
  if (!outfile) {
    fprintf(stderr, "failed to open file %s\n", filename.c_str());
  }
}

MemTraceWriter::~MemTraceWriter() {
  fclose(outfile);
  printf("MemTraceWriter(handle=%zu) destroyed\n", handle);
}

void MemTraceWriter::write_line_to_trace(const MemTraceLine line) {
  fprintf(outfile, "%ld %s %d %d %d 0x%lx 0x%lx %u\n", line.cycle,
          (line.is_store ? "STORE" : "LOAD"), line.core_id, line.lane_id,
          line.source, line.address, line.data, (1u << line.log_data_size));
}

// Returns the "handle" ID for this particular logger instance.
extern "C" int memtracelogger_init(int is_response, const char *filename,
                                   const char *filename_suffix) {
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
      filename = input_arg + strlen(TRACEFILENAME_PLUSARG);
      break;
    }
  }
#endif

  auto fullfilename = std::string{filename} + std::string{filename_suffix};

  int handle = loggers.size();
  loggers.emplace_back(
      std::make_unique<MemTraceWriter>(is_response, fullfilename));
  loggers.back()->handle = handle;

  printf("memtracelogger_init: handle=%d, is_response=%d, filename=[%s]\n",
         handle, is_response, fullfilename.c_str());

  return handle;
}

// This is used to log both TileLink A and D channels.
// TODO: accept core_id as well
extern "C" void memtracelogger_log(int handle,
                                   unsigned char trace_log_valid,
                                   unsigned long trace_log_cycle,
                                   int           trace_log_lane_id,
                                   int           trace_log_source,
                                   unsigned long trace_log_address,
                                   unsigned char trace_log_is_store,
                                   int           trace_log_size,
                                   unsigned long trace_log_data,
                                   unsigned char *trace_log_ready) {
  // printf("memtrace_query(cycle=%ld, tid=%d)\n", trace_read_cycle,
  //        trace_read_lane_id);
  *trace_log_ready = 1;

  if (!trace_log_valid) {
    return;
  }

  // printf("%s: [%lu] valid: address=%lx, tid=%u, size=%d\n", __func__,
  //        trace_log_cycle, trace_log_address, trace_log_lane_id, trace_log_size);

  MemTraceLine line{.valid = (trace_log_valid == 1),
                    .cycle = static_cast<long>(trace_log_cycle),
                    .core_id = 0, // TODO support multicores
                    .lane_id = trace_log_lane_id,
                    .source = trace_log_source,
                    .address = trace_log_address,
                    .is_store = (trace_log_is_store == 1),
                    .data = trace_log_data,
                    .log_data_size = trace_log_size};

  assert(0 <= handle && static_cast<size_t>(handle) < loggers.size() &&
         "wrong trace logger handle");
  auto logger = loggers[handle].get();
  logger->write_line_to_trace(line);
}
