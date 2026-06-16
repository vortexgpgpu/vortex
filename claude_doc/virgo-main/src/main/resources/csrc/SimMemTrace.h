#include <vector>
#include <memory>
#include <fstream>

extern std::string tracefilename;

struct MemTraceLine {
  bool valid = false;
  long cycle = 0;
  int core_id = 0;
  int lane_id = 0;
  int source = 0;
  unsigned long address = 0;
  bool is_store = 0;
  unsigned long data = 0;
  int log_data_size = 0;
};

class MemTraceReader {
public:
  MemTraceReader(const std::string &filename);
  ~MemTraceReader();
  void parse(const bool has_source);
  MemTraceLine read_trace_at(const long cycle, const int lane_id, unsigned char trace_read_ready);
  bool finished() const { return read_pos == trace_buf.cend(); }
  MemTraceLine peek() const { return *read_pos; }
  void next() { read_pos++; }
  void error(long fileline, const std::string &msg);

  const std::string filename;

private:
  std::ifstream infile;
  std::vector<MemTraceLine> trace_buf;
  std::vector<MemTraceLine>::const_iterator read_pos;
};

class MemTraceWriter {
public:
  MemTraceWriter(const bool is_response, const std::string &filename);
  ~MemTraceWriter();
  void write_line_to_trace(const MemTraceLine line);

  size_t handle;
  bool is_response;
  FILE *outfile;
};

extern "C" void memtrace_init(const char *filename, bool has_source);
extern "C" void memtrace_query(unsigned char trace_read_ready,
                               unsigned long trace_read_cycle,
                               int           trace_read_lane_id,
                               unsigned char *trace_read_valid,
                               unsigned long *trace_read_address,
                               unsigned char *trace_read_is_store,
                               unsigned char *trace_read_size,
                               unsigned long *trace_read_data,
                               unsigned char *trace_read_finished);
extern "C" int memtracelogger_init(int is_response, const char *filename,
                                   const char *filename_suffix);
extern "C" void memtracelogger_log(int handle,
                                   unsigned char trace_log_valid,
                                   unsigned long trace_log_cycle,
                                   int           trace_log_lane_id,
                                   int           trace_log_source,
                                   unsigned long trace_log_address,
                                   unsigned char trace_log_is_store,
                                   int           trace_log_size,
                                   unsigned long trace_log_data,
                                   unsigned char *trace_log_ready);
