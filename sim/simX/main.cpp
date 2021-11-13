#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>
#include "processor.h"
#include "args.h"

using namespace vortex;

int main(int argc, char **argv) {
  int ret;

  std::string archStr("rv32imf");
  std::string imgFileName;
  int num_cores(NUM_CORES * NUM_CLUSTERS);
  int num_warps(NUM_WARPS);
  int num_threads(NUM_THREADS);  
  bool showHelp(false);
  bool showStats(false);
  bool riscv_test(false);

  /* Read the command line arguments. */
  CommandLineArgFlag fh("-h", "--help", "", showHelp);
  CommandLineArgSetter<std::string> fa("-a", "--arch", "", archStr);  
  CommandLineArgSetter<std::string> fi("-i", "--image", "", imgFileName);
  CommandLineArgSetter<int> fc("-c", "--cores", "", num_cores);
  CommandLineArgSetter<int> fw("-w", "--warps", "", num_warps);
  CommandLineArgSetter<int> ft("-t", "--threads", "", num_threads);
  CommandLineArgFlag fr("-r", "--riscv", "", riscv_test);
  CommandLineArgFlag fs("-s", "--stats", "", showStats);

  CommandLineArg::readArgs(argc - 1, argv + 1);

  if (showHelp || imgFileName.empty()) {
    std::cout << "Vortex emulator command line arguments:\n"
                 "  -i, --image <filename> Program RAM image\n"
                 "  -c, --cores <num> Number of cores\n"
                 "  -w, --warps <num> Number of warps\n"
                 "  -t, --threads <num> Number of threads\n"
                 "  -a, --arch <arch string> Architecture string\n"
                 "  -r, --riscv riscv test\n"
                 "  -s, --stats Print stats on exit.\n";
    return 0;
  }

  std::cout << "Running " << imgFileName << "..." << std::endl;
  
  if (!SimPlatform::instance().initialize())
    return -1;

  {
    ArchDef arch(archStr, num_cores, num_warps, num_threads);
    Processor processor(arch);
    ret = processor.run(imgFileName, riscv_test, showStats);
  }  

  SimPlatform::instance().finalize();

  return ret;
}
