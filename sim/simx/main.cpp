#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>
#include "processor.h"
#include "archdef.h"
#include "mem.h"
#include "constants.h"
#include <util.h>
#include "args.h"

using namespace vortex;

int main(int argc, char **argv) {
  int exitcode = 0;

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
  
  {
    // create processor configuation
    ArchDef arch(archStr, num_cores, num_warps, num_threads);

    // create memory module
    RAM ram(RAM_PAGE_SIZE);

    // load program
    {
      std::string program_ext(fileExtension(imgFileName.c_str()));
      if (program_ext == "bin") {
        ram.loadBinImage(imgFileName.c_str(), STARTUP_ADDR);
      } else if (program_ext == "hex") {
        ram.loadHexImage(imgFileName.c_str());
      } else {
        std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }

    // create processor
    Processor processor(arch);
  
    // attach memory module
    processor.attach_ram(&ram);   

    // run simulation
    processor.run();
  } 

  if (riscv_test) {
    if (1 == exitcode) {
      std::cout << "Passed." << std::endl;
      exitcode = 0;
    } else {
      std::cout << "Failed." << std::endl;
    }
  } else {
    if (exitcode != 0) {
      std::cout << "*** error: exitcode=" << exitcode << std::endl;
    }
  }  

  return exitcode;
}
