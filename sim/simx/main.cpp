#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>

#include "debug.h"
#include "types.h"
#include "core.h"
#include "args.h"

using namespace vortex;

int main(int argc, char **argv) {

  std::string archString("rv32imf");
  int num_cores(NUM_CORES * NUM_CLUSTERS);
  int num_warps(NUM_WARPS);
  int num_threads(NUM_THREADS);
  std::string imgFileName;
  bool showHelp(false);
  bool showStats(false);
  bool riscv_test(false);

  /* Read the command line arguments. */
  CommandLineArgFlag fh("-h", "--help", "", showHelp);
  CommandLineArgSetter<std::string> fa("-a", "--arch", "", archString);  
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

  ArchDef arch(archString, num_cores, num_warps, num_threads);

  Decoder decoder(arch);
  MemoryUnit mu(0, arch.wsize(), true);
  
  RAM ram((1<<12), (1<<20));

  std::string program_ext(fileExtension(imgFileName.c_str()));
  if (program_ext == "bin") {
    ram.loadBinImage(imgFileName.c_str(), STARTUP_ADDR);
  } else if (program_ext == "hex") {
    ram.loadHexImage(imgFileName.c_str());
  } else {
    std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
    return -1;
  }

  mu.attach(ram, 0, 0xFFFFFFFF);

  struct stat hello;
  fstat(0, &hello);

  std::vector<std::shared_ptr<Core>> cores(num_cores);
  for (int i = 0; i < num_cores; ++i) {
    cores[i] = std::make_shared<Core>(arch, decoder, mu, i);
  }

  bool running;
  int exitcode = 0;
  do {
    running = false;
    for (auto& core : cores) {            
      core->step();
      if (core->running()) {
          running = true;
      }
      if (core->check_ebreak()) {
        exitcode = core->getIRegValue(3);
        running = false;
        break;
      }
    }
  } while (running);

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
