/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>


#include "include/debug.h"
#include "include/types.h"
#include "include/core.h"
#include "include/enc.h"
#include "include/instruction.h"
#include "include/mem.h"
#include "include/obj.h"
#include "include/archdef.h"

#include "include/args.h"
#include "include/help.h"

#include <sys/stat.h>

//////////////
/////////////

using namespace Harp;
using namespace HarpTools;
using namespace std; 

enum HarpToolMode { HARPTOOL_MODE_ASM, HARPTOOL_MODE_DISASM, HARPTOOL_MODE_EMU, 
                    HARPTOOL_MODE_LD,  HARPTOOL_MODE_HELP };

HarpToolMode findMode(int argc, char** argv) {
  bool mode_asm, mode_disasm, mode_emu, mode_ld, mode_help;

  if (argc == 0) return HARPTOOL_MODE_HELP;

  CommandLineArgFlag fh("--help", "-h", "", mode_help);
  CommandLineArgFlag fa("-A", "--asm", "", mode_asm);
  CommandLineArgFlag fd("-D", "--disasm", "", mode_disasm);
  CommandLineArgFlag fe("-E", "--emu", "", mode_emu);
  CommandLineArgFlag fl("-L", "--ld", "", mode_ld);

  CommandLineArg::readArgs((argc == 0?0:1), argv);
  CommandLineArg::clearArgs();

  if (mode_asm)    return HARPTOOL_MODE_ASM;
  if (mode_disasm) return HARPTOOL_MODE_DISASM;
  if (mode_emu)    return HARPTOOL_MODE_EMU;
  if (mode_ld)     return HARPTOOL_MODE_LD;
  return HARPTOOL_MODE_HELP;
}

int emu_main(int argc, char **argv) {
  string archString("rv32i"), imgFileName("a.dsfsdout.bin");
  bool showHelp, showStats, basicMachine, batch;

  /* Read the command line arguments. */
  CommandLineArgFlag          fh("-h", "--help", "", showHelp);
  CommandLineArgSetter<string>fc("-c", "--core", "", imgFileName);
  CommandLineArgSetter<string>fa("-a", "--arch", "", archString);
  CommandLineArgFlag          fs("-s", "--stats", "", showStats);
  CommandLineArgFlag          fb("-b", "--basic", "", basicMachine);
  CommandLineArgFlag          fi("-i", "--batch", "", batch);
  
  CommandLineArg::readArgs(argc, argv);
  if (showHelp) {
    cout << Help::emuHelp;
    return 0;
  }

  /* Instantiate a Core, RAM, and console output. */
  ArchDef arch(archString);

  Decoder *dec;

  switch (arch.getEncChar()) {
    case 'b': dec = new WordDecoder(arch); break;
    case 'w': dec = new WordDecoder(arch); break;
    case 'r': dec = new WordDecoder(arch); break;
    default:
      cout << "Unrecognized decoder type: '" << arch.getEncChar() << "'.\n";
      return 1;
  }

    // std::cout << "TESTING:  " << tests[t] << "\n"; 


    MemoryUnit mu(4096, arch.getWordSize(), basicMachine);
    Core core(arch, *dec, mu/*, ID in multicore implementations*/);

    // RamMemDevice mem(imgFileName.c_str(), arch.getWordSize());
    RAM old_ram;
    old_ram.loadHexImpl(imgFileName.c_str());
    // old_ram.loadHexImpl(tests[t]);
    // MemDevice * memory = &old_ram;

    ConsoleMemDevice console(arch.getWordSize(), cout, core, batch);
    mu.attach(old_ram,     0);
    mu.attach(console, 1ll<<(arch.getWordSize()*8 - 1));
    // mu.attach(console, 0xf0000000);

    // core.w[0].pc = 0x8000007c; // If I want to start at a specific location
    std::cout << "ABOUT TO START\n";
    // bool count_down = false;
    // int cycles_left;
    // while (!count_down || (count_down && (cycles_left == 0)))
    // {

    //   if (count_down)
    //   {
    //     cycles_left--;
    //   }

    //   console.poll();
    //   core.step();
    //   bool run = core.running();
    //   if (!run)
    //   {
    //     count_down = true;
    //   }
    // }

    struct stat hello;
    fstat(0, &hello);

    while (core.running()) { console.poll(); core.step(); }

    if (showStats) core.printStats();


    std::cout << "\n";
  return 0;
}


int main(int argc, char** argv) {

  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  try {
    switch (findMode(argc - 1, argv + 1)) {
      case HARPTOOL_MODE_ASM:    cout << "ASM not supported\n";
      case HARPTOOL_MODE_DISASM: cout << "DISASM not supported\n";
      case HARPTOOL_MODE_EMU:    return emu_main   (argc - 2, argv + 2);
      case HARPTOOL_MODE_LD:     cout << "LD not supported\n";
      case HARPTOOL_MODE_HELP:
      default:
        cout << "Usage:\n" << Help::mainHelp;
        return 0;
    }
  } catch (BadArg ba) {
    cout << "Unrecognized argument \"" << ba.arg << "\".\n";
    return 1;
  }

  return 0;
}
