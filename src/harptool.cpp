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

int asm_main(int argc, char **argv) {
  // string archString("8w32/32/8/8"), outFileName("a.out.HOF"),
  //        inFileName(argv[argc-1]);
  // bool showHelp;  

  // /* Get command line arguments. */
  // CommandLineArgFlag          fh("-h", "--help", "", showHelp);
  // CommandLineArgSetter<string>fo("-o", "--output", "", outFileName);
  // CommandLineArgSetter<string>fa("-a", "--arch", "", archString);

  // CommandLineArg::readArgs(argc-1, argv);

  // if (showHelp || argc == 0) {
  //   cout << Help::asmHelp;
  //   exit(0);
  // }

  // ArchDef arch(archString);

  // D(0, "Created ArchDef for " << string(arch));

  // /* Create an appropriate encoder. */
  // Encoder *enc;
  // switch (arch.getEncChar()) {
  //   case 'b': enc = new ByteEncoder(arch); break;
  //   case 'w': enc = new WordEncoder(arch); break;
  //   defaulet:
  //     cout << "Unknown encoding type, \"" << arch.getEncChar() << "\"\n";
  //     exit(1);
  // }

  // /* Open files. */
  // if (outFileName == "") {
  //   cout << "HARP Assembler: No output filename given.\n";
  //   exit(1);
  // }

  // ifstream asmFile(inFileName.c_str());
  // ofstream outFile(outFileName.c_str());

  // if (!asmFile) {
  //   cout << "Could not open \"" << inFileName << "\" for reading.\n";
  //   exit(1);
  // }

  // if (!outFile) {
  //   cout << "Could not open \"" << outFileName << "\" for writing.\n";
  //   exit(1);
  // }

  // /* Read an Obj from the assembly file. */
  // D(0, "Passing AsmReader ArchDef: " << string(arch));
  // AsmReader ar(arch);
  // Obj *o = ar.read(asmFile);

  // /* Encode the text chunks read from the assembly file. */
  // for (Size j = 0; j < o->chunks.size(); j++) {
  //   Chunk *&c = o->chunks[j];
  //   TextChunk *tc;
  //   DataChunk *dc;
  //   if ((tc = dynamic_cast<TextChunk*>(c)) != NULL) {
  //     /* Encode it. */
  //     dc = new DataChunk(tc->name);
  //     enc->encodeChunk(*dc, *tc);

  //     /* Delete the text chunk. */
  //     delete tc;

  //     /* Do the switch. */
  //     c = dc;
  //   }
  // }
  // asmFile.close();
  // delete enc;

  // /* Write a HOF binary. */
  // D(0, "Creating a HOFWriter, passing it ArchDef: " << string(arch));
  // HOFWriter hw(arch);
  // hw.write(outFile, *o);
  // outFile.close();

  // delete o;

  return 0;
}

int disasm_main(int argc, char **argv) {
  // bool showHelp;
  // string outFileName("a.out.s"), archString("8w32/32/8/8");
  

  // /* Get command line arguments. */
  // CommandLineArgFlag          fh("-h", "--help", "", showHelp);
  // CommandLineArgSetter<string>fa("-a", "--arch", "", archString);
  // CommandLineArgSetter<string>fo("-o", "--output", "", outFileName);

  // if (argc != 0) CommandLineArg::readArgs(argc-1, argv);

  // if (argc == 0 || showHelp) {
  //   cout << Help::disasmHelp;
  //   exit(0);
  // }

  // ifstream objFile(argv[argc-1]);
  // ofstream outFile(outFileName.c_str());
  // ArchDef arch(archString);

  // if (!objFile) {
  //   cout << "Disassembler could not open \"" << argv[argc-1] 
  //        << "\" for reading.\n";
  //   exit(1);
  // }

  // if (!outFile) {
  //   cout << "Disassembler could not open \"" << outFileName 
  //        << "\" for output.\n";
  //   exit(1);
  // }

  // HOFReader hr(arch);
  // Obj *o = hr.read(objFile);
  // objFile.close();
  // Decoder *dec;

  // switch (arch.getEncChar()) {
  //   case 'b': dec = new ByteDecoder(arch); break;
  //   case 'w': dec = new WordDecoder(arch); break;
  //   default:
  //     cout << "Unrecognized encoding character for disassembler.\n";
  //     exit(1);
  // }

  // /* Decode the chunks read from the object. */
  // for (Size j = 0; j < o->chunks.size(); j++) {
  //   Chunk *&c = o->chunks[j];
  //   if (c->flags & EX_USR) {
  //     TextChunk *tc;
  //     DataChunk *dc;
  //     if ((dc = dynamic_cast<DataChunk*>(c)) != NULL) {
  //       TextChunk *tc = new TextChunk(dc->name);
  //       dec->decodeChunk(*tc, *dc);
  //       delete dc;
  //       c = tc;
  //     }
  //   }
  // }
  // delete dec;

  // AsmWriter aw(arch);
  // aw.write(outFile, *o);
  // outFile.close();

  // delete o;

  return 0;
}

int emu_main(int argc, char **argv) {
  string archString("8w32/32/8/8"), imgFileName("a.out.bin");
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
    case 'b': dec = new ByteDecoder(arch); break;
    case 'w': dec = new WordDecoder(arch); break;
    case 'r': dec = new WordDecoder(arch); break;
    default:
      cout << "Unrecognized decoder type: '" << arch.getEncChar() << "'.\n";
      return 1;
  }

  MemoryUnit mu(4096, arch.getWordSize(), basicMachine);
  Core core(arch, *dec, mu/*, ID in multicore implementations*/);

  RamMemDevice mem(imgFileName.c_str(), arch.getWordSize());
  ConsoleMemDevice console(arch.getWordSize(), cout, core, batch);
  mu.attach(mem,     0);
  mu.attach(console, 1ll<<(arch.getWordSize()*8 - 1));

  while (core.running()) { console.poll(); core.step(); }

  if (showStats) core.printStats();

  return 0;
}

int ld_main(int argc, char **argv) {
  // bool showHelp, mustResolveRefs(true);
  // string outFileName("a.out.bin"), archString("8w32/32/8/8"),
  //        formatString("bin");
  // Size nObjects;
  // Addr binOffset(0);

  // /* Get command line arguments. */
  // CommandLineArgFlag          fh("-h", "--help", "", showHelp);
  // CommandLineArgSetter<string>fa("-a", "--arch", "", archString);
  // CommandLineArgSetter<string>ff("-f", "--format", "", formatString);
  // CommandLineArgSetter<Addr>  foffset("--offset", "", binOffset);
  // CommandLineArgSetter<string>fo("-o", "--output", "", outFileName);

  // int firstInput(0), newArgc;
  // for (size_t i = 0; i < argc; i++) {
  //   if (*(argv[i]) != '-') { firstInput = i; newArgc = i; break; }
  //   else if (string(argv[i]) == "--") { firstInput = i+1; newArgc = i; break; }
  //   else i++; /* Skip both the switch and its argument. */
  // }
  // nObjects = argc - firstInput;

  // if (argc != 0) CommandLineArg::readArgs(newArgc, argv);

  // if (argc == 0 || showHelp) {
  //   cout << Help::ldHelp;
  //   exit(0);
  // }

  // if (firstInput == argc) {
  //   cout << "Linker: no input files given.\n";
  //   exit(1);
  // }

  // ArchDef arch(archString);

  // /* Read all of the objects, assign addresses to their chunks, and place them
  //    in an address map.*/
  // vector<Obj *> objects(nObjects);
  // vector<DataChunk*> chunks;
  // map<string, Addr> gChunkMap;
  // Addr nextOffset(binOffset);

  // for (Size i = 0; i < nObjects; i++) {
  //   map <string, Addr> lChunkMap;

  //   /* Read the object. */
  //   HOFReader hr(arch);
  //   ifstream objFile(argv[firstInput + i]);
  //   if (!objFile) {
  //     cout << "Could not open \"" << argv[firstInput + i] 
  //          << "\" for reading.\n";
  //     exit(1);
  //   }
  //   objects[i] = hr.read(objFile);

  //   /* Assign addresses to chunks. */
  //   Obj &obj = *objects[i];
  //   for (Size j = 0; j < obj.chunks.size(); j++) {
  //     DataChunk *c = dynamic_cast<DataChunk*>(obj.chunks[j]);
  //     if (c->alignment != 0 && nextOffset % c->alignment)
  //       nextOffset += c->alignment - (nextOffset % c->alignment);
  //     c->bind(nextOffset);
  //     chunks.push_back(c);
  //     if (obj.chunks[j]->name != "") {
  //       if (c->isGlobal()) gChunkMap[c->name] = nextOffset;
  //       else               lChunkMap[c->name] = nextOffset;
  //     }
  //     nextOffset += (c->size);
  //   }

  //   /* Resolve local references. */
  //   for (Size i = 0; i < obj.chunks.size(); i++) {
  //     DataChunk *dc = dynamic_cast<DataChunk*>(obj.chunks[i]);
  //     for (Size j = 0; j < dc->refs.size(); j++) {
  //       Ref &ref = *(dc->refs[j]);
  //       if (lChunkMap.find(dc->refs[j]->name) != lChunkMap.end()) {
  //         dc->refs[j]->bind(lChunkMap[dc->refs[j]->name],
  //                           dc->address + dc->refs[j]->ibase);
  //       }
  //     }
  //   }
  // }

  // /* Resolve references. */
  // for (Size i = 0; i < chunks.size(); i++) {
  //   DataChunk *dc = chunks[i];
  //   for (Size j = 0; j < dc->refs.size(); j++) {
  //     Ref &ref = *(dc->refs[j]);
  //     if (!ref.bound && (gChunkMap.find(ref.name) != gChunkMap.end())) {
  //       ref.bind(gChunkMap[ref.name], dc->address + ref.ibase);
  //     } else if (!ref.bound && mustResolveRefs) {
  //       cout << "Undefined symbol: \"" << ref.name << "\"\n";
  //       exit(1);
  //     }
  //   }
  // }  

  // /* Write out the chunks. */
  // ofstream outFile(outFileName.c_str());
  // for (Size i = 0; i < chunks.size(); i++) {
  //   if (outFile.tellp() > chunks[i]->address - binOffset) {
  //     cout << "Linker internal error. Wrote past next chunk address.\n";
  //     exit(1);
  //   }
  //   while (outFile.tellp() < chunks[i]->address - binOffset) outFile.put('\0');
  //   outFile.seekp(chunks[i]->address - binOffset);
  //   outFile.write((char*)&chunks[i]->contents[0], chunks[i]->contents.size());
  // }

  // /* Clean up. */
  // for (Size i = 0; i < nObjects; i++) delete objects[i];

  return 0;
}

int main(int argc, char** argv) {
  try {
    switch (findMode(argc - 1, argv + 1)) {
      case HARPTOOL_MODE_ASM:    return asm_main   (argc - 2, argv + 2);
      case HARPTOOL_MODE_DISASM: return disasm_main(argc - 2, argv + 2);
      case HARPTOOL_MODE_EMU:    return emu_main   (argc - 2, argv + 2);
      case HARPTOOL_MODE_LD:     return ld_main    (argc - 2, argv + 2);
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
