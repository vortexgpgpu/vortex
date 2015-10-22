/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __HELP_H
#define __HELP_H

/* Help messages. */
namespace HarpTools {
  namespace Help {
    const char *mainHelp = 
      "--help, -h, no arguments\n"
      "  Print this message.\n"
      "-E, --emu; -A, --asm; -L, --ld; -D, --disasm\n"
      "  Invoke the emulator, assembler, linker, and disassembler, "
        "respectively.\n"
      "<mode> --help\n"
      "  Display contextual help.\n",
      *emuHelp = "HARP Emulator command line arguments:\n"
                  "  -c, --core <filename>    RAM image\n"
                  "  -a, --arch <arch string> Architecture string\n"
                  "  -s, --stats              Print stats on exit.\n"
                  "  -b, --basic              Disable virtual memory.\n"
                  "  -i, --batch              Disable console input.\n",
      *asmHelp = "HARP Assembler command line arguments:\n"
                  "  -a, --arch <arch string>\n"
                  "  -o, --output <filename>\n",
      *ldHelp = "HARP Linker command line arguments:\n"
                "  -o, --output <filename>\n"
                "  -a, --arch <filename>\n"
                "  -f, --format <foramt string>\n"
                "  --offset <bytes>\n",
      *disasmHelp = "HARP Disassembler command line arguments:\n"
                     "  -a, --arch <arch string> Architecture string.\n"
                     "  -o, --output <filename>  Output filename.\n";
  };
};
#endif
