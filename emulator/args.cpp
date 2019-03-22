/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include "include/args.h"

#include <iostream>
#include <string>

using namespace HarpTools;
using std::string;

std::string CommandLineArg::helpString;
std::map<string, CommandLineArg *> CommandLineArg::longArgs;
std::map<string, CommandLineArg *> CommandLineArg::shortArgs;

CommandLineArg::CommandLineArg(string s, string l, const char *helpText)
{
  helpString += helpText;
  longArgs[l] = this;
  shortArgs[s] = this;
}

CommandLineArg::CommandLineArg(string l, const char *helpText) {
  helpString += helpText;
  longArgs[l] = this;
}

void CommandLineArg::readArgs(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::map<string, CommandLineArg *>::iterator 
      s = shortArgs.find(std::string(argv[i])), 
      l = longArgs.find(std::string(argv[i]));

    if (s != shortArgs.end()) {
      i += s->second->read(argc - i, &argv[i]);
    } else if (l != longArgs.end()) {
      i += l->second->read(argc - i, &argv[i]);
    } else {
      throw BadArg(string(argv[i]));
    }
  }
}

void CommandLineArg::clearArgs() {
  shortArgs.clear();
  longArgs.clear();
  helpString = "";
}

void CommandLineArg::showHelp(std::ostream &os) {
  os << helpString;
}
