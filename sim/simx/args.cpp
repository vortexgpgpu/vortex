#include <iostream>
#include <string>
#include "args.h"

using namespace vortex;
using std::string;

std::string CommandLineArg::helpString_;
std::unordered_map<string, CommandLineArg *> CommandLineArg::longArgs_;
std::unordered_map<string, CommandLineArg *> CommandLineArg::shortArgs_;

CommandLineArg::CommandLineArg(string s, string l, const char *helpText) {
  helpString_ += helpText;
  longArgs_[l] = this;
  shortArgs_[s] = this;
}

CommandLineArg::CommandLineArg(string l, const char *helpText) {
  helpString_ += helpText;
  longArgs_[l] = this;
}

void CommandLineArg::readArgs(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::unordered_map<string, CommandLineArg *>::iterator 
      s = shortArgs_.find(std::string(argv[i])), 
      l = longArgs_.find(std::string(argv[i]));

    if (s != shortArgs_.end()) {
      i += s->second->read(argc - i, &argv[i]);
    } else if (l != longArgs_.end()) {
      i += l->second->read(argc - i, &argv[i]);
    } else {
      throw BadArg(string(argv[i]));
    }
  }
}

void CommandLineArg::clearArgs() {
  shortArgs_.clear();
  longArgs_.clear();
  helpString_ = "";
}

void CommandLineArg::showHelp(std::ostream &os) {
  os << helpString_;
}
