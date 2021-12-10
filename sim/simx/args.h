#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <util.h>

namespace vortex {

struct BadArg { BadArg(std::string s) : arg(s) {} std::string arg; };

class CommandLineArg {
public:
  CommandLineArg(std::string s, std::string l, const char *helpText);
  CommandLineArg(std::string l, const char *helpText);
  virtual int read(int argc, char** argv) = 0;

  static void readArgs(int argc, char **argv);
  static void clearArgs();
  static void showHelp(std::ostream &os);

private:
  static std::string helpString_;
  static std::unordered_map<std::string, CommandLineArg *> longArgs_;
  static std::unordered_map<std::string, CommandLineArg *> shortArgs_;
};

template <typename T> class CommandLineArgSetter : public CommandLineArg {
public:
  CommandLineArgSetter(std::string s, std::string l, const char *ht, T &x) :
    CommandLineArg(s, l, ht), arg_(x) {}

  CommandLineArgSetter(std::string l, const char *ht, T &x) :
    CommandLineArg(l, ht), arg_(x) {}

  int read(int argc, char **argv) {
    __unused(argc);
    std::istringstream iss(argv[1]);
    iss >> arg_;
    return 1;
  }
private:
  T &arg_;
};

class CommandLineArgFlag : public CommandLineArg {
public:
  CommandLineArgFlag(std::string s, std::string l, const char *ht, bool &x) :
    CommandLineArg(s, l, ht), arg_(x) { arg_ = false; }

  CommandLineArgFlag(std::string l, const char *ht, bool &x) :
    CommandLineArg(l, ht), arg_(x) { arg_ = false; }

  int read(int argc, char **argv) { 
    __unused(argc, argv);
    arg_ = true; 
    return 0; 
  }
private:
  bool &arg_;
};
  
}