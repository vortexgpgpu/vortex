/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __ARGS_H
#define __ARGS_H

#include <iostream>
#include <string>
#include <sstream>
#include <map>

namespace HarpTools {
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
    static std::string helpString;
    static std::map<std::string, CommandLineArg *> longArgs;
    static std::map<std::string, CommandLineArg *> shortArgs;
  };

  template <typename T> class CommandLineArgSetter : public CommandLineArg {
  public:
    CommandLineArgSetter(std::string s, std::string l, const char *ht, T &x) :
      CommandLineArg(s, l, ht), x(x) {}
    CommandLineArgSetter(std::string l, const char *ht, T &x) :
      CommandLineArg(l, ht), x(x) {}

    int read(int argc, char **argv) {
      std::istringstream iss(argv[1]);
      iss >> x;
      return 1;
    }
  private:
    T &x;
  };

  class CommandLineArgFlag : public CommandLineArg {
  public:
    CommandLineArgFlag(std::string s, std::string l, const char *ht, bool &x) :
      CommandLineArg(s, l, ht), x(x) { x = false; }
    CommandLineArgFlag(std::string l, const char *ht, bool &x) :
      CommandLineArg(l, ht), x(x) { x = false; }

    int read(int argc, char **argv) { x = true; return 0; }
  private:
    bool &x;
  };
  
};

#endif
