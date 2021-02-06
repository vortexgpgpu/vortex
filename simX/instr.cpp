#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "instr.h"

using namespace vortex;

void Instr::setVlmul(Word lmul) { 
  vlmul_ = std::pow(2, lmul);
}

void Instr::setVsew(Word sew) { 
  vsew_ = std::pow(2, 3+sew); 
}

void Instr::setVediv(Word ediv) { 
  vediv_ = std::pow(2,ediv);
}