#include <iostream>
#include <iomanip>
#include "include/qsim-harp.h"

class CallbackAdaptor {
public:
  void inst_cb(int c, uint64_t v, uint64_t p, uint8_t l, const uint8_t *b,
               enum inst_type t)
  {
    std::cout << "Inst @ 0x" << std::hex << v << "(0x" << p << ")\n";
  }
} cba;

int main(int argc, char** argv) {
  Harp::ArchDef arch("8w32/32/8");
  Harp::OSDomain osd(arch, std::string("../test/sieve.bin"));

  osd.set_inst_cb(&cba, &CallbackAdaptor::inst_cb);

  for (unsigned i = 0; i < 1000; ++i) {
    osd.run(0, 1);
  }

  return 0;
}
