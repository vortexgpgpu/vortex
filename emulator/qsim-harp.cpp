/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011 
*******************************************************************************/
#include "include/qsim-harp.h"

#include <iostream>
#include <string>

using namespace Harp;
using namespace Qsim;
using namespace std;

Harp::OSDomain* Harp::OSDomain::osDomain(NULL);

Harp::OSDomain::OSDomain(ArchDef &archref, string imgFile) :
  /* TODO: Move the mu to the Cpu. They're sharing a TLB now! */
  arch(archref), mu(4096, arch.getWordSize()), 
  ram(imgFile.c_str(), arch.getWordSize()),
  cpus(0)
{
  if (osDomain != NULL) {
    cout << "Error: OSDomain is a singleton.";
    std::abort();
  }
  osDomain = this;

  std::cout << "Constructing an OSDomain with archref, " << archref.getNPRegs() << '\n';

  std::cout << "Pushing back a Cpu in OSDomain constructor.\n";
  cpus.push_back(Cpu(*this));

  console = new ConsoleMemDevice(arch.getWordSize(), cout, *cpus[0].core);

  mu.attach(ram, 0);
  mu.attach(*console, 1ll<<(arch.getWordSize()*8 - 1));
}

void Harp::OSDomain::connect_console(std::ostream &s) {
  /* For now this does nothing. ConsoleMemDevice is not redirectable. */
  std::cout << "in connect_console\n";
}

Harp::OSDomain::Cpu::Cpu(Harp::OSDomain &o) :
  /* TODO: This should support non-word decoders! */
  osd(&o), dec(new WordDecoder(osd->arch)),
  core(new Core(osd->arch, *dec, osd->mu))
{
  std::cout << "Constructing a new Cpu.\n";
}

uint64_t Harp::OSDomain::Cpu::run(uint64_t n) {
  uint64_t i;
  std::cout << "pc=0x" << std::hex << core->pc << ", " << std::dec << sizeof(*core) << '\n';
  //osd->console->poll();
  for (i = 0; i < n; ++i) core->step();
  return i;
}
