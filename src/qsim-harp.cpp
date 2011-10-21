/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011 
*******************************************************************************/
#include "include/qsim-harp.h"

#include <iostream>
#include <string>

using namespace Harp;
using namespace std;

Harp::OSDomain* Harp::OSDomain::osDomain(NULL);

Harp::OSDomain::OSDomain(ArchDef &arch, string imgFile) :
  /* TODO: Move the mu to the Cpu. They're sharing a TLB now! */
  arch(arch), mu(4096, arch.getWordSize()), 
  ram(imgFile.c_str(), arch.getWordSize()),
  cpus()
{
  if (osDomain != NULL) {
    cout << "Error: OSDomain is a singleton.";
    exit(1);
  }
  osDomain = this;

  cpus.push_back(Cpu(*this));

  console = new ConsoleMemDevice(arch.getWordSize(), cout, *cpus[0].core);

  mu.attach(ram, 0);
  mu.attach(*console, 1ll<<(arch.getWordSize()*8 - 1));
}

void Harp::OSDomain::connect_console(std::ostream &s) {
  /* For now this does nothing. ConsoleMemDevice is not redirectable. */
}

Harp::OSDomain::Cpu::Cpu(Harp::OSDomain &o) :
  /* TODO: This should support non-word decoders! */
  osd(&o), dec(new WordDecoder(osd->arch)),
  core(new Core(osd->arch, *dec, osd->mu))
{
}

uint64_t Harp::OSDomain::Cpu::run(uint64_t n) {
  uint64_t i;
  for (i = 0; i < n; ++i) core->step();
  return i;
}
