/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __QSIM_HARP_H
#define __QSIM_HARP_H

#include "types.h"
#include "core.h"
#include "enc.h"
#include "instruction.h"
#include "mem.h"
#include "obj.h"
#include "archdef.h"

#include <stdint.h>
#include <iostream>
#include <vector>
#include <string>

#include <qsim.h>

namespace Harp {
  class OSDomain {
  public:
    OSDomain(Harp::ArchDef arch, std::string imgFile);

    bool idle(unsigned i) const { return cpus[i].idle(); }
    int get_tid(unsigned i) const { return cpus[i].get_tid(); }
    bool get_prot(unsigned i) const { return cpus[i].get_prot(); }

    int get_n() const { return cpus.size(); }

    uint64_t run(unsigned i, uint64_t n) { return cpus[i].run(n); }
    void connect_console(std::ostream &s);
    void timer_interrupt() { /* TODO: timer convention */ }
    void interrupt(unsigned i, int vec) { cpus[i].interrupt(vec); }
    bool booted(unsigned i) const { return cpus[i].booted(); }
    void save_state(const char* state_file);

    template <typename T>
      void set_atomic_cb
        (T *p, typename Qsim::OSDomain::atomic_cb_obj<T>::atomic_cb_t f)
    {
      atomic_cbs.push_back(new Qsim::OSDomain::atomic_cb_obj<T>(p, f));
    }

    template <typename T>
      void set_inst_cb
        (T* p, typename Qsim::OSDomain::inst_cb_obj<T>::inst_cb_t f)
    { 
      inst_cbs.push_back(new Qsim::OSDomain::inst_cb_obj<T>(p, f));
    }

    template <typename T>
      void set_int_cb
        (T *p, typename Qsim::OSDomain::int_cb_obj<T>::int_cb_t f)
    {
      int_cbs.push_back(new Qsim::OSDomain::int_cb_obj<T>(p, f));
    }

    template <typename T>
      void set_mem_cb
        (T *p, typename Qsim::OSDomain::mem_cb_obj<T>::mem_cb_t f)
    {
      mem_cbs.push_back(new Qsim::OSDomain::mem_cb_obj<T>(p, f));
    }

    template <typename T>
      void set_magic_cb
        (T *p, typename Qsim::OSDomain::magic_cb_obj<T>::magic_cb_t f)
    {
      magic_cbs.push_back(new Qsim::OSDomain::magic_cb_obj<T>(p, f));
    }

    template <typename T>
      void set_io_cb
        (T *p, typename Qsim::OSDomain::io_cb_obj<T>::io_cb_t f)
      { /* Do nothing. We have no separate IO address space. */ }

    template <typename T>
      void set_reg_cb
        (T *p, typename Qsim::OSDomain::reg_cb_obj<T>::reg_cb_t f)
    {
      reg_cbs.push_back(new Qsim::OSDomain::reg_cb_obj<T>(p, f));
    }

    template <typename T> void mem_rd(T& d, uint64_t paddr);
    template <typename T> void mem_rd_virt(unsigned i, T& d, uint64_t vaddr);
    template <typename T> void mem_wr(T& d, uint64_t paddr);
    template <typename T> void mem_wr_virt(unsigned i, T& d, uint64_t vaddr);

  private:
    bool do_atomic(unsigned c) {
      bool rval(false);
      for (unsigned i = 0; i < atomic_cbs.size(); ++i)
        if (atomic_cbs[i](c)) rval = true;
      return rval;
    }

    void do_inst(unsigned c, uint64_t va, uint64_t pa, uint8_t l, const char *b,
                 enum inst_type t)
    {
      for (unsigned i = 0; i < inst_cbs.size(); ++i)
        inst_cbs[i](c, va, pa, l, b, t);
    }

    void do_int(unsigned c, int v) {
      for (unsigned i = 0; i < int_cbs.size(); ++i)
        int_cbs[i](c, v);
    }

    void do_mem(unsigned c, uint64_t va, uint64_t pa, uint8_t s, bool w) {
      for (unsigned i = 0; i < mem_cbs.size(); ++i)
        mem_cbs[i](c, va, pa, s, w);
    }

    void do_magic(unsigned c, uint64_t r0) {
      bool rval(false);
      for (unsigned i = 0; i < magic_cbs.size(); ++i)
        if (magic_cbs[i](c, r0)) rval = true;
      return rval;
    }

    void do_reg(unsigned c, int r, uint8_t s, bool w) {
      for (unsigned i = 0; i < reg_cbs.size(); ++i)
        reg_cbs[i](c, r, s, w);
    }

    struct Cpu {
      Cpu(Harp::OSDomain &osd);

      bool idle() const { return false; }
      int get_tid() const { return 0; }
      bool get_prot() const { return core.supervisorMode(); }
      uint64_t run(uint64_t n);
      void interrupt(int vec) { core.interrupt(vec); }
      bool booted() { return core.running(); }

      Harp::Decoder *dec;
      Harp::Core core;
      Harp::OSDomain &osd;
    };

    Harp::ArchDef arch;

    Harp::MemoryUnit mu;
    Harp::RamMemDevice ram;
    Harp::ConsoleMemDevice *console;

    std::vector <Harp::OSDomain::Cpu> cpus;

    std::vector <Harp::OSdomain::atomic_cb_obj_base*> atomic_cbs;
    std::vector <Harp::OSdomain::inst_cb_obj_base*>   inst_cbs;
    std::vector <Harp::OSdomain::int_cb_obj_base*>    int_cbs;
    std::vector <Harp::OSdomain::mem_cb_obj_base*>    mem_cbs;
    std::vector <Harp::OSdomain::magic_cb_obj_base*>  magic_cbs;
    std::vector <Harp::OSdomain::reg_cb_obj_base*>    reg_cbs;
  };
};
#endif
