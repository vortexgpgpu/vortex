#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <rvfloats.h>
#include "warp.h"
#include "instr.h"
#include "core.h"
#include "execute.h"
#include "decode.h"

using namespace vortex;

template <typename T, typename R>
class Add {
  public:
    static R apply(T first, T second) {
      return (R)first + (R)second;
    }
    static std::string name() {return "Add";}
};

template <typename T, typename R>
class Sub {
  public:
    static R apply(T first, T second) {
      return (R)second - (R)first;
    }
    static std::string name() {return "Sub";}
};

template <typename T, typename R>
class Rsub {
  public:
    static R apply(T first, T second) {
      return first - second;
    }
    static std::string name() {return "Rsub";}
};

template <typename T, typename R>
class Mul {
  public:
    static R apply(T first, T second) {
      return (R)first * (R)second;
    }
    static std::string name() {return "Mul";}
};

template <typename T, typename R>
class Mulh {
  public:
    static R apply(T first, T second) {
      __int128_t first_ext = sext((__int128_t)second, (sizeof(T) * 8));
      __int128_t second_ext = sext((__int128_t)first, (sizeof(T) * 8));
      return (first_ext * second_ext) >> (sizeof(T) * 8);
    }
    static std::string name() {return "Mulh";}
};

template <typename T, typename R>
class Mulhu {
  public:
    static R apply(T first, T second) {
      return ((__uint128_t)first * (__uint128_t)second) >> (sizeof(T) * 8);
    }
    static std::string name() {return "Mulhu";}
};

template <typename T, typename R>
class Min {
  public:
    static R apply(T first, T second) {
      return std::min(first, second);
    }
    static std::string name() {return "Min";}
};

template <typename T, typename R>
class Max {
  public:
    static R apply(T first, T second) {
      return std::max(first, second);
    }
    static std::string name() {return "Max";}
};

template <typename T, typename R>
class And {
  public:
    static R apply(T first, T second) {
      return first & second;
    }
    static std::string name() {return "And";}
};

template <typename T, typename R>
class Or {
  public:
    static R apply(T first, T second) {
      return first | second;
    }
    static std::string name() {return "Or";}
};

template <typename T, typename R>
class Xor {
  public:
    static R apply(T first, T second) {
      return first ^ second;
    }
    static std::string name() {return "Xor";}
};

template <typename T, typename R>
class Sll {
  public:
    static R apply(T first, T second) {
      // Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.
      return second << (first & (sizeof(T) * 8 - 1));
    }
    static std::string name() {return "Sll";}
};

template <typename T, typename R>
class SrlSra {
  public:
    static R apply(T first, T second) {
      // Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.
      return second >> (first & (sizeof(T) * 8 - 1));
    }
    static std::string name() {return "SrlSra";}
};

template <typename T, typename R>
class Fmin {
  public:
    static R apply(T first, T second) {
      // ignoring rounding modes for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_fmin_s(first, second, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fmin_d(first, second, &fflags);
      } else {
        std::cout << "Fmin only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fmin";}
};

template <typename T, typename R>
class Fmax {
  public:
    static R apply(T first, T second) {
      // ignoring rounding modes for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_fmax_s(first, second, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fmax_d(first, second, &fflags);
      } else {
        std::cout << "Fmax only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fmax";}
};

template <typename T, typename R>
class Fcvt {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      // ignoring rounding mode for now
      uint32_t frm = 0;
      if (sizeof(T) == 4) {
        switch (first) {
          case 0b001: // vfcvt.x.f.v
            return rv_ftoi_s(second, frm, &fflags);
          case 0b011: // vfcvt.f.x.v
            return rv_itof_s(second, frm, &fflags);
          default:
            std::cout << "Fcvt has unsupported value for first: " << first << std::endl;
            std::abort();
        }
      } else if (sizeof(T) == 8) {
        switch (first) {
          case 0b001: // vfcvt.x.f.v
            return rv_ftoi_d(second, frm, &fflags);
          case 0b011: // vfcvt.f.x.v
            return rv_itof_d(second, frm, &fflags);
          default:
            std::cout << "Fcvt has unsupported value for first: " << first << std::endl;
            std::abort();
        }
      } else {
        std::cout << "Fcvt only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fcvt";}
};

template <typename T, typename R>
class Funary1 {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      // ignoring rounding mode for now
      uint32_t frm = 0;
      if (sizeof(T) == 4) {
        switch (first) {
          case 0b100: // vfrsqrt7.v
            return rv_frsqrt7_s(second, frm, &fflags);
          case 0b101: // vfrec7.v
            return rv_frecip7_s(second, frm, &fflags);
          default:
            std::cout << "Funary1 has unsupported value for first: " << first << std::endl;
            std::abort();
        }
      } else if (sizeof(T) == 8) {
        switch (first) {
          case 0b100: // vfrsqrt7.v
            return rv_frsqrt7_d(second, frm, &fflags);
          case 0b101: // vfrec7.v
            return rv_frecip7_d(second, frm, &fflags);
          default:
            std::cout << "Funary1 has unsupported value for first: " << first << std::endl;
            std::abort();
        }
      } else {
        std::cout << "Funary1 only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Funary1";}
};

template <typename DT>
void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, std::vector<Byte> mask, uint32_t vl, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    uint8_t emask = *(uint8_t *)(mask.data() + i / 8);
    uint8_t value = (emask >> (i % 8)) & 0x1;
    DP(1, "VLE masking enabled: " << vmask << " mask element: " << +value);
    if (!vmask && value == 0) continue;
    
    auto &vd = vreg_file.at((rdest + (i / (VLEN / vsew))) % 32);
    Word mem_addr = ((rsdata[0][0].i) & 0xFFFFFFFC) + (i * vsew / 8);
    Word mem_data = 0;
    core_->dcache_read(&mem_data, mem_addr, vsew / 8);
    DP(1, "Loading data " << mem_data << " from: " << mem_addr << " to vec reg: " << (rdest + (i / (VLEN / 8))) % 32);
    DT *result_ptr = (DT *)(vd.data() + ((i % (VLEN / vsew)) * vsew / 8));
    DP(1, "Previous data: " << +(*result_ptr));
    *result_ptr = (DT) mem_data;
  }
}

void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
    case 8:
      loadVector<uint8_t>(vreg_file, core_, rsdata, rdest, mask, vl, vmask);
      break;
    case 16:
      loadVector<uint16_t>(vreg_file, core_, rsdata, rdest, mask, vl, vmask);
      break;
    case 32:
      loadVector<uint32_t>(vreg_file, core_, rsdata, rdest, mask, vl, vmask);
      break;
    case 64:
      loadVector<uint64_t>(vreg_file, core_, rsdata, rdest, mask, vl, vmask);
      break;
    default:
      std::cout << "Failed to execute VLE for vsew: " << vsew << std::endl;
      std::abort();
  }
}

template <typename DT>
void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, std::vector<Byte> mask, uint32_t vl, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    uint8_t emask = *(uint8_t *)(mask.data() + i / 8);
    uint8_t value = (emask >> (i % 8)) & 0x1;
    DP(1, "VSE masking enabled: " << vmask << " mask element: " << +value);
    if (!vmask && value == 0) continue;

    uint64_t mem_addr = rsdata[0][0].i + (i * vsew / 8);
    auto &vr = vreg_file.at((rsrc3 + (i / (VLEN / vsew))) % 32);      
    uint32_t mem_data = 0;
    int n = (vsew / 8);

    for (int j = 0; j < n; j++){
        mem_data += (*(Byte *)(vr.data() + j + ((i % (VLEN / vsew)) * vsew / 8))) * ((uint32_t)1 << (j * 8));
    }
    DP(1, "Storing: " << std::hex << mem_data << " at: " << mem_addr << " from vec reg: " << (rsrc3 + (i / (VLEN / vsew))) % 32);
    core_->dcache_write(&mem_data, mem_addr, vsew / 8);
  }
}

void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
    case 8:
      storeVector<uint8_t>(vreg_file, core_, rsdata, rsrc3, mask, vl, vmask);
      break;
    case 16:
      storeVector<uint16_t>(vreg_file, core_, rsdata, rsrc3, mask, vl, vmask);
      break;
    case 32:
      storeVector<uint32_t>(vreg_file, core_, rsdata, rsrc3, mask, vl, vmask);
      break;
    case 64:
      storeVector<uint64_t>(vreg_file, core_, rsdata, rsrc3, mask, vl, vmask);
      break;
    default:
      std::cout << "Failed to execute VSE for vsew: " << vsew << std::endl;
      std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, std::vector<Byte> mask, uint32_t vl, uint32_t vmask)
{
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    auto& vr2 = vreg_file.at((rsrc0 + (i / (VLEN / vsew))) % 32);
    auto& vd = vreg_file.at((rdest + (i / (VLEN / vsew))) % 32);
    uint8_t emask = *(uint8_t *)(mask.data() + i / 8);
    uint8_t value = (emask >> (i % 8)) & 0x1;
    DP(1, "VI/VX masking enabled: " << vmask << " mask element: " << +value);
    if (!vmask && value == 0) continue;
    
    DT second = *(DT *)(vr2.data() + (i % (VLEN / vsew)) * vsew / 8);
    DT result = OP<DT, DT>::apply(first, second);
    DP(1, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    *(DT *)(vd.data() + (i % (VLEN / vsew)) * vsew / 8) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32>
void vector_op_vix(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vix<OP, DT8>(src1, vreg_file, rsrc0, rdest, mask, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vix<OP, DT16>(src1, vreg_file, rsrc0, rdest, mask, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vix<OP, DT32>(src1, vreg_file, rsrc0, rdest, mask, vl, vmask);
  } else {
    std::cout << "Failed to execute VI/VX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, std::vector<Byte> mask, uint32_t vl, uint32_t vmask)
{
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    auto& vr1 = vreg_file.at((rsrc0 + (i / (VLEN / vsew))) % 32);
    auto& vr2 = vreg_file.at((rsrc1 + (i / (VLEN / vsew))) % 32);
    auto& vd = vreg_file.at((rdest + (i / (VLEN / vsew))) % 32);
    uint8_t emask = *(uint8_t *)(mask.data() + i / 8);
    uint8_t value = (emask >> (i % 8)) & 0x1;
    DP(1, "VI/VX masking enabled: " << vmask << " mask element: " << +value);
    if (!vmask && value == 0) continue;

    DT first  = *(DT *)(vr1.data() + (i % (VLEN / vsew)) * vsew / 8);
    DT second = *(DT *)(vr2.data() + (i % (VLEN / vsew)) * vsew / 8);
    DT result = OP<DT, DT>::apply(first, second);
    DP(1, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    *(DT *)(vd.data() + (i % (VLEN / vsew)) * vsew / 8) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vv<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, mask, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vv<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, mask, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vv<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, mask, vl, vmask);
  } else {
    std::cout << "Unhandled sew of: " << vsew << std::endl;
    std::abort();
  }
}

void executeVector(const Instr &instr, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata, std::vector<std::vector<Word>> &ireg_file_, std::vector<std::vector<Byte>> &vreg_file_, vtype &vtype_, uint32_t &vl_, uint32_t warp_id_, ThreadMask &tmask_, uint32_t num_threads) {
  auto func3  = instr.getFunc3();
  auto func6  = instr.getFunc6();

  auto rdest  = instr.getRDest();
  auto rsrc0  = instr.getRSrc(0);
  auto rsrc1  = instr.getRSrc(1);
  auto immsrc = sext((Word)instr.getImm(), 32);
  auto vmask  = instr.getVmask();
  
    uint32_t VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
    switch (func3) {
    case 0: { // vector - vector
        switch (func6) { 
          case 0: { // vadd.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Add, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 2: { // vsub.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Sub, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 4: { // vminu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Min, uint8_t, uint16_t, uint32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 5: { // vmin.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Min, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 6: { // vmaxu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Max, uint8_t, uint16_t, uint32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 7: { // vmax.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Max, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 9: { // vand.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<And, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 10: { // vor.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Or, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 11: { // vxor.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Xor, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 37: { // vsll.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Sll, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 40: { // vsrl.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<SrlSra, uint8_t, uint16_t, uint32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 41: { // vsra.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<SrlSra, int8_t, int16_t, int32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          default:
            std::cout << "Unrecognised vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
            std::abort();
        } 
      } break;
    case 1: { // float vector - vector
        switch (func6) {
          case 4: { // vfmin.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Fmin, uint8_t, uint16_t, uint32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 6: { // vfmax.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vv<Fmax, uint8_t, uint16_t, uint32_t>(vreg_file_, rsrc0, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 18: { // vfcvt.f.x.v, vfcvt.x.f.v
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vix<Fcvt, uint8_t, uint16_t, uint32_t>(rsrc0, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 19: { // vfrec7.v, vfrsqrt7.v
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto& mask = vreg_file_.at(0);
              vector_op_vix<Funary1, uint8_t, uint16_t, uint32_t>(rsrc0, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
            }
          } break;
          default:
            std::cout << "Unrecognised float vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
            std::abort();
        }
      } break;
    case 3: { // vector - immidiate
      switch (func6) {
      case 0: { // vadd.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<Add, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 3: { // vrsub.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<Rsub, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 9: { // vand.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<And, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 10: { // vor.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<Or, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 11: { // vxor.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<Xor, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 37: { // vsll.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<Sll, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 40: { // vsrl.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 41: { // vsra.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          auto &mask = vreg_file_.at(0);
          vector_op_vix<SrlSra, int8_t, int16_t, int32_t>(immsrc, vreg_file_, rsrc0, rdest, mask, vtype_.vsew, vl_, vmask);
        }
      } break;
      default:
        std::cout << "Unrecognised vector - immidiate instruction func3: " << func3 << " func6: " << func6 << std::endl;
        std::abort();
      }
    } break;
    case 2: {
      switch (func6) {
      case 24: { 
        // vmandnot
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }            
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 25: {
        // vmand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 26: {
        // vmor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 27: { 
        // vmxor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 28: {
        // vmornot
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 29: {
        // vmnand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 30: {
        // vmnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 31: {
        // vmxnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 45: {
        // vmacc
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      }
    } break;
    case 4:{
      switch (func6){
        case 0: { // vadd.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Add, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 2: { // vsub.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Sub, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 3: { // vrsub.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Rsub, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 4: { // vminu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Min, uint8_t, uint16_t, uint32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 5: { // vmin.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Min, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 6: { // vmaxu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Max, uint8_t, uint16_t, uint32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 7: { // vmax.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Max, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 9: { // vand.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<And, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 10: { // vor.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Or, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 11: { // vxor.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Xor, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 37: { // vsll.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<Sll, uint8_t, uint16_t, uint32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 40: { // vsrl.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 41: { // vsra.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            auto& mask = vreg_file_.at(0);
            vector_op_vix<SrlSra, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        default:
          std::cout << "Unrecognised vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
          std::abort();
      }
    } break;
    case 6: {
      switch (func6) {
        case 36: { // vmulhu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            auto &mask = vreg_file_.at(0);
            vector_op_vix<Mulhu, uint8_t, uint16_t, uint32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 37: { // vmul.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            auto &mask = vreg_file_.at(0); 
            vector_op_vix<Mul, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 39: { // vmulh.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            auto &mask = vreg_file_.at(0);
            vector_op_vix<Mulh, int8_t, int16_t, int32_t>(src1, vreg_file_, rsrc1, rdest, mask, vtype_.vsew, vl_, vmask);
          }
        } break;
        default:
          std::cout << "Unrecognised vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
          std::abort();
      }
    } break;
    case 7: {
      uint32_t vma = instr.getVma();
      uint32_t vta = instr.getVta();
      uint32_t vsewO = instr.getVsewO();
      uint32_t vsew = instr.getVsew();
      uint32_t vlmul = instr.getVlmul();

      if(!instr.hasZimm()){ // vsetvl
        uint32_t zimm = rsdata[0][1].u;
        vlmul = zimm & mask_v_lmul;
        vsewO = (zimm >> shift_v_sew) & mask_v_sew;
        vsew = 1 << (3 + vsewO);
        vta = (zimm >> shift_v_ta) & mask_v_ta;
        vma = (zimm >> shift_v_ma) & mask_v_ma;
      }

      bool negativeLmul = vlmul >> 2;
      uint32_t vlenDividedByLmul = VLEN >> (0x8 - vlmul);
      uint32_t vlenMultipliedByLmul = VLEN << vlmul;
      uint32_t vlenTimesLmul = negativeLmul ? vlenDividedByLmul : vlenMultipliedByLmul;
      VLMAX = vlenTimesLmul / vsew;
      vtype_.vill  = vsew > XLEN || VLMAX < 8;

      uint32_t s0 = instr.getImm(); // vsetivli
      if (!instr.hasImm()) { // vsetvli/vsetvl
        s0 = rsdata[0][0].u;
      }

      DP(1, "Vset(i)vl(i) - vill: " << +vtype_.vill << " vma: " << vma << " vta: " << vta << " lmul: " << vlmul << " sew: " << vsew << " s0: " << s0 << " VLMAX: " << VLMAX);

      if (s0 <= VLMAX) {
        vl_ = s0;
      } else if (s0 >= (2 * VLMAX)) {
        vl_ = VLMAX;
      }

      if (vtype_.vill) {
        core_->set_csr(VX_CSR_VTYPE, 1 << 31, 0, warp_id_);
        vtype_.vma = 0;
        vtype_.vta = 0;
        vtype_.vsew  = 0;
        vtype_.vlmul = 0;
        core_->set_csr(VX_CSR_VL, 0, 0, warp_id_);
        rddata[0].i = vl_;
      } else {
        vtype_.vma = vma;
        vtype_.vta = vta;
        vtype_.vsew  = vsew;
        vtype_.vlmul = vlmul;
        uint32_t vtype = vlmul;
        vtype |= vsewO << shift_v_sew;
        vtype |= vta << shift_v_ta;
        vtype |= vma << shift_v_ma;
        core_->set_csr(VX_CSR_VTYPE, vtype, 0, warp_id_);
        core_->set_csr(VX_CSR_VL, vl_, 0, warp_id_);
        rddata[0].i = vl_;
      }
    }
    core_->set_csr(VX_CSR_VSTART, 0, 0, warp_id_);
    break;
    default:
      std::cout << "Unrecognised vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
}