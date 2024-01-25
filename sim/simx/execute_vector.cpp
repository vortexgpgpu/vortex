#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <rvfloats.h>
#include "warp.h"
#include "instr.h"
#include "core.h"
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
class Eq {
  public:
    static R apply(T first, T second) {
      return first == second;
    }
    static std::string name() {return "Eq";}
};

template <typename T, typename R>
class Ne {
  public:
    static R apply(T first, T second) {
      return first != second;
    }
    static std::string name() {return "Ne";}
};

template <typename T, typename R>
class Lt {
  public:
    static R apply(T first, T second) {
      return first > second;
    }
    static std::string name() {return "Lt";}
};

template <typename T, typename R>
class Le {
  public:
    static R apply(T first, T second) {
      return first >= second;
    }
    static std::string name() {return "Le";}
};

template <typename T, typename R>
class Gt {
  public:
    static R apply(T first, T second) {
      return first < second;
    }
    static std::string name() {return "Gt";}
};

template <typename T, typename R>
class AndNot {
  public:
    static R apply(T first, T second) {
      return second & ~first;
    }
    static std::string name() {return "AndNot";}
};

template <typename T, typename R>
class OrNot {
  public:
    static R apply(T first, T second) {
      return second | ~first;
    }
    static std::string name() {return "OrNot";}
};

template <typename T, typename R>
class Nand {
  public:
    static R apply(T first, T second) {
      return ~(second & first);
    }
    static std::string name() {return "Nand";}
};

template <typename T, typename R>
class Nor {
  public:
    static R apply(T first, T second) {
      return ~(second | first);
    }
    static std::string name() {return "Nor";}
};

template <typename T, typename R>
class Xnor {
  public:
    static R apply(T first, T second) {
      return ~(second ^ first);
    }
    static std::string name() {return "Xnor";}
};

template <typename T, typename R>
class Fadd {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      // ignoring rounding mode for now
      uint32_t frm = 0;
      if (sizeof(T) == 4) {
        return rv_fadd_s(first, second, frm, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fadd_d(first, second, frm, &fflags);
      } else {
        std::cout << "Fadd only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fadd";}
};

template <typename T, typename R>
class Fsub {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      // ignoring rounding mode for now
      uint32_t frm = 0;
      if (sizeof(T) == 4) {
        return rv_fsub_s(second, first, frm, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fsub_d(second, first, frm, &fflags);
      } else {
        std::cout << "Fsub only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fsub";}
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
class Fsgnj {
  public:
    static R apply(T first, T second) {
      if (sizeof(T) == 4) {
        return rv_fsgnj_s(second, first);
      } else if (sizeof(T) == 8) {
        return rv_fsgnj_d(second, first);
      } else {
        std::cout << "Fsgnj only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fsgnj";}
};

template <typename T, typename R>
class Fsgnjn {
  public:
    static R apply(T first, T second) {
      if (sizeof(T) == 4) {
        return rv_fsgnjn_s(second, first);
      } else if (sizeof(T) == 8) {
        return rv_fsgnjn_d(second, first);
      } else {
        std::cout << "Fsgnjn only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fsgnjn";}
};

template <typename T, typename R>
class Fsgnjx {
  public:
    static R apply(T first, T second) {
      if (sizeof(T) == 4) {
        return rv_fsgnjx_s(second, first);
      } else if (sizeof(T) == 8) {
        return rv_fsgnjx_d(second, first);
      } else {
        std::cout << "Fsgnjx only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fsgnjx";}
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
            return rv_ftol_d(second, frm, &fflags);
          case 0b011: // vfcvt.f.x.v
            return rv_ltof_d(second, frm, &fflags);
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
          case 0b00000: // vfsqrt.v
            return rv_fsqrt_s(second, frm, &fflags);
          case 0b00100: // vfrsqrt7.v
            return rv_frsqrt7_s(second, frm, &fflags);
          case 0b00101: // vfrec7.v
            return rv_frecip7_s(second, frm, &fflags);
          case 0b10000: // vfclass.v
            return rv_fclss_s(second);
          default:
            std::cout << "Funary1 has unsupported value for first: " << first << std::endl;
            std::abort();
        }
      } else if (sizeof(T) == 8) {
        switch (first) {
          case 0b00000: // vfsqrt.v
            return rv_fsqrt_d(second, frm, &fflags);
          case 0b00100: // vfrsqrt7.v
            return rv_frsqrt7_d(second, frm, &fflags);
          case 0b00101: // vfrec7.v
            return rv_frecip7_d(second, frm, &fflags);
          case 0b10000: // vfclass.v
            return rv_fclss_d(second);
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

template <typename T, typename R>
class Feq {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_feq_s(second, first, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_feq_d(second, first, &fflags);
      } else {
        std::cout << "Feq only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Feq";}
};

template <typename T, typename R>
class Fle {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_fle_s(second, first, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fle_d(second, first, &fflags);
      } else {
        std::cout << "Fle only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fle";}
};

template <typename T, typename R>
class Flt {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_flt_s(second, first, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_flt_d(second, first, &fflags);
      } else {
        std::cout << "Flt only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Flt";}
};

template <typename T, typename R>
class Fne {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return !rv_feq_s(second, first, &fflags);
      } else if (sizeof(T) == 8) {
        return !rv_feq_d(second, first, &fflags);
      } else {
        std::cout << "Fne only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fne";}
};

template <typename T, typename R>
class Fgt {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_flt_s(first, second, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_flt_d(first, second, &fflags);
      } else {
        std::cout << "Fgt only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fgt";}
};

template <typename T, typename R>
class Fge {
  public:
    static R apply(T first, T second) {
      // ignoring flags for now
      uint32_t fflags = 0;
      if (sizeof(T) == 4) {
        return rv_fle_s(first, second, &fflags);
      } else if (sizeof(T) == 8) {
        return rv_fle_d(first, second, &fflags);
      } else {
        std::cout << "Fge only supports f32 and f64" << std::endl;
        std::abort();
      }
    }
    static std::string name() {return "Fge";}
};

template <typename T, typename R>
class Clip {
  private:
    static bool bitAt(T value, R pos, R negOffset) {
      R offsetPos = pos - negOffset;
      return pos >= negOffset && ((value >> offsetPos) & 0x1);
    }
    static bool anyBitUpTo(T value, R to, R negOffset) {
      R offsetTo = to - negOffset;
      return to >= negOffset && (value & ((1 << (offsetTo + 1)) - 1));
    }
    static bool roundBit(T value, R shiftDown, uint32_t vxrm) {
      switch (vxrm){
        case 0: // round-to-nearest-up
          return bitAt(value, shiftDown, 1);
        case 1: // round-to-nearest-even
          return bitAt(value, shiftDown, 1) && (anyBitUpTo(value, shiftDown, 2) || bitAt(value, shiftDown, 0));
        case 2: // round-down (truncate)
          return 0;
        case 3: // round-to-odd
          return !bitAt(value, shiftDown, 0) && anyBitUpTo(value, shiftDown, 1);
        default:
          std::cout << "Clip - invalid value for vxrm: " << vxrm << std::endl;
          std::abort();
      }
    }
  public:
    static R apply(T first, T second, uint32_t vxrm, uint32_t &vxsat_) {
      // ignoring rounding mode for now, simply rounding up to pass the tests
      // The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to
      // SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.
      R firstValid = first & (sizeof(T) * 8 - 1);
      T unclippedResult = (second >> firstValid) + roundBit(second, firstValid, vxrm);
      R clippedResult = std::clamp(unclippedResult, (T)std::numeric_limits<R>::min(), (T)std::numeric_limits<R>::max());
      vxsat_ |= clippedResult != unclippedResult;
      return clippedResult;
    }
    static std::string name() {return "Clip";}
};

bool isMasked(std::vector<std::vector<Byte>> &vreg_file, uint32_t maskVreg, uint32_t byteI, bool vmask) {
  auto& mask = vreg_file.at(maskVreg);
  uint8_t emask = *(uint8_t *)(mask.data() + byteI / 8);
  uint8_t value = (emask >> (byteI % 8)) & 0x1;
  DP(1, "Masking enabled: " << +!vmask << " mask element: " << +value);
  return !vmask && value == 0;
}

template <typename DT>
uint32_t getVreg(uint32_t baseVreg, uint32_t byteI) {
  uint32_t vsew = sizeof(DT) * 8;
  return (baseVreg + (byteI / (VLEN / vsew))) % 32;
}

template <typename DT>
DT &getVregData(std::vector<vortex::Byte> &baseVregVec, uint32_t byteI) {
  uint32_t vsew = sizeof(DT) * 8;
  return *(DT *)(baseVregVec.data() + (byteI % (VLEN / vsew)) * vsew / 8);
}

template <typename DT>
DT &getVregData(std::vector<std::vector<vortex::Byte>> &vreg_file, uint32_t baseVreg, uint32_t byteI) {
  auto& vr1 = vreg_file.at(getVreg<DT>(baseVreg, byteI));
  return getVregData<DT>(vr1, byteI);
}

template <typename DT>
void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;
    
    Word mem_addr = ((rsdata[0][0].i) & 0xFFFFFFFC) + (i * vsew / 8);
    Word mem_data = 0;
    core_->dcache_read(&mem_data, mem_addr, vsew / 8);
    DP(1, "Loading data " << mem_data << " from: " << mem_addr << " to vec reg: " << getVreg<DT>(rdest, i));
    DT &result = getVregData<DT>(vreg_file, rdest, i);
    DP(1, "Previous data: " << +result);
    result = (DT) mem_data;
  }
}

void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
    case 8:
      loadVector<uint8_t>(vreg_file, core_, rsdata, rdest, vl, vmask);
      break;
    case 16:
      loadVector<uint16_t>(vreg_file, core_, rsdata, rdest, vl, vmask);
      break;
    case 32:
      loadVector<uint32_t>(vreg_file, core_, rsdata, rdest, vl, vmask);
      break;
    case 64:
      loadVector<uint64_t>(vreg_file, core_, rsdata, rdest, vl, vmask);
      break;
    default:
      std::cout << "Failed to execute VLE for vsew: " << vsew << std::endl;
      std::abort();
  }
}

template <typename DT>
void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, uint32_t vl, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    Word mem_addr = rsdata[0][0].i + (i * vsew / 8); 
    Word mem_data = getVregData<DT>(vreg_file, rsrc3, i);
    DP(1, "Storing: " << std::hex << mem_data << " at: " << mem_addr << " from vec reg: " << getVreg<DT>(rsrc3, i));
    core_->dcache_write(&mem_data, mem_addr, vsew / 8);
  }
}

void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
    case 8:
      storeVector<uint8_t>(vreg_file, core_, rsdata, rsrc3, vl, vmask);
      break;
    case 16:
      storeVector<uint16_t>(vreg_file, core_, rsdata, rsrc3, vl, vmask);
      break;
    case 32:
      storeVector<uint32_t>(vreg_file, core_, rsdata, rsrc3, vl, vmask);
      break;
    case 64:
      storeVector<uint64_t>(vreg_file, core_, rsdata, rsrc3, vl, vmask);
      break;
    default:
      std::cout << "Failed to execute VSE for vsew: " << vsew << std::endl;
      std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;
    
    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DT result = OP<DT, DT>::apply(first, second);
    DP(1, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vix<OP, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vix<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vix<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 64) {
    vector_op_vix<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VI/VX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_w(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DTR result = OP<DT, DTR>::apply(first, second);
    DP(1, "Widening " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_w(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm)
{
  if (vsew == 8) {
    vector_op_vix_w<OP, DT8, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm);
  } else if (vsew == 16) {
    vector_op_vix_w<OP, DT16, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm);
  } else if (vsew == 32) {
    vector_op_vix_w<OP, DT32, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm);
  } else {
    std::cout << "Failed to execute VI/VX widening for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_n(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(1, "Narrowing " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_n(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat)
{
  if (vsew == 8) {
    vector_op_vix_n<OP, DT16, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
  } else if (vsew == 16) {
    vector_op_vix_n<OP, DT32, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
  } else if (vsew == 32) {
    vector_op_vix_n<OP, DT64, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
  } else {
    std::cout << "Failed to execute VI/VX narrowing for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix_mask(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    bool result = OP<DT, bool>::apply(first, second);
    DP(1, "Integer/float compare mask " << (OP<DT, bool>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_mask(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vix_mask<OP, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vix_mask<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vix_mask<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else if (vsew == 64) {
    vector_op_vix_mask<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VI/VX integer/float compare mask for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT first  = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DT result = OP<DT, DT>::apply(first, second);
    DP(1, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vv<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vv<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vv<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 64) {
    vector_op_vv<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR result = OP<DT, DTR>::apply(first, second);
    DP(1, "Widening " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vv_w<OP, DT8, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vv_w<OP, DT16, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vv_w<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV widening for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_n(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DTR first = getVregData<DTR>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(1, "Narrowing " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_n(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat)
{
  if (vsew == 8) {
    vector_op_vv_n<OP, DT16, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
  } else if (vsew == 16) {
    vector_op_vv_n<OP, DT32, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
  } else if (vsew == 32) {
    vector_op_vv_n<OP, DT64, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
  } else {
    std::cout << "Failed to execute VV narrowing for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv_red(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    // use rdest as accumulator
    if (i == 0) {
      getVregData<DT>(vreg_file, rdest, 0) = getVregData<DT>(vreg_file, rsrc0, 0);
    }
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT first = getVregData<DT>(vreg_file, rdest, 0);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DT result = OP<DT, DT>::apply(first, second);
    DP(1, "Reduction " << (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, 0) = result;
  } 
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_red(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vv_red<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vv_red<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vv_red<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 64) {
    vector_op_vv_red<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV reduction for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask)
{
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask)) continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    bool result = OP<DT, bool>::apply(first, second);
    DP(1, "Integer/float compare mask " << (OP<DT, bool>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask)
{
  if (vsew == 8) {
    vector_op_vv_mask<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 16) {
    vector_op_vv_mask<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 32) {
    vector_op_vv_mask<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else if (vsew == 64) {
    vector_op_vv_mask<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV integer/float compare mask for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl)
{
  for (uint32_t i = 0; i < vl; i++) {
    uint8_t firstMask = getVregData<uint8_t>(vreg_file, rsrc0, i / 8);
    bool first = (firstMask >> (i % 8)) & 0x1;
    uint8_t secondMask = getVregData<uint8_t>(vreg_file, rsrc1, i / 8);
    bool second = (secondMask >> (i % 8)) & 0x1;
    bool result = OP<uint8_t, uint8_t>::apply(first, second) & 0x1;
    DP(1, "Compare mask bits " << (OP<uint8_t, uint8_t>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <typename DT>
void vector_op_vv_compress(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl)
{
  int currPos = 0;
  for (uint32_t i = 0; i < vl; i++) {
    // Special case: use rsrc0 as mask vector register instead of default v0
    // This instruction is always masked (vmask == 0), but encoded as unmasked (vmask == 1)
    if (isMasked(vreg_file, rsrc0, i, 0)) continue;

    DT value = getVregData<DT>(vreg_file, rsrc1, i);
    DP(1, "Compression - Moving value " << value << " from position " << i << " to position " << currPos);
    getVregData<DT>(vreg_file, rdest, currPos) = value;
    currPos++;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_compress(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl_)
{
  if (vsew == 8) {
    vector_op_vv_compress<DT8>(vreg_file, rsrc0, rsrc1, rdest, vl_);
  } else if (vsew == 16) {
    vector_op_vv_compress<DT16>(vreg_file, rsrc0, rsrc1, rdest, vl_);
  } else if (vsew == 32) {
    vector_op_vv_compress<DT32>(vreg_file, rsrc0, rsrc1, rdest, vl_);
  } else if (vsew == 64) {
    vector_op_vv_compress<DT64>(vreg_file, rsrc0, rsrc1, rdest, vl_);
  } else {
    std::cout << "Failed to execute VV compression for vsew: " << vsew << std::endl;
    std::abort();
  }
}

void Warp::executeVector(const Instr &instr, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata) {
  auto func3  = instr.getFunc3();
  auto func6  = instr.getFunc6();

  auto rdest  = instr.getRDest();
  auto rsrc0  = instr.getRSrc(0);
  auto rsrc1  = instr.getRSrc(1);
  auto immsrc = sext((Word)instr.getImm(), 32);
  auto vmask  = instr.getVmask();
  auto num_threads = arch_.num_threads();
  
    uint32_t VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
    switch (func3) {
    case 0: { // vector - vector
        switch (func6) { 
          case 0: { // vadd.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 2: { // vsub.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Sub, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 4: { // vminu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Min, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 5: { // vmin.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Min, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 6: { // vmaxu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Max, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 7: { // vmax.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Max, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 9: { // vand.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<And, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 10: { // vor.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Or, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 11: { // vxor.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Xor, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 24: { // vmseq.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Eq, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 25: {  // vmsne.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Ne, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 26: { // vmsltu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 27: { // vmslt.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Lt, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 28: { // vmsleu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 29: { // vmsle.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Le, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 30: { // vmsgtu.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 31: { // vmsgt.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Gt, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 37: { // vsll.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Sll, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 40: { // vsrl.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 41: { // vsra.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<SrlSra, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 46: { // vnclipu.wv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
              uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
              vector_op_vv_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
              core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
            }
          } break;
          case 47: { // vnclip.wv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
              uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
              vector_op_vv_n<Clip, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
              core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
            }
          } break;
          default:
            std::cout << "Unrecognised vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
            std::abort();
        } 
      } break;
    case 1: { // float vector - vector
        switch (func6) {
          case 0: { // vfadd.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 2: { // vfsub.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 1: // vfredusum.vs - treated the same as vfredosum.vs
          case 3: { // vfredosum.vs
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_red<Fadd, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 4: { // vfmin.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 5: { // vfredmin.vs
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_red<Fmin, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 6: { // vfmax.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 7: { // vfredmax.vs
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_red<Fmax, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 8: { // vfsgnj.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 9: { // vfsgnjn.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 10: { // vfsgnjx.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 18: { // vfcvt.f.x.v, vfcvt.x.f.v
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vix<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 19: { // vfsqrt.v, vfrsqrt7.v, vfrec7.v, vfclass.v
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vix<Funary1, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 24: { // vmfeq.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Feq, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 25: { // vmfle.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Fle, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 27: { // vmflt.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Flt, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 28: { // vmfne.vv
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              vector_op_vv_mask<Fne, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          default:
            std::cout << "Unrecognised float vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
            std::abort();
        }
      } break;
    case 2: { // mask vector - vector
      switch (func6) {
        case 0: { // vredsum.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 1: { // vredand.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<And, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 2: { // vredor.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Or, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 3: { // vredxor.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Xor, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 4: { // vredminu.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Min, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 5: { // vredmin.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Min, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 6: { // vredmaxu.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Max, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 7: { // vredmax.vs
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_red<Max, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 23: { // vcompress.vm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_compress<uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_);
          }
        } break;
        case 24: { // vmandn.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<AndNot>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 25: { // vmand.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<And>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 26: { // vmor.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<Or>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 27: { // vmxor.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<Xor>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 28: { // vmorn.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<OrNot>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 29: { // vmnand.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<Nand>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 30: { // vmnor.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<Nor>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 31: { // vmxnor.mm
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_mask<Xnor>(vreg_file_, rsrc0, rsrc1, rdest, vl_);
          }
        } break;
        case 36: { // vmulhu.vv
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 37: { // vmul.vv
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv<Mul, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 39: { // vmulh.vv
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv<Mulh, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 59: { // vwmul.vv
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            vector_op_vv_w<Mul, int8_t, int16_t, int32_t, int64_t>(vreg_file_, rsrc0, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        default:
          std::cout << "Unrecognised mask vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
          std::abort();
      }
    } break;
    case 3: { // vector - immidiate
      switch (func6) {
      case 0: { // vadd.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 3: { // vrsub.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 9: { // vand.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 10: { // vor.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 11: { // vxor.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 24: { // vmseq.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 25: {  // vmsne.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 26: { // vmsltu.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 27: { // vmslt.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 28: { // vmsleu.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 29: { // vmsle.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 30: { // vmsgtu.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 31: { // vmsgt.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 37: { // vsll.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<Sll, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 40: { // vsrl.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 41: { // vsra.vi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask);
        }
      } break;
      case 46: { // vnclipu.wi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
          uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
          vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
          core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
        }
      } break;
      case 47: { // vnclip.wi
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t)) continue;
          uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
          uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
          vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file_, rsrc0, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
          core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
        }
      } break;
      default:
        std::cout << "Unrecognised vector - immidiate instruction func3: " << func3 << " func6: " << func6 << std::endl;
        std::abort();
      }
    } break;
    case 4:{
      switch (func6){
        case 0: { // vadd.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 2: { // vsub.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Sub, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 3: { // vrsub.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 4: { // vminu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Min, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 5: { // vmin.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Min, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 6: { // vmaxu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Max, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 7: { // vmax.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Max, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 9: { // vand.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 10: { // vor.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 11: { // vxor.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 24: { // vmseq.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 25: {  // vmsne.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 26: { // vmsltu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 27: { // vmslt.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 28: { // vmsleu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 29: { // vmsle.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 30: { // vmsgtu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 31: { // vmsgt.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 37: { // vsll.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Sll, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 40: { // vsrl.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 41: { // vsra.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 46: { // vnclipu.wx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
            uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
            vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
            core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
          }
        } break;
        case 47: { // vnclip.wx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto& src1 = ireg_file_.at(t).at(rsrc0);
            uint32_t vxrm = core_->get_csr(VX_CSR_VXRM, t, warp_id_);
            uint32_t vxsat = core_->get_csr(VX_CSR_VXSAT, t, warp_id_);
            vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask, vxrm, vxsat);
            core_->set_csr(VX_CSR_VXSAT, vxsat, t, warp_id_);
          }
        } break;
        default:
          std::cout << "Unrecognised vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
          std::abort();
      }
    } break;
    case 5: { // float vector - scalar
        switch (func6) {
          case 0: { // vfadd.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 2: { // vfsub.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 4: { // vfmin.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 6: { // vfmax.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 8: { // vfsgnj.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 9: { // vfsgnjn.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 10: { // vfsgnjx.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 24: { // vmfeq.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Feq, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 25: { // vmfle.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Fle, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 27: { // vmflt.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Flt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 28: { // vmfne.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Fne, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 29: { // vmfgt.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Fgt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          case 31: { // vmfge.vf
            for (uint32_t t = 0; t < num_threads; ++t) {
              if (!tmask_.test(t)) continue;
              auto &src1 = freg_file_.at(t).at(rsrc0);
              vector_op_vix_mask<Fge, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
            }
          } break;
          default:
            std::cout << "Unrecognised float vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
            std::abort();
        }
      } break;
    case 6: {
      switch (func6) {
        case 36: { // vmulhu.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 37: { // vmul.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Mul, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
          }
        } break;
        case 39: { // vmulh.vx
          for (uint32_t t = 0; t < num_threads; ++t) {
            if (!tmask_.test(t)) continue;
            auto &src1 = ireg_file_.at(t).at(rsrc0);
            vector_op_vix<Mulh, int8_t, int16_t, int32_t, int64_t>(src1, vreg_file_, rsrc1, rdest, vtype_.vsew, vl_, vmask);
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
      vtype_.vill  = vsew > XLEN || VLMAX < VLEN / XLEN;

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
        core_->set_csr(VX_CSR_VTYPE, (Word)1 << (XLEN - 1), 0, warp_id_);
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
        Word vtype = vlmul;
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