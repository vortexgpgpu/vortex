// This is a fork of https://github.com/troibe/vortex/tree/simx-v2-vector
// The purpose of this fork is to make the simx-v2-vector up to date with master
// Thanks to Troibe for his amazing work

#include "emulator.h"
#include "instr.h"
#include "processor_impl.h"
#include <iostream>
#include <limits>
#include <math.h>
#include <rvfloats.h>
#include <stdlib.h>

using namespace vortex;

template <typename T, typename R>
class Add {
public:
  static R apply(T first, T second, R) {
    return (R)first + (R)second;
  }
  static std::string name() { return "Add"; }
};

template <typename T, typename R>
class Sub {
public:
  static R apply(T first, T second, R) {
    return (R)second - (R)first;
  }
  static std::string name() { return "Sub"; }
};

template <typename T, typename R>
class Adc {
public:
  static R apply(T first, T second, R third) {
    return (R)first + (R)second + third;
  }
  static std::string name() { return "Adc"; }
};

template <typename T, typename R>
class Madc {
public:
  static R apply(T first, T second, R third) {
    return ((R)first + (R)second + third) > (R)std::numeric_limits<T>::max();
  }
  static std::string name() { return "Madc"; }
};

template <typename T, typename R>
class Sbc {
public:
  static R apply(T first, T second, R third) {
    return (R)second - (R)first - third;
  }
  static std::string name() { return "Sbc"; }
};

template <typename T, typename R>
class Msbc {
public:
  static R apply(T first, T second, R third) {
    return (R)second < ((R)first + third);
  }
  static std::string name() { return "Msbc"; }
};

template <typename T, typename R>
class Ssub {
public:
  static R apply(T first, T second, uint32_t, uint32_t &vxsat_) {
    // rounding mode is not relevant for this operation
    T unclippedResult = second - first;
    R clippedResult = std::clamp(unclippedResult, (T)std::numeric_limits<R>::min(), (T)std::numeric_limits<R>::max());
    vxsat_ |= clippedResult != unclippedResult;
    return clippedResult;
  }
  static std::string name() { return "Ssub"; }
};

template <typename T, typename R>
class Ssubu {
public:
  static R apply(T first, T second, uint32_t, uint32_t &vxsat_) {
    // rounding mode is not relevant for this operation
    if (first > second) {
      vxsat_ = true;
      return 0;
    } else {
      vxsat_ = false;
      return second - first;
    }
  }
  static std::string name() { return "Ssubu"; }
};

template <typename T, typename R>
class Sadd {
public:
  static R apply(T first, T second, uint32_t, uint32_t &vxsat_) {
    // rounding mode is not relevant for this operation
    T unclippedResult = second + first;
    R clippedResult = std::clamp(unclippedResult, (T)std::numeric_limits<R>::min(), (T)std::numeric_limits<R>::max());
    vxsat_ |= clippedResult != unclippedResult;
    return clippedResult;
  }
  static std::string name() { return "Sadd"; }
};

template <typename T, typename R>
class Rsub {
public:
  static R apply(T first, T second, R) {
    return first - second;
  }
  static std::string name() { return "Rsub"; }
};

template <typename T, typename R>
class Div {
public:
  static R apply(T first, T second, R) {
    // logic taken from scalar div
    if (first == 0) {
      return -1;
    } else if (second == std::numeric_limits<T>::min() && first == T(-1)) {
      return second;
    } else {
      return (R)second / (R)first;
    }
  }
  static std::string name() { return "Div"; }
};

template <typename T, typename R>
class Rem {
public:
  static R apply(T first, T second, R) {
    // logic taken from scalar rem
    if (first == 0) {
      return second;
    } else if (second == std::numeric_limits<T>::min() && first == T(-1)) {
      return 0;
    } else {
      return (R)second % (R)first;
    }
  }
  static std::string name() { return "Rem"; }
};

template <typename T, typename R>
class Mul {
public:
  static R apply(T first, T second, R) {
    return (R)first * (R)second;
  }
  static std::string name() { return "Mul"; }
};

template <typename T, typename R>
class Mulsu {
public:
  static R apply(T first, T second, R) {
    R first_ext = zext((R)first, (sizeof(T) * 8));
    return first_ext * (R)second;
  }
  static std::string name() { return "Mulsu"; }
};

template <typename T, typename R>
class Mulh {
public:
  static R apply(T first, T second, R) {
    __int128_t first_ext = sext((__int128_t)first, (sizeof(T) * 8));
    __int128_t second_ext = sext((__int128_t)second, (sizeof(T) * 8));
    return (first_ext * second_ext) >> (sizeof(T) * 8);
  }
  static std::string name() { return "Mulh"; }
};

template <typename T, typename R>
class Mulhsu {
public:
  static R apply(T first, T second, R) {
    __int128_t first_ext = zext((__int128_t)first, (sizeof(T) * 8));
    __int128_t second_ext = sext((__int128_t)second, (sizeof(T) * 8));
    return (first_ext * second_ext) >> (sizeof(T) * 8);
  }
  static std::string name() { return "Mulhsu"; }
};

template <typename T, typename R>
class Mulhu {
public:
  static R apply(T first, T second, R) {
    return ((__uint128_t)first * (__uint128_t)second) >> (sizeof(T) * 8);
  }
  static std::string name() { return "Mulhu"; }
};

template <typename T, typename R>
class Madd {
public:
  static R apply(T first, T second, R third) {
    return ((R)first * third) + (R)second;
  }
  static std::string name() { return "Madd"; }
};

template <typename T, typename R>
class Nmsac {
public:
  static R apply(T first, T second, R third) {
    return -((R)first * (R)second) + third;
  }
  static std::string name() { return "Nmsac"; }
};

template <typename T, typename R>
class Macc {
public:
  static R apply(T first, T second, R third) {
    return ((R)first * (R)second) + third;
  }
  static std::string name() { return "Macc"; }
};

template <typename T, typename R>
class Maccsu {
public:
  static R apply(T first, T second, R third) {
    R first_ext = sext((R)first, (sizeof(T) * 8));
    R second_ext = zext((R)second, (sizeof(T) * 8));
    return (first_ext * second_ext) + third;
  }
  static std::string name() { return "Maccsu"; }
};

template <typename T, typename R>
class Maccus {
public:
  static R apply(T first, T second, R third) {
    R first_ext = zext((R)first, (sizeof(T) * 8));
    R second_ext = sext((R)second, (sizeof(T) * 8));
    return (first_ext * second_ext) + third;
  }
  static std::string name() { return "Maccus"; }
};

template <typename T, typename R>
class Nmsub {
public:
  static R apply(T first, T second, R third) {
    return -((R)first * third) + (R)second;
  }
  static std::string name() { return "Nmsub"; }
};

template <typename T, typename R>
class Min {
public:
  static R apply(T first, T second, R) {
    return std::min(first, second);
  }
  static std::string name() { return "Min"; }
};

template <typename T, typename R>
class Max {
public:
  static R apply(T first, T second, R) {
    return std::max(first, second);
  }
  static std::string name() { return "Max"; }
};

template <typename T, typename R>
class And {
public:
  static R apply(T first, T second, R) {
    return first & second;
  }
  static std::string name() { return "And"; }
};

template <typename T, typename R>
class Or {
public:
  static R apply(T first, T second, R) {
    return first | second;
  }
  static std::string name() { return "Or"; }
};

template <typename T, typename R>
class Xor {
public:
  static R apply(T first, T second, R) {
    return first ^ second;
  }
  static std::string name() { return "Xor"; }
};

template <typename T, typename R>
class Sll {
public:
  static R apply(T first, T second, R) {
    // Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.
    return second << (first & (sizeof(T) * 8 - 1));
  }
  static std::string name() { return "Sll"; }
};

template <typename T, typename R>
bool bitAt(T value, R pos, R negOffset) {
  R offsetPos = pos - negOffset;
  return pos >= negOffset && ((value >> offsetPos) & 0x1);
}

template <typename T, typename R>
bool anyBitUpTo(T value, R to, R negOffset) {
  R offsetTo = to - negOffset;
  return to >= negOffset && (value & (((R)1 << (offsetTo + 1)) - 1));
}

template <typename T, typename R>
bool roundBit(T value, R shiftDown, uint32_t vxrm) {
  switch (vxrm) {
  case 0: // round-to-nearest-up
    return bitAt(value, shiftDown, (R)1);
  case 1: // round-to-nearest-even
    return bitAt(value, shiftDown, (R)1) && (anyBitUpTo(value, shiftDown, (R)2) || bitAt(value, shiftDown, (R)0));
  case 2: // round-down (truncate)
    return 0;
  case 3: // round-to-odd
    return !bitAt(value, shiftDown, (R)0) && anyBitUpTo(value, shiftDown, (R)1);
  default:
    std::cout << "Roundoff - invalid value for vxrm: " << vxrm << std::endl;
    std::abort();
  }
}

template <typename T, typename R>
class SrlSra {
public:
  static R apply(T first, T second, R) {
    // Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.
    return second >> (first & (sizeof(T) * 8 - 1));
  }
  static R apply(T first, T second, uint32_t vxrm, uint32_t) {
    // Saturation is not relevant for this operation
    // Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.
    T firstValid = first & (sizeof(T) * 8 - 1);
    return apply(firstValid, second, 0) + roundBit(second, firstValid, vxrm);
  }
  static std::string name() { return "SrlSra"; }
};

template <typename T, typename R>
class Aadd {
public:
  static R apply(T first, T second, uint32_t vxrm, uint32_t) {
    // Saturation is not relevant for this operation
    T sum = second + first;
    return (sum >> 1) + roundBit(sum, 1, vxrm);
  }
  static std::string name() { return "Aadd"; }
};

template <typename T, typename R>
class Asub {
public:
  static R apply(T first, T second, uint32_t vxrm, uint32_t) {
    // Saturation is not relevant for this operation
    T difference = second - first;
    return (difference >> 1) + roundBit(difference, 1, vxrm);
  }
  static std::string name() { return "Asub"; }
};

template <typename T, typename R>
class Eq {
public:
  static R apply(T first, T second, R) {
    return first == second;
  }
  static std::string name() { return "Eq"; }
};

template <typename T, typename R>
class Ne {
public:
  static R apply(T first, T second, R) {
    return first != second;
  }
  static std::string name() { return "Ne"; }
};

template <typename T, typename R>
class Lt {
public:
  static R apply(T first, T second, R) {
    return first > second;
  }
  static std::string name() { return "Lt"; }
};

template <typename T, typename R>
class Le {
public:
  static R apply(T first, T second, R) {
    return first >= second;
  }
  static std::string name() { return "Le"; }
};

template <typename T, typename R>
class Gt {
public:
  static R apply(T first, T second, R) {
    return first < second;
  }
  static std::string name() { return "Gt"; }
};

template <typename T, typename R>
class AndNot {
public:
  static R apply(T first, T second, R) {
    return second & ~first;
  }
  static std::string name() { return "AndNot"; }
};

template <typename T, typename R>
class OrNot {
public:
  static R apply(T first, T second, R) {
    return second | ~first;
  }
  static std::string name() { return "OrNot"; }
};

template <typename T, typename R>
class Nand {
public:
  static R apply(T first, T second, R) {
    return ~(second & first);
  }
  static std::string name() { return "Nand"; }
};

template <typename T, typename R>
class Mv {
public:
  static R apply(T first, T, R) {
    return first;
  }
  static std::string name() { return "Mv"; }
};

template <typename T, typename R>
class Nor {
public:
  static R apply(T first, T second, R) {
    return ~(second | first);
  }
  static std::string name() { return "Nor"; }
};

template <typename T, typename R>
class Xnor {
public:
  static R apply(T first, T second, R) {
    return ~(second ^ first);
  }
  static std::string name() { return "Xnor"; }
};

template <typename T, typename R>
class Fadd {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fadd_s(first, second, frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fadd_d(first_d, second_d, frm, &fflags);
    } else {
      std::cout << "Fadd only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fadd"; }
};

template <typename T, typename R>
class Fsub {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fsub_s(second, first, frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fsub_d(second_d, first_d, frm, &fflags);
    } else {
      std::cout << "Fsub only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fsub"; }
};

template <typename T, typename R>
class Fmacc {
public:
  static R apply(T first, T second, R third) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fmadd_s(first, second, third, frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fmadd_d(first_d, second_d, third, frm, &fflags);
    } else {
      std::cout << "Fmacc only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fmacc"; }
};

template <typename T, typename R>
class Fnmacc {
public:
  static R apply(T first, T second, R third) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fnmadd_s(first, second, third, frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fnmadd_d(first_d, second_d, third, frm, &fflags);
    } else {
      std::cout << "Fnmacc only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fnmacc"; }
};

template <typename T, typename R>
class Fmsac {
public:
  static R apply(T first, T second, R third) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fmadd_s(first, second, rv_fsgnjn_s(third, third), frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fmadd_d(first_d, second_d, rv_fsgnjn_d(third, third), frm, &fflags);
    } else {
      std::cout << "Fmsac only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fmsac"; }
};

template <typename T, typename R>
class Fnmsac {
public:
  static R apply(T first, T second, R third) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fnmadd_s(first, second, rv_fsgnjn_s(third, third), frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fnmadd_d(first_d, second_d, rv_fsgnjn_d(third, third), frm, &fflags);
    } else {
      std::cout << "Fnmsac only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fnmsac"; }
};

template <typename T, typename R>
class Fmadd {
public:
  static R apply(T first, T second, R third) {
    if (sizeof(T) == 4 || sizeof(T) == 8) {
      return Fmacc<T, R>::apply(first, third, second);
    } else {
      std::cout << "Fmadd only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fmadd"; }
};

template <typename T, typename R>
class Fnmadd {
public:
  static R apply(T first, T second, R third) {
    if (sizeof(T) == 4 || sizeof(T) == 8) {
      return Fnmacc<T, R>::apply(first, third, second);
    } else {
      std::cout << "Fnmadd only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fnmadd"; }
};

template <typename T, typename R>
class Fmsub {
public:
  static R apply(T first, T second, R third) {
    if (sizeof(T) == 4 || sizeof(T) == 8) {
      return Fmsac<T, R>::apply(first, third, second);
    } else {
      std::cout << "Fmsub only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fmsub"; }
};

template <typename T, typename R>
class Fnmsub {
public:
  static R apply(T first, T second, R third) {
    if (sizeof(T) == 4 || sizeof(T) == 8) {
      return Fnmsac<T, R>::apply(first, third, second);
    } else {
      std::cout << "Fnmsub only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fnmsub"; }
};

template <typename T, typename R>
class Fmin {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fmin"; }
};

template <typename T, typename R>
class Fmax {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fmax"; }
};

template <typename T, typename R>
class Fsgnj {
public:
  static R apply(T first, T second, R) {
    if (sizeof(T) == 4) {
      return rv_fsgnj_s(second, first);
    } else if (sizeof(T) == 8) {
      return rv_fsgnj_d(second, first);
    } else {
      std::cout << "Fsgnj only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fsgnj"; }
};

template <typename T, typename R>
class Fsgnjn {
public:
  static R apply(T first, T second, R) {
    if (sizeof(T) == 4) {
      return rv_fsgnjn_s(second, first);
    } else if (sizeof(T) == 8) {
      return rv_fsgnjn_d(second, first);
    } else {
      std::cout << "Fsgnjn only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fsgnjn"; }
};

template <typename T, typename R>
class Fsgnjx {
public:
  static R apply(T first, T second, R) {
    if (sizeof(T) == 4) {
      return rv_fsgnjx_s(second, first);
    } else if (sizeof(T) == 8) {
      return rv_fsgnjx_d(second, first);
    } else {
      std::cout << "Fsgnjx only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fsgnjx"; }
};

template <typename T, typename R>
class Fcvt {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(T) == 4) {
      switch (first) {
      case 0b00000: // vfcvt.xu.f.v
        return rv_ftou_s(second, frm, &fflags);
      case 0b00001: // vfcvt.x.f.v
        return rv_ftoi_s(second, frm, &fflags);
      case 0b00010: // vfcvt.f.xu.v
        return rv_utof_s(second, frm, &fflags);
      case 0b00011: // vfcvt.f.x.v
        return rv_itof_s(second, frm, &fflags);
      case 0b00110: // vfcvt.rtz.xu.f.v
        return rv_ftou_s(second, 1, &fflags);
      case 0b00111: // vfcvt.rtz.x.f.v
        return rv_ftoi_s(second, 1, &fflags);
      case 0b01000: // vfwcvt.xu.f.v
        return rv_ftolu_s(second, frm, &fflags);
      case 0b01001: // vfwcvt.x.f.v
        return rv_ftol_s(second, frm, &fflags);
      case 0b01010: // vfwcvt.f.xu.v
        return rv_utof_d(second, frm, &fflags);
      case 0b01011: // vfwcvt.f.x.v
        return rv_itof_d(second, frm, &fflags);
      case 0b01100: // vfwcvt.f.f.v
        return rv_ftod(second);
      case 0b01110: // vfwcvt.rtz.xu.f.v
        return rv_ftolu_s(second, 1, &fflags);
      case 0b01111: // vfwcvt.rtz.x.f.v
        return rv_ftol_s(second, 1, &fflags);
      default:
        std::cout << "Fcvt has unsupported value for first: " << first << std::endl;
        std::abort();
      }
    } else if (sizeof(T) == 8) {
      switch (first) {
      case 0b00000: // vfcvt.xu.f.v
        return rv_ftolu_d(second, frm, &fflags);
      case 0b00001: // vfcvt.x.f.v
        return rv_ftol_d(second, frm, &fflags);
      case 0b00010: // vfcvt.f.xu.v
        return rv_lutof_d(second, frm, &fflags);
      case 0b00011: // vfcvt.f.x.v
        return rv_ltof_d(second, frm, &fflags);
      case 0b00110: // vfcvt.rtz.xu.f.v
        return rv_ftolu_d(second, 1, &fflags);
      case 0b00111: // vfcvt.rtz.x.f.v
        return rv_ftol_d(second, 1, &fflags);
      case 0b01000: // vfwcvt.xu.f.v
      case 0b01001: // vfwcvt.x.f.v
      case 0b01010: // vfwcvt.f.xu.v
      case 0b01011: // vfwcvt.f.x.v
      case 0b01100: // vfwcvt.f.f.v
      case 0b01110: // vfwcvt.rtz.xu.f.v
      case 0b01111: // vfwcvt.rtz.x.f.v
        std::cout << "Fwcvt only supports f32" << std::endl;
        std::abort();
      default:
        std::cout << "Fcvt has unsupported value for first: " << first << std::endl;
        std::abort();
      }
    } else {
      std::cout << "Fcvt only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static R apply(T first, T second, uint32_t vxrm, uint32_t &) { // saturation argument is unused
    // ignoring flags for now
    uint32_t fflags = 0;
    if (sizeof(T) == 8) {
      switch (first) {
      case 0b10000: // vfncvt.xu.f.w
        return rv_ftou_d(second, vxrm, &fflags);
      case 0b10001: // vfncvt.x.f.w
        return rv_ftoi_d(second, vxrm, &fflags);
      case 0b10010: // vfncvt.f.xu.w
        return rv_lutof_s(second, vxrm, &fflags);
      case 0b10011: // vfncvt.f.x.w
        return rv_ltof_s(second, vxrm, &fflags);
      case 0b10100: // vfncvt.f.f.w
        return rv_dtof_r(second, vxrm);
      case 0b10101: // vfncvt.rod.f.f.w
        return rv_dtof_r(second, 6);
      case 0b10110: // vfncvt.rtz.xu.f.w
        return rv_ftou_d(second, 1, &fflags);
      case 0b10111: // vfncvt.rtz.x.f.w
        return rv_ftoi_d(second, 1, &fflags);
      default:
        std::cout << "Fncvt has unsupported value for first: " << first << std::endl;
        std::abort();
      }
    } else {
      std::cout << "Fncvt only supports f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fcvt"; }
};

template <typename T, typename R>
class Funary1 {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Funary1"; }
};

template <typename T, typename R>
class Xunary0 {
public:
  static R apply(T, T second, T) {
    return second;
  }
  static std::string name() { return "Xunary0"; }
};

template <typename T, typename R>
class Feq {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Feq"; }
};

template <typename T, typename R>
class Fle {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fle"; }
};

template <typename T, typename R>
class Flt {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Flt"; }
};

template <typename T, typename R>
class Fne {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fne"; }
};

template <typename T, typename R>
class Fgt {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fgt"; }
};

template <typename T, typename R>
class Fge {
public:
  static R apply(T first, T second, R) {
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
  static std::string name() { return "Fge"; }
};

template <typename T, typename R>
class Fdiv {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(T) == 4) {
      return rv_fdiv_s(second, first, frm, &fflags);
    } else if (sizeof(T) == 8) {
      return rv_fdiv_d(second, first, frm, &fflags);
    } else {
      std::cout << "Fdiv only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fdiv"; }
};

template <typename T, typename R>
class Frdiv {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(T) == 4) {
      return rv_fdiv_s(first, second, frm, &fflags);
    } else if (sizeof(T) == 8) {
      return rv_fdiv_d(first, second, frm, &fflags);
    } else {
      std::cout << "Frdiv only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Frdiv"; }
};

template <typename T, typename R>
class Fmul {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(R) == 4) {
      return rv_fmul_s(first, second, frm, &fflags);
    } else if (sizeof(R) == 8) {
      uint64_t first_d = sizeof(T) == 8 ? first : rv_ftod(first);
      uint64_t second_d = sizeof(T) == 8 ? second : rv_ftod(second);
      return rv_fmul_d(first_d, second_d, frm, &fflags);
    } else {
      std::cout << "Fmul only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Fmul"; }
};

template <typename T, typename R>
class Frsub {
public:
  static R apply(T first, T second, R) {
    // ignoring flags for now
    uint32_t fflags = 0;
    // ignoring rounding mode for now
    uint32_t frm = 0;
    if (sizeof(T) == 4) {
      return rv_fsub_s(first, second, frm, &fflags);
    } else if (sizeof(T) == 8) {
      return rv_fsub_d(first, second, frm, &fflags);
    } else {
      std::cout << "Frsub only supports f32 and f64" << std::endl;
      std::abort();
    }
  }
  static std::string name() { return "Frsub"; }
};

template <typename T, typename R>
class Clip {
public:
  static R apply(T first, T second, uint32_t vxrm, uint32_t &vxsat_) {
    // The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to
    // SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.
    R firstValid = first & (sizeof(T) * 8 - 1);
    T unclippedResult = (second >> firstValid) + roundBit(second, firstValid, vxrm);
    R clippedResult = std::clamp(unclippedResult, (T)std::numeric_limits<R>::min(), (T)std::numeric_limits<R>::max());
    vxsat_ |= clippedResult != unclippedResult;
    return clippedResult;
  }
  static std::string name() { return "Clip"; }
};

template <typename T, typename R>
class Smul {
public:
  static R apply(T first, T second, uint32_t vxrm, uint32_t &vxsat_) {
    R shift = sizeof(R) * 8 - 1;
    T unshiftedResult = first * second;
    T unclippedResult = (unshiftedResult >> shift) + roundBit(unshiftedResult, shift, vxrm);
    R clippedResult = std::clamp(unclippedResult, (T)std::numeric_limits<R>::min(), (T)std::numeric_limits<R>::max());
    vxsat_ |= clippedResult != unclippedResult;
    return clippedResult;
  }
  static std::string name() { return "Smul"; }
};

///////////////////////////////////////////////////////////////////////////////

bool isMasked(std::vector<std::vector<Byte>> &vreg_file, uint32_t maskVreg, uint32_t byteI, bool vmask) {
  auto &mask = vreg_file.at(maskVreg);
  uint8_t emask = *(uint8_t *)(mask.data() + byteI / 8);
  uint8_t value = (emask >> (byteI % 8)) & 0x1;
  DP(4, "Masking enabled: " << +!vmask << " mask element: " << +value);
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
  auto &vr1 = vreg_file.at(getVreg<DT>(baseVreg, byteI));
  return getVregData<DT>(vr1, byteI);
}

template <typename DT>
void vector_op_vix_load(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rdest, uint32_t vl, bool strided, WordI stride, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  uint32_t emul = lmul >> 2 ? 1 : 1 << (lmul & 0b11);
  if (nfields * emul > 8) {
    std::cout << "NFIELDS * EMUL = " << nfields * lmul << " but it should be <= 8" << std::endl;
    std::abort();
  }
  for (uint32_t i = 0; i < vl * nfields; i++) {
    if (isMasked(vreg_file, 0, i / nfields, vmask))
      continue;

    uint32_t nfields_strided = strided ? nfields : 1;
    Word mem_addr = (base_addr & 0xFFFFFFFC) + (i / nfields_strided) * stride + (i % nfields_strided) * sizeof(DT);
    Word mem_data = 0;
    emul_->dcache_read(&mem_data, mem_addr, vsew / 8);
    DP(4, "Loading data " << mem_data << " from: " << mem_addr << " to vec reg: " << getVreg<DT>(rdest + (i % nfields) * emul, i / nfields) << " i: " << i / nfields);
    DT &result = getVregData<DT>(vreg_file, rdest + (i % nfields) * emul, i / nfields);
    DP(4, "Previous data: " << +result);
    result = (DT)mem_data;
  }
}

void vector_op_vix_load(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rdest, uint32_t vsew, uint32_t vl, bool strided, WordI stride, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_load<uint8_t>(vreg_file, emul_, base_addr, rdest, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 16:
    vector_op_vix_load<uint16_t>(vreg_file, emul_, base_addr, rdest, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 32:
    vector_op_vix_load<uint32_t>(vreg_file, emul_, base_addr, rdest, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 64:
    vector_op_vix_load<uint64_t>(vreg_file, emul_, base_addr, rdest, vl, strided, stride, nfields, lmul, vmask);
    break;
  default:
    std::cout << "Failed to execute VLE for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vv_load(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc1, uint32_t rdest, uint32_t iSew, uint32_t vl, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  uint32_t emul = lmul >> 2 ? 1 : 1 << (lmul & 0b11);
  if (nfields * emul > 8) {
    std::cout << "NFIELDS * EMUL = " << nfields * lmul << " but it should be <= 8" << std::endl;
    std::abort();
  }
  for (uint32_t i = 0; i < vl * nfields; i++) {
    if (isMasked(vreg_file, 0, i / nfields, vmask))
      continue;

    Word offset = 0;
    switch (iSew) {
    case 8:
      offset = getVregData<uint8_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 16:
      offset = getVregData<uint16_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 32:
      offset = getVregData<uint32_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 64:
      offset = getVregData<uint64_t>(vreg_file, rsrc1, i / nfields);
      break;
    default:
      std::cout << "Unsupported iSew: " << iSew << std::endl;
      std::abort();
    }

    Word mem_addr = (base_addr & 0xFFFFFFFC) + offset + (i % nfields) * sizeof(DT);
    Word mem_data = 0;
    emul_->dcache_read(&mem_data, mem_addr, vsew / 8);
    DP(4, "VLUX/VLOX - Loading data " << mem_data << " from: " << mem_addr << " with offset: " << std::dec << offset << " to vec reg: " << getVreg<DT>(rdest + (i % nfields) * emul, i / nfields) << " i: " << i / nfields);
    DT &result = getVregData<DT>(vreg_file, rdest + (i % nfields) * emul, i / nfields);
    DP(4, "Previous data: " << +result);
    result = (DT)mem_data;
  }
}

void vector_op_vv_load(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t iSew, uint32_t vl, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_load<uint8_t>(vreg_file, emul_, base_addr, rsrc1, rdest, iSew, vl, nfields, lmul, vmask);
    break;
  case 16:
    vector_op_vv_load<uint16_t>(vreg_file, emul_, base_addr, rsrc1, rdest, iSew, vl, nfields, lmul, vmask);
    break;
  case 32:
    vector_op_vv_load<uint32_t>(vreg_file, emul_, base_addr, rsrc1, rdest, iSew, vl, nfields, lmul, vmask);
    break;
  case 64:
    vector_op_vv_load<uint64_t>(vreg_file, emul_, base_addr, rsrc1, rdest, iSew, vl, nfields, lmul, vmask);
    break;
  default:
    std::cout << "Failed to execute VLUX/VLOX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vix_store(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc3, uint32_t vl, bool strided, WordI stride, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  uint32_t emul = lmul >> 2 ? 1 : 1 << (lmul & 0b11);
  for (uint32_t i = 0; i < vl * nfields; i++) {
    if (isMasked(vreg_file, 0, i / nfields, vmask))
      continue;

    uint32_t nfields_strided = strided ? nfields : 1;
    Word mem_addr = base_addr + (i / nfields_strided) * stride + (i % nfields_strided) * sizeof(DT);
    Word mem_data = getVregData<DT>(vreg_file, rsrc3 + (i % nfields) * emul, i / nfields);
    DP(4, "Storing: " << std::hex << mem_data << " at: " << mem_addr << " from vec reg: " << getVreg<DT>(rsrc3 + (i % nfields) * emul, i / nfields) << " i: " << i / nfields);
    emul_->dcache_write(&mem_data, mem_addr, vsew / 8);
  }
}

void vector_op_vix_store(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc3, uint32_t vsew, uint32_t vl, bool strided, WordI stride, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_store<uint8_t>(vreg_file, emul_, base_addr, rsrc3, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 16:
    vector_op_vix_store<uint16_t>(vreg_file, emul_, base_addr, rsrc3, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 32:
    vector_op_vix_store<uint32_t>(vreg_file, emul_, base_addr, rsrc3, vl, strided, stride, nfields, lmul, vmask);
    break;
  case 64:
    vector_op_vix_store<uint64_t>(vreg_file, emul_, base_addr, rsrc3, vl, strided, stride, nfields, lmul, vmask);
    break;
  default:
    std::cout << "Failed to execute VSE for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vv_store(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc1, uint32_t rsrc3, uint32_t iSew, uint32_t vl, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  uint32_t vsew = sizeof(DT) * 8;
  uint32_t emul = lmul >> 2 ? 1 : 1 << (lmul & 0b11);
  for (uint32_t i = 0; i < vl * nfields; i++) {
    if (isMasked(vreg_file, 0, i / nfields, vmask))
      continue;

    Word offset = 0;
    switch (iSew) {
    case 8:
      offset = getVregData<uint8_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 16:
      offset = getVregData<uint16_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 32:
      offset = getVregData<uint32_t>(vreg_file, rsrc1, i / nfields);
      break;
    case 64:
      offset = getVregData<uint64_t>(vreg_file, rsrc1, i / nfields);
      break;
    default:
      std::cout << "Unsupported iSew: " << iSew << std::endl;
      std::abort();
    }

    Word mem_addr = base_addr + offset + (i % nfields) * sizeof(DT);
    Word mem_data = getVregData<DT>(vreg_file, rsrc3 + (i % nfields) * emul, i / nfields);
    DP(4, "VSUX/VSOX - Storing: " << std::hex << mem_data << " at: " << mem_addr << " with offset: " << std::dec << offset << " from vec reg: " << getVreg<DT>(rsrc3 + (i % nfields) * emul, i / nfields) << " i: " << i / nfields);
    emul_->dcache_write(&mem_data, mem_addr, vsew / 8);
  }
}

void vector_op_vv_store(std::vector<std::vector<Byte>> &vreg_file, vortex::Emulator *emul_, WordI base_addr, uint32_t rsrc1, uint32_t rsrc3, uint32_t vsew, uint32_t iSew, uint32_t vl, uint32_t nfields, uint32_t lmul, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_store<uint8_t>(vreg_file, emul_, base_addr, rsrc1, rsrc3, iSew, vl, nfields, lmul, vmask);
    break;
  case 16:
    vector_op_vv_store<uint16_t>(vreg_file, emul_, base_addr, rsrc1, rsrc3, iSew, vl, nfields, lmul, vmask);
    break;
  case 32:
    vector_op_vv_store<uint32_t>(vreg_file, emul_, base_addr, rsrc1, rsrc3, iSew, vl, nfields, lmul, vmask);
    break;
  case 64:
    vector_op_vv_store<uint64_t>(vreg_file, emul_, base_addr, rsrc1, rsrc3, iSew, vl, nfields, lmul, vmask);
    break;
  default:
    std::cout << "Failed to execute VSUX/VSOX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DT third = getVregData<DT>(vreg_file, rdest, i);
    DT result = OP<DT, DT>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix<OP, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vix<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix_carry(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl) {
  for (uint32_t i = 0; i < vl; i++) {
    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    bool third = !isMasked(vreg_file, 0, i, false);
    DT result = OP<DT, DT>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_carry(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl) {
  switch (vsew) {
  case 8:
    vector_op_vix_carry<OP, DT8>(src1, vreg_file, rsrc0, rdest, vl);
    break;
  case 16:
    vector_op_vix_carry<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl);
    break;
  case 32:
    vector_op_vix_carry<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl);
    break;
  case 64:
    vector_op_vix_carry<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl);
    break;
  default:
    std::cout << "Failed to execute VI/VX carry for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_carry_out(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    bool third = !vmask && !isMasked(vreg_file, 0, i, vmask);
    bool result = OP<DT, DTR>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64, typename DT128>
void vector_op_vix_carry_out(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_carry_out<OP, DT8, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix_carry_out<OP, DT16, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix_carry_out<OP, DT32, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vix_carry_out<OP, DT64, DT128>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX carry out for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vix_merge(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    DT result = isMasked(vreg_file, 0, i, vmask) ? getVregData<DT>(vreg_file, rsrc0, i) : first;
    DP(4, "Merge - Choosing result: " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_merge(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_merge<DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix_merge<DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix_merge<DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vix_merge<DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_scalar(DT &dest, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t vsew) {
  if (rsrc0 != 0) {
    std::cout << "Vwxunary0/Vwfunary0 has unsupported value for vs2: " << rsrc0 << std::endl;
    std::abort();
  }
  switch (vsew) {
  case 8:
    dest = getVregData<uint8_t>(vreg_file, rsrc1, 0);
    break;
  case 16:
    dest = getVregData<uint16_t>(vreg_file, rsrc1, 0);
    break;
  case 32:
    dest = getVregData<uint32_t>(vreg_file, rsrc1, 0);
    break;
  case 64:
    dest = getVregData<uint64_t>(vreg_file, rsrc1, 0);
    break;
  default:
    std::cout << "Failed to execute vmv.x.s/vfmv.f.s for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_w(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DTR third = getVregData<DTR>(vreg_file, rdest, i);
    DTR result = OP<DT, DTR>::apply(first, second, third);
    DP(4, "Widening " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_w(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_w<OP, DT8, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix_w<OP, DT16, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix_w<OP, DT32, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX widening for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_wx(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX widening wx for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_n(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(4, "Narrowing " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_n(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vix_n<OP, DT16, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vix_n<OP, DT32, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vix_n<OP, DT64, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VI/VX narrowing for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vix_sat(DTR first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT second = getVregData<DTR>(vreg_file, rsrc0, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(4, "Saturating " << (OP<DT, DTR>::name()) << "(" << +(DTR)first << ", " << +(DTR)second << ")" << " = " << +(DTR)result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64, typename DT128>
void vector_op_vix_sat(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vix_sat<OP, DT16, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vix_sat<OP, DT32, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vix_sat<OP, DT64, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 64:
    vector_op_vix_sat<OP, DT128, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VI/VX saturating for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_scale(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vix_sat<OP, DT8, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vix_sat<OP, DT16, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vix_sat<OP, DT32, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 64:
    vector_op_vix_sat<OP, DT64, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VI/VX scale for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP>
void vector_op_vix_ext(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  if (vsew == 16) {
    switch (src1) {
    case 0b00110: // vzext.vf2
      vector_op_vix_w<OP, uint8_t, uint16_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00111: // vsext.vf2
      vector_op_vix_w<OP, int8_t, int16_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    default:
      std::cout << "Xunary0 has unsupported value for vf: " << src1 << std::endl;
      std::abort();
    }
  } else if (vsew == 32) {
    switch (src1) {
    case 0b00100: // vzext.vf4
      vector_op_vix_w<OP, uint8_t, uint32_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00101: // vsext.vf4
      vector_op_vix_w<OP, int8_t, int32_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00110: // vzext.vf2
      vector_op_vix_w<OP, uint16_t, uint32_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00111: // vsext.vf2
      vector_op_vix_w<OP, int16_t, int32_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    default:
      std::cout << "Xunary0 has unsupported value for vf: " << src1 << std::endl;
      std::abort();
    }
  } else if (vsew == 64) {
    switch (src1) {
    case 0b00010: // vzext.vf8
      vector_op_vix_w<OP, uint8_t, uint64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00011: // vsext.vf8
      vector_op_vix_w<OP, int8_t, int64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00100: // vzext.vf4
      vector_op_vix_w<OP, uint16_t, uint64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00101: // vsext.vf4
      vector_op_vix_w<OP, int16_t, int64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00110: // vzext.vf2
      vector_op_vix_w<OP, uint32_t, uint64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    case 0b00111: // vsext.vf2
      vector_op_vix_w<OP, int32_t, int64_t>(src1, vreg_file, rsrc0, rdest, vl, vmask);
      break;
    default:
      std::cout << "Xunary0 has unsupported value for vf: " << src1 << std::endl;
      std::abort();
    }
  } else {
    std::cout << "Failed to execute Xunary0 for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vix_mask(DT first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT second = getVregData<DT>(vreg_file, rsrc0, i);
    bool result = OP<DT, bool>::apply(first, second, 0);
    DP(4, "Integer/float compare mask " << (OP<DT, bool>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_mask(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_mask<OP, DT8>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vix_mask<OP, DT16>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vix_mask<OP, DT32>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vix_mask<OP, DT64>(src1, vreg_file, rsrc0, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX integer/float compare mask for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vix_slide(Word first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, Word vlmax, uint32_t vmask, bool scalar) {
  // If vlmax > 0 this means we have a vslidedown instruction, vslideup does not require vlmax
  bool slideDown = vlmax;
  uint32_t scalarPos = slideDown ? vl - 1 : 0;
  // If scalar set is set this means we have a v(f)slide1up or v(f)slide1down instruction,
  // so first is our scalar value and we need to overwrite it with 1 for later computations
  if (scalar && vl && !isMasked(vreg_file, 0, scalarPos, vmask)) {
    DP(4, "Slide - Moving scalar value " << +first << " to position " << +scalarPos);
    getVregData<DT>(vreg_file, rdest, scalarPos) = first;
  }
  first = scalar ? 1 : first;

  for (Word i = slideDown ? 0 : first; i < vl - (scalar && vl && slideDown); i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    __uint128_t iSrc = slideDown ? (__uint128_t)i + (__uint128_t)first : (__uint128_t)i - (__uint128_t)first; // prevent overflows/underflows
    DT value = (!slideDown || iSrc < vlmax) ? getVregData<DT>(vreg_file, rsrc0, iSrc) : 0;
    DP(4, "Slide - Moving value " << +value << " from position " << (uint64_t)iSrc << " to position " << +i);
    getVregData<DT>(vreg_file, rdest, i) = value;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_slide(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, Word vlmax, uint32_t vmask, bool scalar) {
  switch (vsew) {
  case 8:
    vector_op_vix_slide<DT8>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask, scalar);
    break;
  case 16:
    vector_op_vix_slide<DT16>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask, scalar);
    break;
  case 32:
    vector_op_vix_slide<DT32>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask, scalar);
    break;
  case 64:
    vector_op_vix_slide<DT64>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask, scalar);
    break;
  default:
    std::cout << "Failed to execute VI/VX slide for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vix_gather(Word first, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vl, Word vlmax, uint32_t vmask) {
  for (Word i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT value = first < vlmax ? getVregData<DT>(vreg_file, rsrc0, first) : 0;
    DP(4, "Register gather - Moving value " << +value << " from position " << +first << " to position " << +i);
    getVregData<DT>(vreg_file, rdest, i) = value;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vix_gather(Word src1, std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rdest, uint32_t vsew, uint32_t vl, Word vlmax, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vix_gather<DT8>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask);
    break;
  case 16:
    vector_op_vix_gather<DT16>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask);
    break;
  case 32:
    vector_op_vix_gather<DT32>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask);
    break;
  case 64:
    vector_op_vix_gather<DT64>(src1, vreg_file, rsrc0, rdest, vl, vlmax, vmask);
    break;
  default:
    std::cout << "Failed to execute VI/VX register gather for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DT third = getVregData<DT>(vreg_file, rdest, i);
    DT result = OP<DT, DT>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vv<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv_carry(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl) {
  for (uint32_t i = 0; i < vl; i++) {
    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    bool third = !isMasked(vreg_file, 0, i, false);
    DT result = OP<DT, DT>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_carry(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl) {
  switch (vsew) {
  case 8:
    vector_op_vv_carry<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 16:
    vector_op_vv_carry<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 32:
    vector_op_vv_carry<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 64:
    vector_op_vv_carry<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  default:
    std::cout << "Failed to execute VV carry for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_carry_out(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    bool third = !vmask && !isMasked(vreg_file, 0, i, vmask);
    bool result = OP<DT, DTR>::apply(first, second, third);
    DP(4, (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64, typename DT128>
void vector_op_vv_carry_out(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_carry_out<OP, DT8, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_carry_out<OP, DT16, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_carry_out<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vv_carry_out<OP, DT64, DT128>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV carry out for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vv_merge(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    uint32_t rsrc = isMasked(vreg_file, 0, i, vmask) ? rsrc1 : rsrc0;
    DT result = getVregData<DT>(vreg_file, rsrc, i);
    DP(4, "Merge - Choosing result: " << +result);
    getVregData<DT>(vreg_file, rdest, i) = result;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_merge(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_merge<DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_merge<DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_merge<DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vv_merge<DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vv_gather(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, bool ei16, uint32_t vlmax, uint32_t vmask) {
  for (Word i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    uint32_t first = ei16 ? getVregData<uint16_t>(vreg_file, rsrc0, i) : getVregData<DT>(vreg_file, rsrc0, i);
    DT value = first < vlmax ? getVregData<DT>(vreg_file, rsrc1, first) : 0;
    DP(4, "Register gather - Moving value " << +value << " from position " << +first << " to position " << +i);
    getVregData<DT>(vreg_file, rdest, i) = value;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_gather(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, bool ei16, uint32_t vlmax, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_gather<DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, ei16, vlmax, vmask);
    break;
  case 16:
    vector_op_vv_gather<DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, ei16, vlmax, vmask);
    break;
  case 32:
    vector_op_vv_gather<DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, ei16, vlmax, vmask);
    break;
  case 64:
    vector_op_vv_gather<DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, ei16, vlmax, vmask);
    break;
  default:
    std::cout << "Failed to execute VV register gather for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR third = getVregData<DTR>(vreg_file, rdest, i);
    DTR result = OP<DT, DTR>::apply(first, second, third);
    DP(4, "Widening " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_w<OP, DT8, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_w<OP, DT16, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_w<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV widening for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_wv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DTR second = getVregData<DTR>(vreg_file, rsrc1, i);
    DTR third = getVregData<DTR>(vreg_file, rdest, i);
    DTR result = OP<DTR, DTR>::apply(first, second, third);
    DP(4, "Widening wv " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_wv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_wv<OP, DT8, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_wv<OP, DT16, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_wv<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV widening wv for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_wfv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DTR second = getVregData<DTR>(vreg_file, rsrc1, i);
    DTR third = getVregData<DTR>(vreg_file, rdest, i);
    DTR result = OP<DTR, DTR>::apply(rv_ftod(first), second, third);
    DP(4, "Widening wfv " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ", " << +third << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_wfv(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  if (vsew == 32) {
    vector_op_vv_wfv<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV widening wfv for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_n(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DTR first = getVregData<DTR>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(4, "Narrowing " << (OP<DT, DTR>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_n(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vv_n<OP, DT16, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vv_n<OP, DT32, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vv_n<OP, DT64, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VV narrowing for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_sat(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DTR>(vreg_file, rsrc0, i);
    DT second = getVregData<DTR>(vreg_file, rsrc1, i);
    DTR result = OP<DT, DTR>::apply(first, second, vxrm, vxsat);
    DP(4, "Saturating " << (OP<DT, DTR>::name()) << "(" << +(DTR)first << ", " << +(DTR)second << ")" << " = " << +(DTR)result);
    getVregData<DTR>(vreg_file, rdest, i) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64, typename DT128>
void vector_op_vv_sat(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vv_sat<OP, DT16, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vv_sat<OP, DT32, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vv_sat<OP, DT64, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 64:
    vector_op_vv_sat<OP, DT128, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VV saturating for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_scale(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask, uint32_t vxrm, uint32_t &vxsat) {
  switch (vsew) {
  case 8:
    vector_op_vv_sat<OP, DT8, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 16:
    vector_op_vv_sat<OP, DT16, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 32:
    vector_op_vv_sat<OP, DT32, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  case 64:
    vector_op_vv_sat<OP, DT64, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask, vxrm, vxsat);
    break;
  default:
    std::cout << "Failed to execute VV scale for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv_red(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    // use rdest as accumulator
    if (i == 0) {
      getVregData<DT>(vreg_file, rdest, 0) = getVregData<DT>(vreg_file, rsrc0, 0);
    }
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rdest, 0);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DT result = OP<DT, DT>::apply(first, second, 0);
    DP(4, "Reduction " << (OP<DT, DT>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    getVregData<DT>(vreg_file, rdest, 0) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_red(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_red<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_red<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_red<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vv_red<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV reduction for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_red_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    // use rdest as accumulator
    if (i == 0) {
      getVregData<DTR>(vreg_file, rdest, 0) = getVregData<DTR>(vreg_file, rsrc0, 0);
    }
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DTR first = getVregData<DTR>(vreg_file, rdest, 0);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR second_w = std::is_signed<DT>() ? sext((DTR)second, sizeof(DT) * 8) : zext((DTR)second, sizeof(DT) * 8);
    DTR result = OP<DTR, DTR>::apply(first, second_w, 0);
    DP(4, "Widening reduction " << (OP<DTR, DTR>::name()) << "(" << +first << ", " << +second_w << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, 0) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_red_w(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_red_w<OP, DT8, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_red_w<OP, DT16, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_red_w<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV widening reduction for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT, typename DTR>
void vector_op_vv_red_wf(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    // use rdest as accumulator
    if (i == 0) {
      getVregData<DTR>(vreg_file, rdest, 0) = getVregData<DTR>(vreg_file, rsrc0, 0);
    }
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DTR first = getVregData<DTR>(vreg_file, rdest, 0);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    DTR second_w = rv_ftod(second);
    DTR result = OP<DTR, DTR>::apply(first, second_w, 0);
    DP(4, "Float widening reduction " << (OP<DTR, DTR>::name()) << "(" << +first << ", " << +second_w << ")" << " = " << +result);
    getVregData<DTR>(vreg_file, rdest, 0) = result;
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_red_wf(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  if (vsew == 32) {
    vector_op_vv_red_wf<OP, DT32, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
  } else {
    std::cout << "Failed to execute VV float widening reduction for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <typename DT>
void vector_op_vid(std::vector<std::vector<Byte>> &vreg_file, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DP(4, "Element Index = " << +i);
    getVregData<DT>(vreg_file, rdest, i) = i;
  }
}

void vector_op_vid(std::vector<std::vector<Byte>> &vreg_file, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vid<uint8_t>(vreg_file, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vid<uint16_t>(vreg_file, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vid<uint32_t>(vreg_file, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vid<uint64_t>(vreg_file, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute vector element index for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl, uint32_t vmask) {
  for (uint32_t i = 0; i < vl; i++) {
    if (isMasked(vreg_file, 0, i, vmask))
      continue;

    DT first = getVregData<DT>(vreg_file, rsrc0, i);
    DT second = getVregData<DT>(vreg_file, rsrc1, i);
    bool result = OP<DT, bool>::apply(first, second, 0);
    DP(4, "Integer/float compare mask " << (OP<DT, bool>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <template <typename DT1, typename DT2> class OP, typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask) {
  switch (vsew) {
  case 8:
    vector_op_vv_mask<OP, DT8>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 16:
    vector_op_vv_mask<OP, DT16>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 32:
    vector_op_vv_mask<OP, DT32>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  case 64:
    vector_op_vv_mask<OP, DT64>(vreg_file, rsrc0, rsrc1, rdest, vl, vmask);
    break;
  default:
    std::cout << "Failed to execute VV integer/float compare mask for vsew: " << vsew << std::endl;
    std::abort();
  }
}

template <template <typename DT1, typename DT2> class OP>
void vector_op_vv_mask(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl) {
  for (uint32_t i = 0; i < vl; i++) {
    uint8_t firstMask = getVregData<uint8_t>(vreg_file, rsrc0, i / 8);
    bool first = (firstMask >> (i % 8)) & 0x1;
    uint8_t secondMask = getVregData<uint8_t>(vreg_file, rsrc1, i / 8);
    bool second = (secondMask >> (i % 8)) & 0x1;
    bool result = OP<uint8_t, uint8_t>::apply(first, second, 0) & 0x1;
    DP(4, "Compare mask bits " << (OP<uint8_t, uint8_t>::name()) << "(" << +first << ", " << +second << ")" << " = " << +result);
    if (result) {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) |= 1 << (i % 8);
    } else {
      getVregData<uint8_t>(vreg_file, rdest, i / 8) &= ~(1 << (i % 8));
    }
  }
}

template <typename DT>
void vector_op_vv_compress(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vl) {
  int currPos = 0;
  for (uint32_t i = 0; i < vl; i++) {
    // Special case: use rsrc0 as mask vector register instead of default v0
    // This instruction is always masked (vmask == 0), but encoded as unmasked (vmask == 1)
    if (isMasked(vreg_file, rsrc0, i, 0))
      continue;

    DT value = getVregData<DT>(vreg_file, rsrc1, i);
    DP(4, "Compression - Moving value " << +value << " from position " << i << " to position " << currPos);
    getVregData<DT>(vreg_file, rdest, currPos) = value;
    currPos++;
  }
}

template <typename DT8, typename DT16, typename DT32, typename DT64>
void vector_op_vv_compress(std::vector<std::vector<Byte>> &vreg_file, uint32_t rsrc0, uint32_t rsrc1, uint32_t rdest, uint32_t vsew, uint32_t vl) {
  switch (vsew) {
  case 8:
    vector_op_vv_compress<DT8>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 16:
    vector_op_vv_compress<DT16>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 32:
    vector_op_vv_compress<DT32>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  case 64:
    vector_op_vv_compress<DT64>(vreg_file, rsrc0, rsrc1, rdest, vl);
    break;
  default:
    std::cout << "Failed to execute VV compression for vsew: " << vsew << std::endl;
    std::abort();
  }
}

void Emulator::loadVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
  auto &warp = warps_.at(wid);
  auto vmask = instr.getVmask();
  auto rdest = instr.getRDest();
  auto mop = instr.getVmop();
  switch (mop) {
  case 0b00: { // unit-stride
    auto lumop = instr.getVumop();
    switch (lumop) {
    case 0b10000:  // vle8ff.v, vle16ff.v, vle32ff.v, vle64ff.v - we do not support exceptions -> treat like regular unit stride
                   // vlseg2e8ff.v, vlseg2e16ff.v, vlseg2e32ff.v, vlseg2e64ff.v
                   // vlseg3e8ff.v, vlseg3e16ff.v, vlseg3e32ff.v, vlseg3e64ff.v
                   // vlseg4e8ff.v, vlseg4e16ff.v, vlseg4e32ff.v, vlseg4e64ff.v
                   // vlseg5e8ff.v, vlseg5e16ff.v, vlseg5e32ff.v, vlseg5e64ff.v
                   // vlseg6e8ff.v, vlseg6e16ff.v, vlseg6e32ff.v, vlseg6e64ff.v
                   // vlseg7e8ff.v, vlseg7e16ff.v, vlseg7e32ff.v, vlseg7e64ff.v
                   // vlseg8e8ff.v, vlseg8e16ff.v, vlseg8e32ff.v, vlseg8e64ff.v
    case 0b0000: { // vle8.v, vle16.v, vle32.v, vle64.v
                   // vlseg2e8.v, vlseg2e16.v, vlseg2e32.v, vlseg2e64.v
                   // vlseg3e8.v, vlseg3e16.v, vlseg3e32.v, vlseg3e64.v
                   // vlseg4e8.v, vlseg4e16.v, vlseg4e32.v, vlseg4e64.v
                   // vlseg5e8.v, vlseg5e16.v, vlseg5e32.v, vlseg5e64.v
                   // vlseg6e8.v, vlseg6e16.v, vlseg6e32.v, vlseg6e64.v
                   // vlseg7e8.v, vlseg7e16.v, vlseg7e32.v, vlseg7e64.v
                   // vlseg8e8.v, vlseg8e16.v, vlseg8e32.v, vlseg8e64.v
      WordI stride = warp.vtype.vsew / 8;
      uint32_t nfields = instr.getVnf() + 1;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, warp.vl, false, stride, nfields, warp.vtype.vlmul, vmask);
      break;
    }
    case 0b1000: { // vl1r.v, vl2r.v, vl4r.v, vl8r.v
      uint32_t nreg = instr.getVnf() + 1;
      if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
        std::cout << "Whole vector register load - reserved value for nreg: " << nreg << std::endl;
        std::abort();
      }
      DP(4, "Whole vector register load with nreg: " << nreg);
      uint32_t stride = 1 << instr.getVsew();
      uint32_t vsew_bits = stride * 8;
      uint32_t vl = nreg * VLEN / vsew_bits;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, vsew_bits, vl, false, stride, 1, 0, vmask);
      break;
    }
    case 0b1011: { // vlm.v
      if (warp.vtype.vsew != 8) {
        std::cout << "vlm.v only supports SEW=8, but SEW was: " << warp.vtype.vsew << std::endl;
        std::abort();
      }
      WordI stride = warp.vtype.vsew / 8;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, (warp.vl + 7) / 8, false, stride, 1, 0, true);
      break;
    }
    default:
      std::cout << "Load vector - unsupported lumop: " << lumop << std::endl;
      std::abort();
    }
    break;
  }
  case 0b10: { // strided: vlse8.v, vlse16.v, vlse32.v, vlse64.v
               // vlsseg2e8.v, vlsseg2e16.v, vlsseg2e32.v, vlsseg2e64.v
               // vlsseg3e8.v, vlsseg3e16.v, vlsseg3e32.v, vlsseg3e64.v
               // vlsseg4e8.v, vlsseg4e16.v, vlsseg4e32.v, vlsseg4e64.v
               // vlsseg5e8.v, vlsseg5e16.v, vlsseg5e32.v, vlsseg5e64.v
               // vlsseg6e8.v, vlsseg6e16.v, vlsseg6e32.v, vlsseg6e64.v
               // vlsseg7e8.v, vlsseg7e16.v, vlsseg7e32.v, vlsseg7e64.v
               // vlsseg8e8.v, vlsseg8e16.v, vlsseg8e32.v, vlsseg8e64.v
    auto rsrc1 = instr.getRSrc(1);
    auto rdest = instr.getRDest();
    WordI stride = warp.ireg_file.at(0).at(rsrc1);
    uint32_t nfields = instr.getVnf() + 1;
    vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, warp.vl, true, stride, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  case 0b01:   // indexed - unordered, vluxei8.v, vluxei16.v, vluxei32.v, vluxei64.v
               // vluxseg2e8.v, vluxseg2e16.v, vluxseg2e32.v, vluxseg2e64.v
               // vluxseg3e8.v, vluxseg3e16.v, vluxseg3e32.v, vluxseg3e64.v
               // vluxseg4e8.v, vluxseg4e16.v, vluxseg4e32.v, vluxseg4e64.v
               // vluxseg5e8.v, vluxseg5e16.v, vluxseg5e32.v, vluxseg5e64.v
               // vluxseg6e8.v, vluxseg6e16.v, vluxseg6e32.v, vluxseg6e64.v
               // vluxseg7e8.v, vluxseg7e16.v, vluxseg7e32.v, vluxseg7e64.v
               // vluxseg8e8.v, vluxseg8e16.v, vluxseg8e32.v, vluxseg8e64.v
  case 0b11: { // indexed - ordered, vloxei8.v, vloxei16.v, vloxei32.v, vloxei64.v
               // vloxseg2e8.v, vloxseg2e16.v, vloxseg2e32.v, vloxseg2e64.v
               // vloxseg3e8.v, vloxseg3e16.v, vloxseg3e32.v, vloxseg3e64.v
               // vloxseg4e8.v, vloxseg4e16.v, vloxseg4e32.v, vloxseg4e64.v
               // vloxseg5e8.v, vloxseg5e16.v, vloxseg5e32.v, vloxseg5e64.v
               // vloxseg6e8.v, vloxseg6e16.v, vloxseg6e32.v, vloxseg6e64.v
               // vloxseg7e8.v, vloxseg7e16.v, vloxseg7e32.v, vloxseg7e64.v
               // vloxseg8e8.v, vloxseg8e16.v, vloxseg8e32.v, vloxseg8e64.v
    uint32_t nfields = instr.getVnf() + 1;
    uint32_t vsew_bits = 1 << (3 + instr.getVsew());
    vector_op_vv_load(warp.vreg_file, this, rsdata[0][0].i, instr.getRSrc(1), rdest, warp.vtype.vsew, vsew_bits, warp.vl, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  default:
    std::cout << "Load vector - unsupported mop: " << mop << std::endl;
    std::abort();
  }
}

void Emulator::storeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
  auto &warp = warps_.at(wid);
  auto vmask = instr.getVmask();
  auto mop = instr.getVmop();
  switch (mop) {
  case 0b00: { // unit-stride
    auto vs3 = instr.getRSrc(1);
    auto sumop = instr.getVumop();
    WordI stride = warp.vtype.vsew / 8;
    switch (sumop) {
    case 0b0000: { // vse8.v, vse16.v, vse32.v, vse64.v
      uint32_t nfields = instr.getVnf() + 1;
      vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, warp.vl, false, stride, nfields, warp.vtype.vlmul, vmask);
      break;
    }
    case 0b1000: { // vs1r.v, vs2r.v, vs4r.v, vs8r.v
      uint32_t nreg = instr.getVnf() + 1;
      if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
        std::cout << "Whole vector register store - reserved value for nreg: " << nreg << std::endl;
        std::abort();
      }
      DP(4, "Whole vector register store with nreg: " << nreg);
      uint32_t vl = nreg * VLEN / 8;
      vector_op_vix_store<uint8_t>(warp.vreg_file, this, rsdata[0][0].i, vs3, vl, false, stride, 1, 0, vmask);
      break;
    }
    case 0b1011: { // vsm.v
      if (warp.vtype.vsew != 8) {
        std::cout << "vsm.v only supports EEW=8, but EEW was: " << warp.vtype.vsew << std::endl;
        std::abort();
      }
      vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, (warp.vl + 7) / 8, false, stride, 1, 0, true);
      break;
    }
    default:
      std::cout << "Store vector - unsupported sumop: " << sumop << std::endl;
      std::abort();
    }
    break;
  }
  case 0b10: { // strided: vsse8.v, vsse16.v, vsse32.v, vsse64.v
               // vssseg2e8.v, vssseg2e16.v, vssseg2e32.v, vssseg2e64.v
               // vssseg3e8.v, vssseg3e16.v, vssseg3e32.v, vssseg3e64.v
               // vssseg4e8.v, vssseg4e16.v, vssseg4e32.v, vssseg4e64.v
               // vssseg5e8.v, vssseg5e16.v, vssseg5e32.v, vssseg5e64.v
               // vssseg6e8.v, vssseg6e16.v, vssseg6e32.v, vssseg6e64.v
               // vssseg7e8.v, vssseg7e16.v, vssseg7e32.v, vssseg7e64.v
               // vssseg8e8.v, vssseg8e16.v, vssseg8e32.v, vssseg8e64.v
    auto rsrc1 = instr.getRSrc(1);
    auto vs3 = instr.getRSrc(2);
    WordI stride = warp.ireg_file.at(0).at(rsrc1);
    uint32_t nfields = instr.getVnf() + 1;
    vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, warp.vl, true, stride, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  case 0b01:   // indexed - unordered, vsuxei8.v, vsuxei16.v, vsuxei32.v, vsuxei64.v
               // vsuxseg2ei8.v, vsuxseg2ei16.v, vsuxseg2ei32.v, vsuxseg2ei64.v
               // vsuxseg3ei8.v, vsuxseg3ei16.v, vsuxseg3ei32.v, vsuxseg3ei64.v
               // vsuxseg4ei8.v, vsuxseg4ei16.v, vsuxseg4ei32.v, vsuxseg4ei64.v
               // vsuxseg5ei8.v, vsuxseg5ei16.v, vsuxseg5ei32.v, vsuxseg5ei64.v
               // vsuxseg6ei8.v, vsuxseg6ei16.v, vsuxseg6ei32.v, vsuxseg6ei64.v
               // vsuxseg7ei8.v, vsuxseg7ei16.v, vsuxseg7ei32.v, vsuxseg7ei64.v
               // vsuxseg8ei8.v, vsuxseg8ei16.v, vsuxseg8ei32.v, vsuxseg8ei64.v
  case 0b11: { // indexed - ordered, vsoxei8.v, vsoxei16.v, vsoxei32.v, vsoxei64.v
               // vsoxseg2ei8.v, vsoxseg2ei16.v, vsoxseg2ei32.v, vsoxseg2ei64.v
               // vsoxseg3ei8.v, vsoxseg3ei16.v, vsoxseg3ei32.v, vsoxseg3ei64.v
               // vsoxseg4ei8.v, vsoxseg4ei16.v, vsoxseg4ei32.v, vsoxseg4ei64.v
               // vsoxseg5ei8.v, vsoxseg5ei16.v, vsoxseg5ei32.v, vsoxseg5ei64.v
               // vsoxseg6ei8.v, vsoxseg6ei16.v, vsoxseg6ei32.v, vsoxseg6ei64.v
               // vsoxseg7ei8.v, vsoxseg7ei16.v, vsoxseg7ei32.v, vsoxseg7ei64.v
               // vsoxseg8ei8.v, vsoxseg8ei16.v, vsoxseg8ei32.v, vsoxseg8ei64.v
    uint32_t nfields = instr.getVnf() + 1;
    uint32_t vsew_bits = 1 << (3 + instr.getVsew());
    vector_op_vv_store(warp.vreg_file, this, rsdata[0][0].i, instr.getRSrc(1), instr.getRSrc(2), warp.vtype.vsew, vsew_bits, warp.vl, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  default:
    std::cout << "Store vector - unsupported mop: " << mop << std::endl;
    std::abort();
  }
}

void Emulator::executeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata) {
  auto &warp = warps_.at(wid);
  auto func3 = instr.getFunc3();
  auto func6 = instr.getFunc6();

  auto rdest = instr.getRDest();
  auto rsrc0 = instr.getRSrc(0);
  auto rsrc1 = instr.getRSrc(1);
  auto immsrc = sext((Word)instr.getImm(), width_reg);
  auto uimmsrc = (Word)instr.getImm();
  auto vmask = instr.getVmask();
  auto num_threads = arch_.num_threads();

  switch (func3) {
  case 0: { // vector - vector
    switch (func6) {
    case 0: { // vadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vminu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Min, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vmin.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Min, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vmaxu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Max, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vmax.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Max, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<And, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Or, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Xor, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, false, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vrgatherei16.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, true, warp.vlmax, vmask);
      }
    } break;
    case 16: { // vadc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vv, vmadc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 18: { // vsbc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 19: { // vmsbc.vv, vmsbc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.v
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.v vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vvm
          vector_op_vv_merge<int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Eq, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Ne, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Lt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Le, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Gt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 34: { // vssubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 35: { // vssub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Sll, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vsmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 40: { // vsrl.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_n<Clip, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 48: { // vwredsumu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwredsum.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_w<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
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
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vfsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 1:   // vfredusum.vs - treated the same as vfredosum.vs
    case 3: { // vfredosum.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fadd, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vfmin.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vfredmin.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fmin, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vfmax.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vfredmax.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fmax, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vfsgnj.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vfsgnjn.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vfsgnjx.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 16: { // vfmv.f.s
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &dest = rddata[t].u64;
        vector_op_scalar(dest, warp.vreg_file, rsrc0, rsrc1, warp.vtype.vsew);
        DP(4, "Moved " << +dest << " from: " << +rsrc1 << " to: " << +rdest);
      }
    } break;
    case 18: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        switch (rsrc0 >> 3) {
        case 0b00: // vfcvt.xu.f.v, vfcvt.x.f.v, vfcvt.f.xu.v, vfcvt.f.x.v, vfcvt.rtz.xu.f.v, vfcvt.rtz.x.f.v
          vector_op_vix<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
          break;
        case 0b01: // vfwcvt.xu.f.v, vfwcvt.x.f.v, vfwcvt.f.xu.v, vfwcvt.f.x.v, vfwcvt.f.f.v, vfwcvt.rtz.xu.f.v, vfwcvt.rtz.x.f.v
          vector_op_vix_w<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
          break;
        case 0b10: { // vfncvt.xu.f.w, vfncvt.x.f.w, vfncvt.f.xu.w, vfncvt.f.x.w, vfncvt.f.f.w, vfncvt.rod.f.f.w, vfncvt.rtz.xu.f.w, vfncvt.rtz.x.f.w
          uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
          uint32_t vxsat = 0; // saturation argument is unused
          vector_op_vix_n<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
          break;
        }
        default:
          std::cout << "Fcvt unsupported value for rsrc0: " << rsrc0 << std::endl;
          std::abort();
        }
      }
    } break;
    case 19: { // vfsqrt.v, vfrsqrt7.v, vfrec7.v, vfclass.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Funary1, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 24: { // vmfeq.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Feq, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmfle.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Fle, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmflt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Flt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmfne.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Fne, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vfdiv.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fdiv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vfmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 40: { // vfmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vfnmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vfmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vfnmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 44: { // vfmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vfnmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 46: { // vfmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vfnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vfwadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51:   // vfwredosum.vs - treated the same as vfwredosum.vs
    case 49: { // vfwredusum.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_wf<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vfwsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vfwadd.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wfv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vfwsub.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wfv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vfwmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vfwmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vfwnmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vfwmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vfwnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
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
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 1: { // vredand.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<And, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vredor.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Or, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vredxor.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Xor, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vredminu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Min, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vredmin.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Min, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vredmaxu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Max, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vredmax.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Max, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vaaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 9: { // vaadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 10: { // vasubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 11: { // vasub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 16: { // vmv.x.s
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &dest = rddata[t].i;
        vector_op_scalar(dest, warp.vreg_file, rsrc0, rsrc1, warp.vtype.vsew);
        DP(4, "Moved " << +dest << " from: " << +rsrc1 << " to: " << +rdest);
      }
    } break;
    case 18: { // vzext.vf8, vsext.vf8, vzext.vf4, vsext.vf4, vzext.vf2, vsext.vf2
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        bool negativeLmul = warp.vtype.vlmul >> 2;
        uint32_t illegalLmul = negativeLmul && !((8 >> (0x8 - warp.vtype.vlmul)) >> (0x4 - (rsrc0 >> 1)));
        if (illegalLmul) {
          std::cout << "Lmul*vf<1/8 is not supported by vzext and vsext." << std::endl;
          std::abort();
        }
        vector_op_vix_ext<Xunary0>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 20: { // vid.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vid(warp.vreg_file, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: { // vcompress.vm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_compress<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 24: { // vmandn.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<AndNot>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 25: { // vmand.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<And>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 26: { // vmor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Or>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 27: { // vmxor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Xor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 28: { // vmorn.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<OrNot>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 29: { // vmnand.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Nand>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 30: { // vmnor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Nor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 31: { // vmxnor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Xnor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 32: { // vdivu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Div, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vdiv.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Div, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 34: { // vremu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 35: { // vrem.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Rem, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vmulhu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 37: { // vmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 38: { // vmulhsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulhsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmulh.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulh, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Madd, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vnmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Nmsub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Macc, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Nmsac, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vwaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vwsubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51: { // vwsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vwaddu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 53: { // vwadd.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vwsubu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 55: { // vwsub.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vwmulu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 58: { // vwmulsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 59: { // vwmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vwmaccu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vwmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Macc, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vwmaccsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
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
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vrsub.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vslideup.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, 0, vmask, false);
      }
    } break;
    case 15: { // vslidedown.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, false);
      }
    } break;
    case 16: { // vadc.vim
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vi, vmadc.vim
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: { // vmv.v.i
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.i
          if (rsrc0 != 0) {
            std::cout << "For vmv.v.i vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vim
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Sll, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmv1r.v, vmv2r.v, vmv4r.v, vmv8r.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        uint32_t nreg = (immsrc & 0b111) + 1;
        if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
          std::cout << "Reserved value for nreg: " << nreg << std::endl;
          std::abort();
        }
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, nreg * VLEN / warp.vtype.vsew, vmask);
      }
    } break;
    case 40: { // vsrl.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    default:
      std::cout << "Unrecognised vector - immidiate instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 4: {
    switch (func6) {
    case 0: { // vadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Sub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vrsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vminu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Min, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vmin.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Min, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vmaxu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Max, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vmax.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Max, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vslideup.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, false);
      }
    } break;
    case 15: { // vslidedown.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, false);
      }
    } break;
    case 16: { // vadc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vx, vmadc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 18: { // vsbc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 19: { // vmsbc.vx, vmsbc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.x
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.x vs2 must contain v0." << std::endl;
            std::abort();
          }
          auto &src1 = warp.ireg_file.at(t).at(rsrc0);
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vxm
          auto &src1 = warp.ireg_file.at(t).at(rsrc0);
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 34: { // vssubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 35: { // vssub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Sll, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vsmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 40: { // vsrl.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
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
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vfsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vfmin.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vfmax.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vfsgnj.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vfsgnjn.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vfsgnjx.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 14: { // vfslide1up.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, true);
      }
    } break;
    case 15: { // vfslide1down.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, true);
      }
    } break;
    case 16: { // vfmv.s.f
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (rsrc1 != 0) {
          std::cout << "For vfmv.s.f vs2 must contain v0." << std::endl;
          std::abort();
        }
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, std::min(warp.vl, (uint32_t)1), vmask);
      }
    } break;
    case 24: { // vmfeq.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Feq, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vfmv.v.f
          if (rsrc1 != 0) {
            std::cout << "For vfmv.v.f vs2 must contain v0." << std::endl;
            std::abort();
          }
          auto &src1 = warp.freg_file.at(t).at(rsrc0);
          vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vfmerge.vfm
          auto &src1 = warp.freg_file.at(t).at(rsrc0);
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 25: { // vmfle.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fle, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmflt.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Flt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmfne.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fne, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmfgt.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fgt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmfge.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fge, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vfdiv.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fdiv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vfrdiv.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Frdiv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vfmul.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vfrsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Frsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 40: { // vfmadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vfnmadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vfmsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vfnmsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 44: { // vfmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vfnmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 46: { // vfmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vfnmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vfwadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vfwsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vfwadd.wf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        uint64_t src1_d = rv_ftod(src1);
        vector_op_vix_wx<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vfwsub.wf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        uint64_t src1_d = rv_ftod(src1);
        vector_op_vix_wx<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vfwmul.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vfwmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vfwnmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vfwmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vfwnmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised float vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 6: {
    switch (func6) {
    case 8: { // vaaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 9: { // vaadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 10: { // vasubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 11: { // vasub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 14: { // vslide1up.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, true);
      }
    } break;
    case 15: { // vslide1down.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, true);
      }
    } break;
    case 16: { // vmv.s.x
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (rsrc1 != 0) {
          std::cout << "For vmv.s.x vs2 must contain v0." << std::endl;
          std::abort();
        }
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, std::min(warp.vl, (uint32_t)1), vmask);
      }
    } break;
    case 32: { // vdivu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Div, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vdiv.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Div, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 34: { // vremu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 35: { // vrem.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rem, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vmulhu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 37: { // vmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mul, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 38: { // vmulhsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulhsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmulh.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulh, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vmadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Madd, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vnmsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Nmsub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vmacc.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Macc, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vnmsac.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Nmsac, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vwaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Add, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vwsubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51: { // vwsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Sub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vwaddu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_wx<Add, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 53: { // vwadd.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        Word src1_ext = sext(src1, warp.vtype.vsew);
        vector_op_vix_wx<Add, int8_t, int16_t, int32_t, int64_t>(src1_ext, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vwsubu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_wx<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 55: { // vwsub.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        Word &src1 = warp.ireg_file.at(t).at(rsrc0);
        Word src1_ext = sext(src1, warp.vtype.vsew);
        vector_op_vix_wx<Sub, int8_t, int16_t, int32_t, int64_t>(src1_ext, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vwmulu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 58: { // vwmulsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 59: { // vwmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mul, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vwmaccu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vwmacc.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Macc, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vwmaccus.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Maccus, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vwmaccsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
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
    uint32_t vsew = instr.getVsew();
    uint32_t vlmul = instr.getVlmul();

    if (!instr.hasZimm()) { // vsetvl
      uint32_t zimm = rsdata[0][1].u;
      vlmul = zimm & mask_v_lmul;
      vsew = (zimm >> shift_v_sew) & mask_v_sew;
      vta = (zimm >> shift_v_ta) & mask_v_ta;
      vma = (zimm >> shift_v_ma) & mask_v_ma;
    }

    bool negativeLmul = vlmul >> 2;
    uint32_t vlenDividedByLmul = VLEN >> (0x8 - vlmul);
    uint32_t vlenMultipliedByLmul = VLEN << vlmul;
    uint32_t vlenTimesLmul = negativeLmul ? vlenDividedByLmul : vlenMultipliedByLmul;
    uint32_t vsew_bits = 1 << (3 + vsew);
    warp.vlmax = vlenTimesLmul / vsew_bits;
    warp.vtype.vill = (vsew_bits > XLEN) || (warp.vlmax < (VLEN / XLEN));

    Word s0 = instr.getImm(); // vsetivli
    if (!instr.hasImm()) {    // vsetvli/vsetvl
      s0 = rsdata[0][0].u;
    }

    DP(4, "Vset(i)vl(i) - vill: " << +warp.vtype.vill << " vma: " << vma << " vta: " << vta << " lmul: " << vlmul << " sew: " << vsew << " s0: " << s0 << " vlmax: " << warp.vlmax);
    warp.vl = std::min(s0, warp.vlmax);

    if (warp.vtype.vill) {
      this->set_csr(VX_CSR_VTYPE, (Word)1 << (XLEN - 1), 0, wid);
      warp.vtype.vma = 0;
      warp.vtype.vta = 0;
      warp.vtype.vsew = 0;
      warp.vtype.vlmul = 0;
      this->set_csr(VX_CSR_VL, 0, 0, wid);
      rddata[0].i = warp.vl;
    } else {
      warp.vtype.vma = vma;
      warp.vtype.vta = vta;
      warp.vtype.vsew = vsew_bits;
      warp.vtype.vlmul = vlmul;
      Word vtype_ = vlmul;
      vtype_ |= vsew << shift_v_sew;
      vtype_ |= vta << shift_v_ta;
      vtype_ |= vma << shift_v_ma;
      this->set_csr(VX_CSR_VTYPE, vtype_, 0, wid);
      this->set_csr(VX_CSR_VL, warp.vl, 0, wid);
      rddata[0].i = warp.vl;
    }
  }
    this->set_csr(VX_CSR_VSTART, 0, 0, wid);
    break;
  default:
    std::cout << "Unrecognised vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
    std::abort();
  }
}
