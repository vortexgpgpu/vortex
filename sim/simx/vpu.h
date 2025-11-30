#ifdef EXT_V_ENABLE
#pragma once

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
#endif