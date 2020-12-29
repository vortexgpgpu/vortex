#pragma once

#include <iostream>
#include <math.h>
#include <limits>

union Float_t {    
    float f;
    int   i;
    struct {
        uint32_t man  : 23;
        uint32_t exp  : 8;
        uint32_t sign : 1;
    } parts;
};

inline float fround(float x, int32_t precision = 8) {
  auto power_of_10 = std::pow(10, precision);
  return std::round(x * power_of_10) / power_of_10;
}

inline bool almost_equal_eps(float a, float b, int ulp = 128) {
  auto eps = std::numeric_limits<float>::epsilon() * (std::max(fabs(a), fabs(b)) * ulp);
  auto d = fabs(a - b);
  if (d > eps) {
    std::cout << "*** almost_equal_eps: d=" << d << ", eps=" << eps << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal_ulp(float a, float b, int32_t ulp = 6) {
  Float_t fa{a}, fb{b};
  auto d = std::abs(fa.i - fb.i);
  if (d > ulp) {
    std::cout << "*** almost_equal_ulp: a=" << a << ", b=" << b << ", ulp=" << d << ", ia=" << std::hex << fa.i << ", ib=" << fb.i << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal(float a, float b) {
  if (a == b)
    return true;
  /*if (almost_equal_eps(a, b))
    return true;*/
  return almost_equal_ulp(a, b);
}

class ITestCase {
public:
  ITestCase() {}
  virtual ~ITestCase() {}

  virtual void setup(int n, void* src1, void* src2)  = 0;  
  virtual int verify(int n, void* dst, const void* src1, const void* src2) = 0;
};

class Test_IADD : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] + b[i]; 
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IMUL : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * b[i]; 
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IDIV : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] / b[i]; 
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IDIV_MUL : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] / b[i]; 
      auto y = a[i] * b[i]; 
      auto ref = x + y; 
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FADD : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] + b[i]; 
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FSUB : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] - b[i]; 
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMUL : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * b[i]; 
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMADD : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * b[i] + b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMSUB : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * b[i] - b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMADD : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = -a[i] * b[i] - b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMSUB : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = -a[i] * b[i] + b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMADD_MADD : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = -a[i] * b[i] - b[i];
      auto y =  a[i] * b[i] + b[i];
      auto ref = x + y;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FDIV : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] / b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FDIV2 : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] / b[i];
      auto y = b[i] / a[i];
      auto ref = x + y;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FSQRT : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      float q = 1.0f + (i % 64);
      a[i] = q;
      b[i] = q;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = sqrt(a[i] * b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FTOI : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround((n/2 - i) + (float(i)/n));
      b[i] = fround((n/2 - i) + (float(i)/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (int32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (int32_t)x;
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FTOU : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = fround(i + (float(i)/n));
      b[i] = fround(i + (float(i)/n));
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (uint32_t*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (uint32_t)x;
      if (c[i] != ref) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_ITOF : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 - i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (float)x;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_UTOF : public ITestCase {
public:

  void setup(int n, void* src1, void* src2) override {
    auto a = (uint32_t*)src1;
    auto b = (uint32_t*)src2;
    for (int i = 0; i < n; ++i) {
      a[i] = i;
      b[i] = i;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (uint32_t*)src1;
    auto b = (uint32_t*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (float)x;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << i << ": expected " << ref << ", actual " << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};