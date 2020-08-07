#pragma once

#include <iostream>
#include <math.h>

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
      a[i] = n/2 + i;
      b[i] = n/2 - i;
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
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = n/2 + i;
      b[i] = n/2 - i;
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
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] + b[i]; 
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] - b[i]; 
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * b[i]; 
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * 0.5f + b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] * 0.5f - b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = -a[i] * 0.5f - b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = -a[i] * 0.5f + b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = -a[i] * 0.5f - b[i];
      auto y =  a[i] * 0.5f + b[i];
      auto ref = x + y;
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n - i) * 0.125f;
      b[i] = (n + i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = a[i] / b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n - i) * 0.125f;
      b[i] = (n + i) * 0.125f;
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
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.125f;
      b[i] = (n - i) * 0.125f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto ref = sqrt(a[i]) + b[i];
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.5f;
      b[i] = (n - i) * 0.5f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (int32_t)x;
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = (n + i) * 0.5f;
      b[i] = (n - i) * 0.5f;
    }
  }
  
  int verify(int n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (int i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (uint32_t)x;
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = n/2 + i;
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
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
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
      a[i] = n/2 + i;
      b[i] = n/2 - i;
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
      if (c[i] != ref) {
        std::cout << "error at value " << i << ": actual 0x" << c[i] << ", expected 0x" << ref << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};