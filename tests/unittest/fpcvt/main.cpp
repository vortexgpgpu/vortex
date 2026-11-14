// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// fpcvt.h packs fp32 to tf32, fp16 and bf16 on the device for
// flash_attn_tcu_wg. Its results must match the host conversions bit for bit,
// or the device's P diverges from the test's reference.

#include "fpcvt.h"
#include <rvfloats.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {

template <typename Device, typename Host>
long check_format(const char* name, Device device, Host host) {
  long bad = 0, n = 0;
  auto check = [&](float f) {
    ++n;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t got = device(f), want = host(u);
    if (got != want && ++bad <= 4) {
      std::printf("%s mismatch: f=%.9g (0x%08x) device=0x%05x host=0x%05x\n",
                  name, (double)f, u, got, want);
    }
  };
  std::srand(7);
  for (int i = 0; i < 20000; ++i) {
    check((float)std::rand() / RAND_MAX);
  }
  // every exponent, with mantissas at the rounding boundaries
  for (uint32_t e = 0; e < 256; ++e) {
    for (uint32_t m : {0u, 1u, 0x1000u, 0x0FFFu, 0x400000u, 0x7FFFFFu, 0x2AAAAAu}) {
      for (uint32_t s : {0u, 1u}) {
        uint32_t u = (s << 31) | (e << 23) | m;
        float f;
        std::memcpy(&f, &u, 4);
        check(f);
      }
    }
  }
  for (int e = -140; e <= 130; ++e) {
    check(std::ldexp(1.0f, e));
    check(-std::ldexp(1.0f, e));
  }
  std::printf("%s: %ld cases, %ld mismatched\n", name, n, bad);
  return bad;
}

} // namespace

int main() {
  long bad = 0;
  bad += check_format("tf32", fpcvt::f32_to_tf32, [](uint32_t u) { return rv_ftotf32_s(u, 0, nullptr); });
  bad += check_format("fp16", fpcvt::f32_to_fp16, [](uint32_t u) { return rv_ftoh_s(u, 0, nullptr); });
  bad += check_format("bf16", fpcvt::f32_to_bf16, [](uint32_t u) { return rv_ftob_s(u, 0, nullptr); });
  std::printf(bad ? "FAILED!\n" : "PASSED!\n");
  return bad ? 1 : 0;
}
