// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Host fp8 conversions (sw/common/rvfloats): every E4M3 and E5M2 code must
// survive decode + encode, and the E4M3 encoder must follow OCP E4M3 (no
// infinity, max finite 448, NaN = S.1111.111; out-of-range values saturate).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "rvfloats.h"

namespace {

int failures = 0;

uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

void check_round_trip(const char *fmt, uint32_t (*dec)(uint8_t, uint32_t, uint32_t *),
                      uint8_t (*enc)(uint32_t, uint32_t, uint32_t *)) {
  for (int c = 0; c < 256; ++c) {
    uint32_t f = dec(uint8_t(c), 0, nullptr);
    uint8_t back = enc(f, 0, nullptr);
    if (std::isnan(u2f(f))) {
      if (!std::isnan(u2f(dec(back, 0, nullptr)))) {
        printf("FAILED: %s 0x%02x (NaN) encoded back as 0x%02x\n", fmt, c, back);
        ++failures;
      }
    } else if (back != c) {
      printf("FAILED: %s 0x%02x (%g) encoded back as 0x%02x\n", fmt, c, u2f(f), back);
      ++failures;
    }
  }
}

void expect_e4m3(float x, uint8_t expect) {
  uint8_t c = rv_ftoe4m3_s(f2u(x), 0, nullptr);
  if (c != expect) {
    printf("FAILED: e4m3(%g) = 0x%02x, expected 0x%02x\n", x, c, expect);
    ++failures;
  }
}

void expect_mxfp8(float x, uint8_t expect) {
  uint8_t c = rv_ftomxfp8_s(f2u(x), 127, 0, nullptr);
  if (c != expect) {
    printf("FAILED: mxfp8(%g, scale 2^0) = 0x%02x, expected 0x%02x\n", x, c, expect);
    ++failures;
  }
}

} // namespace

int main() {
  check_round_trip("e4m3", rv_e4m3tof_s, rv_ftoe4m3_s);
  check_round_trip("e5m2", rv_e5m2tof_s, rv_ftoe5m2_s);

  // rounding near the top of the E4M3 range (round to nearest even)
  expect_e4m3(247.0f, 0x77);
  expect_e4m3(248.0f, 0x78);
  expect_e4m3(300.0f, 0x79);
  expect_e4m3(304.0f, 0x7a);
  expect_e4m3(447.0f, 0x7e);
  expect_e4m3(464.0f, 0x7e);
  // out of range: saturate; NaN keeps its sign
  expect_e4m3(480.0f, 0x7e);
  expect_e4m3(1.0e6f, 0x7e);
  expect_e4m3(-1.0e6f, 0xfe);
  expect_e4m3(INFINITY, 0x7e);
  expect_e4m3(-INFINITY, 0xfe);
  expect_e4m3(NAN, 0x7f);
  expect_e4m3(-NAN, 0xff);

  // MX quantiser (E4M3 elements, scale 2^0)
  expect_mxfp8(384.0f, 0x7c);
  expect_mxfp8(448.0f, 0x7e);

  if (failures) {
    printf("%d check(s) FAILED!\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
