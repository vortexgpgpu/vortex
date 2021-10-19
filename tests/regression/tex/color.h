//
// Copyright (c) Blaise Tine.  All rights reserved.
//
//
// Use of this sample source code is subject to the terms of the Microsoft
// license agreement under which you licensed this sample source code. If
// you did not accept the terms of the license agreement, you are not
// authorized to use this sample source code. For the terms of the license,
// please see the license agreement between you and Microsoft or, if applicable,
// see the LICENSE.RTF on your install media or the root of your tools
// installation.
// THE SAMPLE SOURCE CODE IS PROVIDED "AS IS", WITH NO WARRANTIES OR
// INDEMNITIES.
//
#pragma once

#include <cstdint>
#include <assert.h>

struct ColorARGB {
  union {
    struct {
      uint32_t value;
    };
    struct {
      uint8_t b, g, r, a;
    };
    struct {
      uint8_t m[4];
    };
  };

  ColorARGB() {}

  ColorARGB(int a, int r, int g, int b) {
    assert((a >= 0) && (a <= 0xff));
    assert((r >= 0) && (r <= 0xff));
    assert((g >= 0) && (g <= 0xff));
    assert((b >= 0) && (b <= 0xff));

    this->b = static_cast<uint8_t>(b);
    this->g = static_cast<uint8_t>(g);
    this->r = static_cast<uint8_t>(r);
    this->a = static_cast<uint8_t>(a);
  }

  ColorARGB(int r, int g, int b) {
    assert((r >= 0) && (r <= 0xff));
    assert((g >= 0) && (g <= 0xff));
    assert((b >= 0) && (b <= 0xff));

    this->b = static_cast<uint8_t>(b);
    this->g = static_cast<uint8_t>(g);
    this->r = static_cast<uint8_t>(r);
  }

  ColorARGB(int value) {
    this->value = value;
  }

  void operator=(const ColorARGB &rhs) {
    this->value = rhs.value;
  }

  operator uint32_t() const {
    return this->value;
  }
};