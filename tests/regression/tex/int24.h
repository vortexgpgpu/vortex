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

struct uint24_t {
  uint8_t m[3];

  explicit uint24_t(uint32_t value) {
    m[0] = (value >> 0) & 0xff;
    m[1] = (value >> 8) & 0xff;
    m[2] = (value >> 16) & 0xff;
  }

  explicit uint24_t(uint8_t x, uint8_t y, uint8_t z) {
    m[0] = x;
    m[1] = y;
    m[2] = z;
  }

  operator uint32_t() const {
    return (m[2] << 16) | (m[1] << 8) | m[0];
  }
};
