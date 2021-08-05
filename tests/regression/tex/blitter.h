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

#include "surfacedesc.h"

class BlitTable {
public:
  typedef int (*PfnCopy)(const SurfaceDesc &dstDesc, 
                         uint32_t dstOffsetX,
                         uint32_t dstOffsetY, 
                         uint32_t copyWidth,
                         uint32_t copyHeight, 
                         const SurfaceDesc &srcDesc,
                         uint32_t srcOffsetX, 
                         uint32_t srcOffsetY);

  BlitTable() {
    for (uint32_t s = 0; s < FORMAT_COLOR_SIZE_; ++s) {
      for (uint32_t d = 0; d < FORMAT_COLOR_SIZE_; ++d) {
        copyFuncs_[s][d] = CopyInvalid;
      }
    }

    for (uint32_t s = 0; s < FORMAT_COLOR_SIZE_; ++s) {
      switch (s) {
      case FORMAT_A8:
      case FORMAT_L8:
        copyFuncs_[s][s] = CopyFast<uint8_t>;
        break;

      case FORMAT_A8L8:
        copyFuncs_[FORMAT_A8L8][FORMAT_A8] = Copy<FORMAT_A8L8, FORMAT_A8>;
        copyFuncs_[FORMAT_A8L8][FORMAT_A8L8] = CopyFast<uint16_t>;
        break;

      case FORMAT_R5G6B5:
        copyFuncs_[FORMAT_R5G6B5][FORMAT_L8] = Copy<FORMAT_R5G6B5, FORMAT_L8>;
        copyFuncs_[FORMAT_R5G6B5][FORMAT_R5G6B5] = CopyFast<uint16_t>;
        copyFuncs_[FORMAT_R5G6B5][FORMAT_R8G8B8] =
            Copy<FORMAT_R5G6B5, FORMAT_R8G8B8>;
        copyFuncs_[FORMAT_R5G6B5][FORMAT_B8G8R8] =
            Copy<FORMAT_R5G6B5, FORMAT_B8G8R8>;
        copyFuncs_[FORMAT_R5G6B5][FORMAT_A8B8G8R8] =
            Copy<FORMAT_R5G6B5, FORMAT_A8B8G8R8>;
        copyFuncs_[FORMAT_R5G6B5][FORMAT_A8R8G8B8] =
            Copy<FORMAT_R5G6B5, FORMAT_A8R8G8B8>;
        break;

      case FORMAT_A1R5G5B5:
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_A8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_A8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_L8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_L8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_A8L8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_A8L8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_R8G8B8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_R8G8B8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_A8R8G8B8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_A8R8G8B8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_R5G5B5A1] =
            Copy<FORMAT_A1R5G5B5, FORMAT_R5G5B5A1>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_R4G4B4A4] =
            Copy<FORMAT_A1R5G5B5, FORMAT_R4G4B4A4>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_B8G8R8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_B8G8R8>;
        copyFuncs_[FORMAT_A1R5G5B5][FORMAT_A8B8G8R8] =
            Copy<FORMAT_A1R5G5B5, FORMAT_A8B8G8R8>;
        break;

      case FORMAT_A4R4G4B4:
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_A8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_A8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_L8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_L8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_A8L8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_A8L8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_R8G8B8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_R8G8B8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_A8R8G8B8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_A8R8G8B8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_R5G5B5A1] =
            Copy<FORMAT_A4R4G4B4, FORMAT_R5G5B5A1>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_R4G4B4A4] =
            Copy<FORMAT_A4R4G4B4, FORMAT_R4G4B4A4>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_B8G8R8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_B8G8R8>;
        copyFuncs_[FORMAT_A4R4G4B4][FORMAT_A8B8G8R8] =
            Copy<FORMAT_A4R4G4B4, FORMAT_A8B8G8R8>;
        break;

      case FORMAT_R8G8B8:
        copyFuncs_[FORMAT_R8G8B8][FORMAT_L8] = Copy<FORMAT_R8G8B8, FORMAT_L8>;
        copyFuncs_[FORMAT_R8G8B8][FORMAT_R5G6B5] =
            Copy<FORMAT_R8G8B8, FORMAT_R5G6B5>;
        copyFuncs_[FORMAT_R8G8B8][FORMAT_R8G8B8] = CopyFast<uint24_t>;
        copyFuncs_[FORMAT_R8G8B8][FORMAT_B8G8R8] =
            Copy<FORMAT_R8G8B8, FORMAT_B8G8R8>;
        copyFuncs_[FORMAT_R8G8B8][FORMAT_A8B8G8R8] =
            Copy<FORMAT_R8G8B8, FORMAT_A8B8G8R8>;
        copyFuncs_[FORMAT_R8G8B8][FORMAT_A8R8G8B8] =
            Copy<FORMAT_R8G8B8, FORMAT_A8R8G8B8>;
        break;

      case FORMAT_A8R8G8B8:
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_A8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_A8>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_L8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_L8>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_A8L8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_A8L8>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_R5G6B5] =
            Copy<FORMAT_A8R8G8B8, FORMAT_R5G6B5>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_R8G8B8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_R8G8B8>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_A8R8G8B8] = CopyFast<uint32_t>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_R5G5B5A1] =
            Copy<FORMAT_A8R8G8B8, FORMAT_R5G5B5A1>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_R4G4B4A4] =
            Copy<FORMAT_A8R8G8B8, FORMAT_R4G4B4A4>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_B8G8R8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_B8G8R8>;
        copyFuncs_[FORMAT_A8R8G8B8][FORMAT_A8B8G8R8] =
            Copy<FORMAT_A8R8G8B8, FORMAT_A8B8G8R8>;
        break;

      case FORMAT_R5G5B5A1:
        copyFuncs_[FORMAT_R5G5B5A1][FORMAT_A8] =
            Copy<FORMAT_R5G5B5A1, FORMAT_A8>;
        copyFuncs_[FORMAT_R5G5B5A1][FORMAT_L8] =
            Copy<FORMAT_R5G5B5A1, FORMAT_L8>;
        copyFuncs_[FORMAT_R5G5B5A1][FORMAT_A8L8] =
            Copy<FORMAT_R5G5B5A1, FORMAT_A8L8>;
        copyFuncs_[FORMAT_R5G5B5A1][FORMAT_RGB] =
            Copy<FORMAT_R5G5B5A1, FORMAT_RGB>;
        copyFuncs_[FORMAT_R5G5B5A1][FORMAT_ARGB] =
            Copy<FORMAT_R5G5B5A1, FORMAT_ARGB>;
        break;

      case FORMAT_R4G4B4A4:
        copyFuncs_[FORMAT_R4G4B4A4][FORMAT_A8] =
            Copy<FORMAT_R4G4B4A4, FORMAT_A8>;
        copyFuncs_[FORMAT_R4G4B4A4][FORMAT_L8] =
            Copy<FORMAT_R4G4B4A4, FORMAT_L8>;
        copyFuncs_[FORMAT_R4G4B4A4][FORMAT_A8L8] =
            Copy<FORMAT_R4G4B4A4, FORMAT_A8L8>;
        copyFuncs_[FORMAT_R4G4B4A4][FORMAT_RGB] =
            Copy<FORMAT_R4G4B4A4, FORMAT_RGB>;
        copyFuncs_[FORMAT_R4G4B4A4][FORMAT_ARGB] =
            Copy<FORMAT_R4G4B4A4, FORMAT_ARGB>;
        break;

      case FORMAT_B8G8R8:
        copyFuncs_[FORMAT_B8G8R8][FORMAT_L8] = Copy<FORMAT_B8G8R8, FORMAT_L8>;
        copyFuncs_[FORMAT_B8G8R8][FORMAT_RGB] = Copy<FORMAT_B8G8R8, FORMAT_RGB>;
        break;

      case FORMAT_A8B8G8R8:
        copyFuncs_[FORMAT_A8B8G8R8][FORMAT_A8] =
            Copy<FORMAT_A8B8G8R8, FORMAT_A8>;
        copyFuncs_[FORMAT_A8B8G8R8][FORMAT_L8] =
            Copy<FORMAT_A8B8G8R8, FORMAT_L8>;
        copyFuncs_[FORMAT_A8B8G8R8][FORMAT_A8L8] =
            Copy<FORMAT_A8B8G8R8, FORMAT_A8L8>;
        copyFuncs_[FORMAT_A8B8G8R8][FORMAT_RGB] =
            Copy<FORMAT_A8B8G8R8, FORMAT_RGB>;
        copyFuncs_[FORMAT_A8B8G8R8][FORMAT_ARGB] =
            Copy<FORMAT_A8B8G8R8, FORMAT_ARGB>;
        break;
      }
    }
  }

  PfnCopy get(uint32_t srcFormat, uint32_t dstFormat) const {
    assert(srcFormat < FORMAT_COLOR_SIZE_);
    assert(dstFormat < FORMAT_COLOR_SIZE_);
    return copyFuncs_[srcFormat][dstFormat];
  }

private:
  template <ePixelFormat SrcFormat, ePixelFormat DstFormat>
  static int Copy(const SurfaceDesc &dstDesc, 
                  uint32_t dstOffsetX,
                  uint32_t dstOffsetY, 
                  uint32_t copyWidth,
                  uint32_t copyHeight, 
                  const SurfaceDesc &srcDesc,
                  uint32_t srcOffsetX, 
                  uint32_t srcOffsetY) {
    auto srcBPP = TFormatInfo<SrcFormat>::CBSIZE;
    auto dstBPP = TFormatInfo<DstFormat>::CBSIZE;
    auto srcNextLine = srcDesc.Pitch;
    auto dstNextLine = dstDesc.Pitch;

    auto pbSrc = srcDesc.pBits + srcOffsetX * srcBPP + srcOffsetY * srcDesc.Pitch;
    auto pbDst = dstDesc.pBits + dstOffsetX * dstBPP + dstOffsetY * dstDesc.Pitch;

    while (copyHeight--) {
      auto pSrc = reinterpret_cast<const typename TFormatInfo<SrcFormat>::TYPE *>(pbSrc);
      for (auto *pDst = reinterpret_cast<typename TFormatInfo<DstFormat>::TYPE *>(
                    pbDst),
                *const pEnd = pDst + copyWidth;
           pDst != pEnd; ++pDst, ++pSrc) {
        auto tmp = Format::ConvertFrom<SrcFormat, true>(pSrc);
        Format::ConvertTo<DstFormat>(pDst, tmp);
      }

      pbSrc += srcNextLine;
      pbDst += dstNextLine;
    }    
    return 0;
  }

  template <typename Type>
  static int CopyFast(const SurfaceDesc &dstDesc, 
                      uint32_t dstOffsetX,
                      uint32_t dstOffsetY, 
                      uint32_t copyWidth,
                      uint32_t copyHeight, 
                      const SurfaceDesc &srcDesc,
                      uint32_t srcOffsetX, 
                      uint32_t srcOffsetY) {
    auto nBPP = sizeof(Type);
    auto srcNextLine = srcDesc.Pitch;
    auto dstNextLine = dstDesc.Pitch;

    auto pbSrc = srcDesc.pBits + srcOffsetX * nBPP + srcOffsetY * srcDesc.Pitch;
    auto pbDst = dstDesc.pBits + dstOffsetX * nBPP + dstOffsetY * dstDesc.Pitch;

    while (copyHeight--) {
      auto pSrc = reinterpret_cast<const Type *>(pbSrc);
      for (auto *pDst = reinterpret_cast<Type *>(pbDst), *const pEnd = pDst + copyWidth;
           pDst != pEnd; ++pDst, ++pSrc) {
        *pDst = *pSrc;
      }
      pbSrc += srcNextLine;
      pbDst += dstNextLine;
    }
    return 0;
  }

  static int CopyInvalid(const SurfaceDesc & /*dstDesc*/,
                         uint32_t /*dstOffsetX*/, 
                         uint32_t /*dstOffsetY*/,
                         uint32_t /*copyWidth*/, 
                         uint32_t /*copyHeight*/,
                         const SurfaceDesc & /*srcDesc*/,
                         uint32_t /*srcOffsetX*/, 
                         uint32_t /*srcOffsetY*/)
  {
    std::cout << "Error: invalid format" << std::endl;
    return -1;
  }

  PfnCopy copyFuncs_[FORMAT_COLOR_SIZE_][FORMAT_COLOR_SIZE_];
};