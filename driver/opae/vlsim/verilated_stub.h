#pragma once

#undef VL_ST_SIG8
#define VL_ST_SIG8(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    CData name

#undef VL_ST_SIG16
#define VL_ST_SIG16(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    SData name

#undef VL_ST_SIG64
#define VL_ST_SIG64(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    QData name

#undef VL_ST_SIG
#define VL_ST_SIG(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    IData name

#undef VL_ST_SIGW
#define VL_ST_SIGW(name, msb, lsb, words) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    WData name[words]

#undef VL_SIG8
#define VL_SIG8(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    CData name

#undef VL_SIG16
#define VL_SIG16(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    SData name

#undef VL_SIG64
#define VL_SIG64(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    QData name

#undef VL_SIG
#define VL_SIG(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    IData name

#undef VL_SIGW
#define VL_SIGW(name, msb, lsb, words) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    WData name[words]

#undef VL_IN8
#define VL_IN8(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    CData name

#undef VL_IN16
#define VL_IN16(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    SData name

#undef VL_IN64
#define VL_IN64(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    QData name

#undef VL_IN
#define VL_IN(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    IData name

#undef VL_INW
#define VL_INW(name, msb, lsb, words) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    WData name[words]

#undef VL_INOUT8
#define VL_INOUT8(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    CData name

#undef VL_INOUT16
#define VL_INOUT16(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    SData name

#undef VL_INOUT64
#define VL_INOUT64(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    QData name

#undef VL_INOUT
#define VL_INOUT(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    IData name

#undef VL_INOUTW
#define VL_INOUTW(name, msb, lsb, words) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    WData name[words]

#undef VL_OUT8
#define VL_OUT8(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    CData name

#undef VL_OUT16
#define VL_OUT16(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    SData name   

#undef VL_OUT64
#define VL_OUT64(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    QData name

#undef VL_OUT
#define VL_OUT(name, msb, lsb) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    IData name

#undef VL_OUTW
#define VL_OUTW(name, msb, lsb, words) \
    enum { VL_MSB_##name = msb, VL_LSB_##name = lsb, VL_BITS_##name = (msb - lsb + 1) }; \
    WData name[words]
