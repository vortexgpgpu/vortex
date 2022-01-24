#pragma once

#ifndef XLEN
#define XLEN 32
#endif

#ifndef FLEN
#define FLEN 32
#endif

#if XLEN == 32
#define uintx_t uint32_t
#define intx_t int32_t
#define intm_t int64_t
#define uintm_t uint64_t
#elif XLEN == 64
#define uintx_t uint64_t
#define intx_t int64_t
#define intm_t __int128_t
#define uintm_t __uint128_t
#else
#error unsupported XLEN
#endif

#if FLEN >= XLEN
#if FLEN == 32
#define uintf_t uint32_t
#elif FLEN == 64
#define uintf_t uint64_t
#else
#error unsupported FLEN
#endif
#else
#error unsupported FLEN
#endif
