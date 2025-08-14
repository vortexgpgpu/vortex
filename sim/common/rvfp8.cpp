// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rvfloats.h"
#include <stdio.h>
// #include <cstdint>

/* 
SoftFloat doesn't support fp8.
These implementations do not return exception flags,
but instead just return the exceptions' cannonical representation,
for eg. NaN is returned as 0x7fc00000 for fp32
Rounding mode is always set to RNE
*/

uint32_t rv_e4m3tof_s(uint8_t a) {
    if (a == 0) return 0;
    
    uint32_t sign = (a & 0x80) << 24;
    uint32_t exp = (a >> 3) & 0xF;
    uint32_t mant = a & 0x7;
    
    if (exp == 0xF) {
        if (mant != 0) return 0x7fc00000; // NaN
        return sign | 0x7f800000; // Inf
    }
    
    if (exp == 0) {
        // Subnormal
        int lz = mant ? (__builtin_clz(mant) - 29) : 32;
        exp = 127 - 6 - lz;
        mant = (mant << (lz + 1)) & 0x7;
    } else {
        exp += 127 - 7;
    }
    
    return sign | (exp << 23) | (mant << 20);
}

uint8_t rv_ftoe4m3_s(uint32_t a) {
    if (a == 0 || (a & 0x7fffffff) == 0) return 0;
    
    uint8_t sign = (a >> 24) & 0x80;
    int32_t exp = ((a >> 23) & 0xff) - 127;
    uint32_t mant = (a >> 20) & 0x7;
    
    if (exp >= 8) return sign | 0x7f; // Inf
    if (exp < -9) return sign; // Zero
    
    if (exp >= -6) {
        // Normal
        exp += 7;
        uint32_t round_bit = (a >> 19) & 1;
        uint32_t sticky = a & 0x7ffff;
        if (round_bit && (sticky || (mant & 1))) {
            mant++;
            if (mant > 7) {
                mant = 0;
                exp++;
                if (exp >= 15) return sign | 0x7f; // Overflow to inf
            }
        }
        return sign | (exp << 3) | mant;
    } else {
        // Subnormal
        int shift = -6 - exp;
        mant |= 8; // Add implicit 1
        uint32_t round_bit = (mant >> (shift - 1)) & 1;
        uint32_t sticky = (mant & ((1 << (shift - 1)) - 1)) || (a & 0x7ffff);
        mant >>= shift;
        if (round_bit && (sticky || (mant & 1))) {
            mant++;
            if (mant > 7) return sign | 0x8; // Normalize
        }
        return sign | mant;
    }
}

uint32_t rv_e5m2tof_s(uint8_t a) {
    if (a == 0) return 0;
    
    uint32_t sign = (a & 0x80) << 24;
    uint32_t exp = (a >> 2) & 0x1F;
    uint32_t mant = a & 0x3;
    
    if (exp == 0x1F) {
        if (mant != 0) return 0x7fc00000; // NaN
        return sign | 0x7f800000; // Inf
    }
    
    if (exp == 0) {
        // Subnormal
        if (mant == 0) return sign;
        int lz = mant == 1 ? 1 : 0;
        exp = 127 - 14 - lz;
        mant = (mant << (lz + 1)) & 0x3;
    } else {
        exp += 127 - 15;
    }
    
    return sign | (exp << 23) | (mant << 21);
}

uint8_t rv_ftoe5m2_s(uint32_t a) {
    if (a == 0 || (a & 0x7fffffff) == 0) return 0;
    
    uint8_t sign = (a >> 24) & 0x80;
    int32_t exp = ((a >> 23) & 0xff) - 127;
    uint32_t mant = (a >> 21) & 0x3;
    
    if (exp >= 16) return sign | 0x7c; // Inf
    if (exp < -17) return sign; // Zero
    
    if (exp >= -14) {
        // Normal
        exp += 15;
        uint32_t round_bit = (a >> 20) & 1;
        uint32_t sticky = a & 0xfffff;
        if (round_bit && (sticky || (mant & 1))) {
            mant++;
            if (mant > 3) {
                mant = 0;
                exp++;
                if (exp >= 31) return sign | 0x7c; // Overflow to inf
            }
        }
        return sign | (exp << 2) | mant;
    } else {
        // Subnormal
        int shift = -14 - exp;
        mant |= 4; // Add implicit 1
        if (shift > 2) return sign; // Underflow
        uint32_t round_bit = (mant >> (shift - 1)) & 1;
        uint32_t sticky = (mant & ((1 << (shift - 1)) - 1)) || (a & 0xfffff);
        mant >>= shift;
        if (round_bit && (sticky || (mant & 1))) {
            mant++;
            if (mant > 3) return sign | 0x4; // Normalize
        }
        return sign | mant;
    }
}

/*

void print_float_bits(uint32_t f) {
    printf("0x%08x (", f);
    union { uint32_t i; float f; } u;
    u.i = f;
    printf("%.6f)", u.f);
}

int main() {
    printf("=== FP8 E4M3 Conversions ===\n");
    
    uint8_t e4m3_vals[] = {0x00, 0x01, 0x08, 0x3f, 0x7f, 0x80, 0xff, 0x70, 0x78};
    int e4m3_count = sizeof(e4m3_vals) / sizeof(e4m3_vals[0]);
    
    for (int i = 0; i < e4m3_count; i++) {
        uint8_t e4m3 = e4m3_vals[i];
        uint32_t f32 = rv_e4m3tof_s(e4m3);
        uint8_t back = rv_ftoe4m3_s(f32);
        
        printf("E4M3 0x%02x -> FP32 ", e4m3);
        print_float_bits(f32);
        printf(" -> E4M3 0x%02x\n", back);
    }
    
    printf("\n=== FP8 E5M2 Conversions ===\n");
    
    uint8_t e5m2_vals[] = {0x00, 0x01, 0x04, 0x3c, 0x7c, 0x80, 0xff, 0x78, 0x7f};
    int e5m2_count = sizeof(e5m2_vals) / sizeof(e5m2_vals[0]);
    
    for (int i = 0; i < e5m2_count; i++) {
        uint8_t e5m2 = e5m2_vals[i];
        uint32_t f32 = rv_e5m2tof_s(e5m2);
        uint8_t back = rv_ftoe5m2_s(f32);
        
        printf("E5M2 0x%02x -> FP32 ", e5m2);
        print_float_bits(f32);
        printf(" -> E5M2 0x%02x\n", back);
    }
    
    printf("\n=== FP32 to FP8 Conversions ===\n");
    
    uint32_t f32_vals[] = {0x00000000, 0x3f800000, 0x40000000, 0x7f800000, 0x7fc00000, 0x80000000, 0xff800000};
    int f32_count = sizeof(f32_vals) / sizeof(f32_vals[0]);
    
    for (int i = 0; i < f32_count; i++) {
        uint32_t f32 = f32_vals[i];
        uint8_t e4m3 = rv_ftoe4m3_s(f32);
        uint8_t e5m2 = rv_ftoe5m2_s(f32);
        
        printf("FP32 ");
        print_float_bits(f32);
        printf(" -> E4M3 0x%02x, E5M2 0x%02x\n", e4m3, e5m2);
    }
    
    return 0;
}

*/
