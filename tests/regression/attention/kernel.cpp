#include <vx_spawn2.h>
#include "common.h"
#include <cmath>
#include <cstring>
#include <algorithm>

__kernel void kernel_qk(kernel_arg_t* __UNIFORM__ arg) {
	auto Q = reinterpret_cast<TYPE*>(arg->Q_addr);
	auto K = reinterpret_cast<TYPE*>(arg->K_addr);
	auto S = reinterpret_cast<TYPE*>(arg->S_addr);
    auto N = arg->N;
    auto d = arg->d;

    uint32_t col = blockIdx.x;
    uint32_t row = blockIdx.y;

    TYPE sum(0);
    for (uint32_t e = 0; e < d; ++e) {
        sum += Q[row * d + e] * K[e * N + col];
    }
    
    S[row * N + col] = sum;
}

__kernel void kernel_softmax(kernel_arg_t* __UNIFORM__ arg) {
	auto S = reinterpret_cast<TYPE*>(arg->S_addr);
	auto P = reinterpret_cast<TYPE*>(arg->P_addr);
    auto N = arg->N;

    int row = blockIdx.x;

    TYPE max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col) {
        max_val = std::max(max_val, S[row * N + col]);
    }

    TYPE exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
        TYPE e = std::exp(S[row * N + col] - max_val);
        P[row * N + col] = e;
        exp_sum += e;
    }

    for (uint32_t col = 0; col < N; ++col) {
        P[row * N + col] /= exp_sum;
    }
}

__kernel void kernel_pv(kernel_arg_t* __UNIFORM__ arg) {
	auto P = reinterpret_cast<TYPE*>(arg->P_addr);
	auto V = reinterpret_cast<TYPE*>(arg->V_addr);
	auto O = reinterpret_cast<TYPE*>(arg->O_addr);
    auto N = arg->N;
    auto d = arg->d;

    uint32_t col = blockIdx.x;
    uint32_t row = blockIdx.y;

    TYPE sum(0);
    for (uint32_t e = 0; e < N; ++e) {
        sum += P[row * N + e] * V[e * d + col];
    }

    O[row * d + col] = sum;
}
