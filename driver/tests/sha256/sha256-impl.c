// From: https://github.gatech.edu/aadams80/cs7290-algos

#include <stdint.h>
#include "sha256.h"

#ifdef SHA_NATIVE
#include <vx_intrinsics.h>
#endif

static const uint32_t K[64];
static const uint32_t Hzero[8];
static void pad_message(uint8_t *, uint32_t);
static void sha256_hash(uint8_t *, uint32_t, uint8_t *);
static uint32_t ch(uint32_t, uint32_t, uint32_t);
static uint32_t maj(uint32_t, uint32_t, uint32_t);
static uint32_t Sigma0(uint32_t);
static uint32_t Sigma1(uint32_t);
static uint32_t sigma0(uint32_t);
static uint32_t sigma1(uint32_t);
#ifndef SHA_NATIVE
static uint32_t rotr(int, uint32_t);
#endif

void sha256(uint8_t *buf, uint32_t n_bytes, uint8_t *digest_out) {
    pad_message(buf, n_bytes);
    uint32_t N = PADDED_SIZE_BYTES(n_bytes) / 64;
    sha256_hash(buf, N, digest_out);
}

static void pad_message(uint8_t *buf, uint32_t n_bytes) {
    // Obligatory first padding byte (with highest-order bit set)
    *(buf + n_bytes) = 0x80;

    uint32_t zero_bytes = PADDING_BYTES(n_bytes);
    for (uint32_t i = 0; i < zero_bytes; i++) {
        *(buf + n_bytes + 1 + i) = 0x00;
    }

    // CRITICAL: this is bits, not bytes!
    // We need to multiply n_bytes by 8 to get the number of bits. So we
    // effectively need to do n_bytes << 3. But also, this is a 64-bit
    // field in memory. So break this field into two 32-bit words. To
    // achieve this << 3, use the three highest-order bits of n_bytes as
    // the three lowest-order bits in the high 32-bit word, and then use
    // the 29 lowest-order bits of n_bytes as the 29 highest-order bits
    // of the lower 32-bit word
    uint32_t n_bits_hi = n_bytes >> 29;
    uint32_t n_bits_lo = n_bytes << 3;
    uint8_t *l = buf + n_bytes + 1 + zero_bytes;
    // Need to store this as big endian
    l[0] = (n_bits_hi >> 24) & 0xff;
    l[1] = (n_bits_hi >> 16) & 0xff;
    l[2] = (n_bits_hi >> 8) & 0xff;
    l[3] = n_bits_hi & 0xff;
    l[4] = (n_bits_lo >> 24) & 0xff;
    l[5] = (n_bits_lo >> 16) & 0xff;
    l[6] = (n_bits_lo >> 8) & 0xff;
    l[7] = n_bits_lo & 0xff;
}

static inline uint32_t ijth_M(uint8_t *M, uint32_t i, int j) {
    uint8_t *msg = M + 64*(i-1) + 4*j;
    return (msg[0] << 24) | (msg[1] << 16) | (msg[2] << 8) | msg[3];
}

static void sha256_hash(uint8_t *M, uint32_t N, uint8_t *digest_out) {
    uint32_t H[8];
    for (int i = 0; i < 8; i++) {
        H[i] = Hzero[i];
    }

    for (uint32_t i = 1; i <= N; i++) {
        uint32_t W[64];
        for (int t = 0; t < 64; t++) {
            if (t <= 15) {
                W[t] = ijth_M(M, i, t);
            } else {
                W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16];
            }
        }

        uint32_t a, b, c, d, e, f, g, h;
        a = H[0];
        b = H[1];
        c = H[2];
        d = H[3];
        e = H[4];
        f = H[5];
        g = H[6];
        h = H[7];

        for (int t = 0; t < 64; t++) {
            uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[t] + W[t];
            uint32_t T2 = Sigma0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        H[0] += a;
        H[1] += b;
        H[2] += c;
        H[3] += d;
        H[4] += e;
        H[5] += f;
        H[6] += g;
        H[7] += h;
    }

    for (int i = 0; i < 8; i++) {
        uint8_t *here = digest_out + 4 * i;
        // Big endian
        here[0] = (H[i] >> 24) & 0xff;
        here[1] = (H[i] >> 16) & 0xff;
        here[2] = (H[i] >> 8) & 0xff;
        here[3] = H[i] & 0xff;
    }
}

#ifndef SHA_NATIVE
static inline uint32_t rotr(int n, uint32_t x) {
    return (x >> n) | (x << (32 - n));
}
#endif

static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t Sigma0(uint32_t x) {
    #ifdef SHA_NATIVE
    return __intrin_sha_Sigma0(x);
    #else
    return rotr(2, x) ^ rotr(13, x) ^ rotr(22, x);
    #endif
}

static inline uint32_t Sigma1(uint32_t x) {
    #ifdef SHA_NATIVE
    return __intrin_sha_Sigma1(x);
    #else
    return rotr(6, x) ^ rotr(11, x) ^ rotr(25, x);
    #endif
}

static inline uint32_t sigma0(uint32_t x) {
    #ifdef SHA_NATIVE
    return __intrin_sha_sigma0(x);
    #else
    return rotr(7, x) ^ rotr(18, x) ^ (x >> 3);
    #endif
}

static inline uint32_t sigma1(uint32_t x) {
    #ifdef SHA_NATIVE
    return __intrin_sha_sigma1(x);
    #else
    return rotr(17, x) ^ rotr(19, x) ^ (x >> 10);
    #endif
}

static const uint32_t K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

static const uint32_t Hzero[8] = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
};
