#include <stdint.h>
#include <string.h>
#include <vx_intrinsics.h>
#include "aes256.h"

static void aes256_key_exp(const uint32_t *, uint32_t *, int);
static void aes256_cipher(const uint8_t *, const uint8_t *, const uint8_t *,
                          uint8_t *, const uint32_t *);
static void aes256_inv_cipher(const uint8_t *, const uint8_t *, uint8_t *,
                              const uint32_t *);
static uint32_t sub_word(uint32_t);
static uint32_t rot_word(uint32_t);
static void increment_128bit(uint32_t *, uint32_t);
static void add_round_key(uint8_t *, const uint32_t *);
#ifndef AES_NATIVE
static void inv_sub_bytes(uint8_t *);
static void inv_shift_rows(uint8_t *);
static void inv_mix_columns(uint8_t *);
static void sub_bytes(uint8_t *);
static void shift_rows(uint8_t *);
static void mix_columns(uint8_t *);
static uint8_t s_box_replace(uint8_t);
static uint8_t inv_s_box_replace(uint8_t);
#endif

void aes256_ecb_enc(const uint8_t *in, const uint8_t *key, uint8_t *out, int nblocks) {
    uint32_t round_keys[Nb * (Nr + 1)];

    aes256_key_exp((const uint32_t *)key, round_keys, 0);

    for (int b = 0; b < nblocks; b++) {
        aes256_cipher(NULL, NULL, in + (Nb * 4 * b), out + (Nb * 4 * b), round_keys);
    }
}

void aes256_ecb_dec(const uint8_t *in, const uint8_t *key, uint8_t *out, int nblocks) {
    uint32_t round_keys[Nb * (Nr + 1)];

    aes256_key_exp((const uint32_t *)key, round_keys, 1);

    for (int b = 0; b < nblocks; b++) {
        aes256_inv_cipher(NULL, in + (Nb * 4 * b), out + (Nb * 4 * b), round_keys);
    }
}

void aes256_cbc_enc(const uint8_t *iv, const uint8_t *in, const uint8_t *key,
                    uint8_t *out, int nblocks) {
    uint32_t round_keys[Nb * (Nr + 1)];

    aes256_key_exp((const uint32_t *)key, round_keys, 0);

    const uint8_t *next_iv = iv;
    for (int b = 0; b < nblocks; b++) {
        aes256_cipher(next_iv, NULL, in + (Nb * 4 * b), out + (Nb * 4 * b), round_keys);
        next_iv = out + (Nb * 4 * b);
    }
}

void aes256_cbc_dec(const uint8_t *iv, const uint8_t *in, const uint8_t *key,
                    uint8_t *out, int nblocks) {
    uint32_t round_keys[Nb * (Nr + 1)];

    aes256_key_exp((const uint32_t *)key, round_keys, 1);

    const uint8_t *next_iv = iv;
    for (int b = 0; b < nblocks; b++) {
        aes256_inv_cipher(next_iv, in + (Nb * 4 * b), out + (Nb * 4 * b), round_keys);
        next_iv = in + (Nb * 4 * b);
    }
}

void aes256_ctr(const uint8_t *init_ctr, uint32_t start_block_idx,
                const uint8_t *in, const uint8_t *key, uint8_t *out,
                int nblocks) {
    uint32_t round_keys[Nb * (Nr + 1)];

    aes256_key_exp((const uint32_t *)key, round_keys, 0);

    uint8_t ctr[4 * Nb];
    memcpy(ctr, init_ctr, sizeof ctr);
    increment_128bit((uint32_t *)ctr, start_block_idx);
    for (int b = 0; b < nblocks; b++) {
        aes256_cipher(NULL, in + (Nb * 4 * b), ctr, out + (Nb * 4 * b), round_keys);
        increment_128bit((uint32_t *)ctr, 1);
    }
}

static inline uint32_t big_endian_add(uint32_t a, uint32_t b, int *overflow) {
    uint8_t *bytes = (uint8_t *)(&a);
    uint32_t native = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8)
                      | bytes[3];
    uint32_t sum = native + b;
    *overflow = sum < native;
    uint8_t big_sum[4] = {(sum >> 24) & 0xff, (sum >> 16) & 0xff,
                          (sum >> 8) & 0xff, sum & 0xff};
    return *(uint32_t *)big_sum;
}

// The CTR cipher mode puts us in a tough situation where we need to
// add n to a 128-bit counter in big endian on a big or little endian
// system. For convenience, use a 32-bit addend n. We will only overflow
// that when our input hits 64GiB, which is far beyond what we plan to
// use this implementation for.
static void increment_128bit(uint32_t *limbs, uint32_t n) {
    int overflow;
    limbs[3] = big_endian_add(limbs[3], n, &overflow);
    __if (overflow) {
        limbs[2] = big_endian_add(limbs[2], 1, &overflow);
    } __else {
        return;
    } __endif
    __if (overflow) {
        limbs[1] = big_endian_add(limbs[1], 1, &overflow);
    } __else {
        return;
    } __endif
    __if (overflow) {
        limbs[0] = big_endian_add(limbs[0], 1, &overflow);
    } __endif
}

// Modified key schedule generation from Section 5.3.5 of the AES spec
static void aes256_key_exp(const uint32_t *key, uint32_t *round_keys, int inv_mix_cols) {
    // "Rcon[i] contains the values given by [x^{i-1},{00},{00},{00}]"
    // attempt to construct this in an endianness-safe way. note that
    // Rcon[0] is never accessed in the algorithm below
    static const uint8_t rcon_bytes[] = {
        [4] = 0x01, [8] = 0x02,
        [12] = 0x04, [16] = 0x08,
        [20] = 0x10, [24] = 0x20,
        [28] = 0x40,
    };
    const uint32_t *rcon = (uint32_t *)rcon_bytes;
    uint32_t temp;
    int i;

    for (i = 0; i < Nk; i++) {
        round_keys[i] = key[i];
    }

    for (; i < Nb * (Nr + 1); i++) {
        temp = round_keys[i - 1];
        if (!(i % Nk)) {
            temp = sub_word(rot_word(temp)) ^ rcon[i / Nk];
        } else if (i % Nk == 4) {
            temp = sub_word(temp);
        }
        round_keys[i] = round_keys[i - Nk] ^ temp;
    }

    // For equivalent inverse cipher. See Section 5.3.5 of AES spec
    if (inv_mix_cols) {
        for (int r = 1; r < Nr; r++) {
            #ifdef AES_NATIVE
            __intrin_aes_inv_mixcols(round_keys + (Nb * r), round_keys + (Nb * r));
            #else
            inv_mix_columns((uint8_t *)(round_keys + (Nb * r)));
            #endif
        }
    }
}

static void aes256_cipher(const uint8_t *xor_before, const uint8_t *xor_after,
                          const uint8_t *in, uint8_t *out,
                          const uint32_t *round_keys) {
    uint8_t state[4 * Nb];

    memcpy(state, in, 4 * Nb);

    // For CBC
    if (xor_before) {
        // Minor hack: use add_round_key() since it is functionally
        // equivalent to what we want to do: xor each column with our IV
        add_round_key(state, (uint32_t *)xor_before);
    }

    add_round_key(state, round_keys);

    for (int round = 1; round <= Nr; round++) {
        const uint32_t *this_round_keys = round_keys + (Nb * round);
        #ifdef AES_NATIVE
        if (round < Nr) {
            __intrin_aes_enc_round((uint32_t *)state, (uint32_t *)state, this_round_keys);
        } else {
            __intrin_aes_last_enc_round((uint32_t *)state, (uint32_t *)state, this_round_keys);
        }
        #else
        sub_bytes(state);
        shift_rows(state);
        if (round < Nr) {
            mix_columns(state);
        }
        add_round_key(state, this_round_keys);
        #endif
    }

    // For CTR
    if (xor_after) {
        add_round_key(state, (uint32_t *)xor_after);
    }

    memcpy(out, state, 4 * Nb);
}

// Equivalent inverse cipher from Section 5.3.5 of AES spec
static void aes256_inv_cipher(const uint8_t *xor_after, const uint8_t *in,
                              uint8_t *out, const uint32_t *round_keys) {
    uint8_t state[4 * Nb];

    memcpy(state, in, 4 * Nb);

    add_round_key(state, round_keys + (Nb * Nr));

    for (int round = Nr - 1; round >= 0; round--) {
        const uint32_t *this_round_keys = round_keys + (Nb * round);
        #ifdef AES_NATIVE
        if (round > 0) {
            __intrin_aes_dec_round((uint32_t *)state, (uint32_t *)state, this_round_keys);
        } else {
            __intrin_aes_last_dec_round((uint32_t *)state, (uint32_t *)state, this_round_keys);
        }
        #else
        inv_sub_bytes(state);
        inv_shift_rows(state);
        if (round > 0) {
            inv_mix_columns(state);
        }
        add_round_key(state, this_round_keys);
        #endif
    }

    // For CBC
    if (xor_after) {
        // Minor hack: use add_round_key() since it is functionally
        // equivalent to what we want to do: xor each column with our IV
        add_round_key(state, (uint32_t *)xor_after);
    }

    memcpy(out, state, 4 * Nb);
}

static uint32_t sub_word(uint32_t word) {
#ifdef AES_NATIVE
    return __intrin_aes_subword(word);
#else
    uint8_t *bytes = (uint8_t *)&word;

    for (int i = 0; i < 4; i++) {
        bytes[i] = s_box_replace(bytes[i]);
    }

    return word;
#endif
}

static inline uint32_t rot_word(uint32_t word) {
#if defined(AES_NATIVE) || defined(AES_HYBRID)
    // For endianness reasons, use rotr even though the algorithm
    // specifies an rotl
    return __intrin_rotr_imm(word, 8);
#else
    uint32_t new;
    uint8_t *bytes = (uint8_t *)&word;
    uint8_t *new_bytes = (uint8_t *)&new;

    new_bytes[0] = bytes[1];
    new_bytes[1] = bytes[2];
    new_bytes[2] = bytes[3];
    new_bytes[3] = bytes[0];

    return new;
#endif
}

static void add_round_key(uint8_t *state, const uint32_t *round_keys) {
    uint32_t *state_cols = (uint32_t *)state;

    for (int i = 0; i < Nb; i++) {
        state_cols[i] ^= round_keys[i];
    }
}

#ifndef AES_NATIVE
static void sub_bytes(uint8_t *state) {
    for (int i = 0; i < 4 * Nb; i++) {
        state[i] = s_box_replace(state[i]);
    }
}

static void inv_sub_bytes(uint8_t *state) {
    for (int i = 0; i < 4 * Nb; i++) {
        state[i] = inv_s_box_replace(state[i]);
    }
}

static void shift_rows(uint8_t *state) {
    uint8_t new[4 * Nb];
    new[0] = state[0]; new[4] = state[4]; new[8] = state[8]; new[12] = state[12];
    new[1] = state[5]; new[5] = state[9]; new[9] = state[13]; new[13] = state[1];
    new[2] = state[10]; new[6] = state[14]; new[10] = state[2]; new[14] = state[6];
    new[3] = state[15]; new[7] = state[3]; new[11] = state[7]; new[15] = state[11];
    memcpy(state, new, sizeof new);
}

static void inv_shift_rows(uint8_t *state) {
    uint8_t new[4 * Nb];
    new[0] = state[0]; new[4] = state[4]; new[8] = state[8]; new[12] = state[12];
    new[1] = state[13]; new[5] = state[1]; new[9] = state[5]; new[13] = state[9];
    new[2] = state[10]; new[6] = state[14]; new[10] = state[2]; new[14] = state[6];
    new[3] = state[7]; new[7] = state[11]; new[11] = state[15]; new[15] = state[3];
    memcpy(state, new, sizeof new);
}

static inline uint8_t xtime(uint8_t byte) {
    // Hack because the following line does not work when using multiple
    // threads, strangely enough:
    //     return ((byte << 1) & 0xff) ^ ((0x80 & byte)? 0x1b : 0);
    static const uint8_t xor_with[] = {0x00, 0x1b};
    return ((byte << 1) & 0xff) ^ xor_with[byte >> 7];
}

static void mix_columns(uint8_t *state) {
    uint32_t *state_cols = (uint32_t *)state;

    for (int i = 0; i < Nb; i++) {
        uint8_t *col = (uint8_t *)(state_cols + i);
        uint32_t new = 0;
        uint8_t *new_col = (uint8_t *)&new;

        // important observation: {03}.b = ({01} ^ {02}).b = b ^ {02}.b
        new_col[0] = xtime(col[0]) ^ col[1] ^ xtime(col[1]) ^ col[2] ^ col[3];
        new_col[1] = col[0] ^ xtime(col[1]) ^ col[2] ^ xtime(col[2]) ^ col[3];
        new_col[2] = col[0] ^ col[1] ^ xtime(col[2]) ^ col[3] ^ xtime(col[3]);
        new_col[3] = col[0] ^ xtime(col[0]) ^ col[1] ^ col[2] ^ xtime(col[3]);

        state_cols[i] = new;
    }
}

static void inv_mix_columns(uint8_t *state) {
    uint32_t *state_cols = (uint32_t *)state;

    for (int i = 0; i < Nb; i++) {
        uint8_t *col = (uint8_t *)(state_cols + i);
        uint32_t new = 0;
        uint8_t *new_col = (uint8_t *)&new;

        // important observations:
        // {0e}.b = ({02} ^ {04} ^ {08}).b = {02}.b ^ {04}.b ^ {08}.b
        // {0b}.b = ({01} ^ {02} ^ {08}).b = b ^ {02}.b ^ {08}.b
        // {0d}.b = ({01} ^ {04} ^ {08}).b = b ^ {04}.b ^ {08}.b
        // {09}.b = ({01} ^ {08}).b = b ^ {08}.b

        new_col[0] = xtime(col[0]) ^ xtime(xtime(col[0])) ^ xtime(xtime(xtime(col[0]))) // {0e}.col[0]
                     ^ col[1] ^ xtime(col[1]) ^ xtime(xtime(xtime(col[1]))) // {0b}.col[1]
                     ^ col[2] ^ xtime(xtime(col[2])) ^ xtime(xtime(xtime(col[2]))) // {0d}.col[2]
                     ^ col[3] ^ xtime(xtime(xtime(col[3]))); // {09}.col[3]

        new_col[1] = col[0] ^ xtime(xtime(xtime(col[0]))) // {09}.col[0]
                     ^ xtime(col[1]) ^ xtime(xtime(col[1])) ^ xtime(xtime(xtime(col[1]))) // {0e}.col[1]
                     ^ col[2] ^ xtime(col[2]) ^ xtime(xtime(xtime(col[2]))) // {0b}.col[2]
                     ^ col[3] ^ xtime(xtime(col[3])) ^ xtime(xtime(xtime(col[3]))); // {0d}.col[3]

        new_col[2] = col[0] ^ xtime(xtime(col[0])) ^ xtime(xtime(xtime(col[0]))) // {0d}.col[0]
                     ^ col[1] ^ xtime(xtime(xtime(col[1]))) // {09}.col[1]
                     ^ xtime(col[2]) ^ xtime(xtime(col[2])) ^ xtime(xtime(xtime(col[2]))) // {0e}.col[2]
                     ^ col[3] ^ xtime(col[3]) ^ xtime(xtime(xtime(col[3]))); // {0b}.col[3]

        new_col[3] = col[0] ^ xtime(col[0]) ^ xtime(xtime(xtime(col[0]))) // {0b}.col[0]
                     ^ col[1] ^ xtime(xtime(col[1])) ^ xtime(xtime(xtime(col[1]))) // {0d}.col[1]
                     ^ col[2] ^ xtime(xtime(xtime(col[2]))) // {09}.col[2]
                     ^ xtime(col[3]) ^ xtime(xtime(col[3])) ^ xtime(xtime(xtime(col[3]))); // {0e}.col[3]

        state_cols[i] = new;
    }
}

static inline uint8_t s_box_replace(uint8_t byte) {
    static const uint8_t s_box[256] = {
        [0x00] = 0x63, [0x01] = 0x7c, [0x02] = 0x77, [0x03] = 0x7b,
        [0x04] = 0xf2, [0x05] = 0x6b, [0x06] = 0x6f, [0x07] = 0xc5,
        [0x08] = 0x30, [0x09] = 0x01, [0x0a] = 0x67, [0x0b] = 0x2b,
        [0x0c] = 0xfe, [0x0d] = 0xd7, [0x0e] = 0xab, [0x0f] = 0x76,
        [0x10] = 0xca, [0x11] = 0x82, [0x12] = 0xc9, [0x13] = 0x7d,
        [0x14] = 0xfa, [0x15] = 0x59, [0x16] = 0x47, [0x17] = 0xf0,
        [0x18] = 0xad, [0x19] = 0xd4, [0x1a] = 0xa2, [0x1b] = 0xaf,
        [0x1c] = 0x9c, [0x1d] = 0xa4, [0x1e] = 0x72, [0x1f] = 0xc0,
        [0x20] = 0xb7, [0x21] = 0xfd, [0x22] = 0x93, [0x23] = 0x26,
        [0x24] = 0x36, [0x25] = 0x3f, [0x26] = 0xf7, [0x27] = 0xcc,
        [0x28] = 0x34, [0x29] = 0xa5, [0x2a] = 0xe5, [0x2b] = 0xf1,
        [0x2c] = 0x71, [0x2d] = 0xd8, [0x2e] = 0x31, [0x2f] = 0x15,
        [0x30] = 0x04, [0x31] = 0xc7, [0x32] = 0x23, [0x33] = 0xc3,
        [0x34] = 0x18, [0x35] = 0x96, [0x36] = 0x05, [0x37] = 0x9a,
        [0x38] = 0x07, [0x39] = 0x12, [0x3a] = 0x80, [0x3b] = 0xe2,
        [0x3c] = 0xeb, [0x3d] = 0x27, [0x3e] = 0xb2, [0x3f] = 0x75,
        [0x40] = 0x09, [0x41] = 0x83, [0x42] = 0x2c, [0x43] = 0x1a,
        [0x44] = 0x1b, [0x45] = 0x6e, [0x46] = 0x5a, [0x47] = 0xa0,
        [0x48] = 0x52, [0x49] = 0x3b, [0x4a] = 0xd6, [0x4b] = 0xb3,
        [0x4c] = 0x29, [0x4d] = 0xe3, [0x4e] = 0x2f, [0x4f] = 0x84,
        [0x50] = 0x53, [0x51] = 0xd1, [0x52] = 0x00, [0x53] = 0xed,
        [0x54] = 0x20, [0x55] = 0xfc, [0x56] = 0xb1, [0x57] = 0x5b,
        [0x58] = 0x6a, [0x59] = 0xcb, [0x5a] = 0xbe, [0x5b] = 0x39,
        [0x5c] = 0x4a, [0x5d] = 0x4c, [0x5e] = 0x58, [0x5f] = 0xcf,
        [0x60] = 0xd0, [0x61] = 0xef, [0x62] = 0xaa, [0x63] = 0xfb,
        [0x64] = 0x43, [0x65] = 0x4d, [0x66] = 0x33, [0x67] = 0x85,
        [0x68] = 0x45, [0x69] = 0xf9, [0x6a] = 0x02, [0x6b] = 0x7f,
        [0x6c] = 0x50, [0x6d] = 0x3c, [0x6e] = 0x9f, [0x6f] = 0xa8,
        [0x70] = 0x51, [0x71] = 0xa3, [0x72] = 0x40, [0x73] = 0x8f,
        [0x74] = 0x92, [0x75] = 0x9d, [0x76] = 0x38, [0x77] = 0xf5,
        [0x78] = 0xbc, [0x79] = 0xb6, [0x7a] = 0xda, [0x7b] = 0x21,
        [0x7c] = 0x10, [0x7d] = 0xff, [0x7e] = 0xf3, [0x7f] = 0xd2,
        [0x80] = 0xcd, [0x81] = 0x0c, [0x82] = 0x13, [0x83] = 0xec,
        [0x84] = 0x5f, [0x85] = 0x97, [0x86] = 0x44, [0x87] = 0x17,
        [0x88] = 0xc4, [0x89] = 0xa7, [0x8a] = 0x7e, [0x8b] = 0x3d,
        [0x8c] = 0x64, [0x8d] = 0x5d, [0x8e] = 0x19, [0x8f] = 0x73,
        [0x90] = 0x60, [0x91] = 0x81, [0x92] = 0x4f, [0x93] = 0xdc,
        [0x94] = 0x22, [0x95] = 0x2a, [0x96] = 0x90, [0x97] = 0x88,
        [0x98] = 0x46, [0x99] = 0xee, [0x9a] = 0xb8, [0x9b] = 0x14,
        [0x9c] = 0xde, [0x9d] = 0x5e, [0x9e] = 0x0b, [0x9f] = 0xdb,
        [0xa0] = 0xe0, [0xa1] = 0x32, [0xa2] = 0x3a, [0xa3] = 0x0a,
        [0xa4] = 0x49, [0xa5] = 0x06, [0xa6] = 0x24, [0xa7] = 0x5c,
        [0xa8] = 0xc2, [0xa9] = 0xd3, [0xaa] = 0xac, [0xab] = 0x62,
        [0xac] = 0x91, [0xad] = 0x95, [0xae] = 0xe4, [0xaf] = 0x79,
        [0xb0] = 0xe7, [0xb1] = 0xc8, [0xb2] = 0x37, [0xb3] = 0x6d,
        [0xb4] = 0x8d, [0xb5] = 0xd5, [0xb6] = 0x4e, [0xb7] = 0xa9,
        [0xb8] = 0x6c, [0xb9] = 0x56, [0xba] = 0xf4, [0xbb] = 0xea,
        [0xbc] = 0x65, [0xbd] = 0x7a, [0xbe] = 0xae, [0xbf] = 0x08,
        [0xc0] = 0xba, [0xc1] = 0x78, [0xc2] = 0x25, [0xc3] = 0x2e,
        [0xc4] = 0x1c, [0xc5] = 0xa6, [0xc6] = 0xb4, [0xc7] = 0xc6,
        [0xc8] = 0xe8, [0xc9] = 0xdd, [0xca] = 0x74, [0xcb] = 0x1f,
        [0xcc] = 0x4b, [0xcd] = 0xbd, [0xce] = 0x8b, [0xcf] = 0x8a,
        [0xd0] = 0x70, [0xd1] = 0x3e, [0xd2] = 0xb5, [0xd3] = 0x66,
        [0xd4] = 0x48, [0xd5] = 0x03, [0xd6] = 0xf6, [0xd7] = 0x0e,
        [0xd8] = 0x61, [0xd9] = 0x35, [0xda] = 0x57, [0xdb] = 0xb9,
        [0xdc] = 0x86, [0xdd] = 0xc1, [0xde] = 0x1d, [0xdf] = 0x9e,
        [0xe0] = 0xe1, [0xe1] = 0xf8, [0xe2] = 0x98, [0xe3] = 0x11,
        [0xe4] = 0x69, [0xe5] = 0xd9, [0xe6] = 0x8e, [0xe7] = 0x94,
        [0xe8] = 0x9b, [0xe9] = 0x1e, [0xea] = 0x87, [0xeb] = 0xe9,
        [0xec] = 0xce, [0xed] = 0x55, [0xee] = 0x28, [0xef] = 0xdf,
        [0xf0] = 0x8c, [0xf1] = 0xa1, [0xf2] = 0x89, [0xf3] = 0x0d,
        [0xf4] = 0xbf, [0xf5] = 0xe6, [0xf6] = 0x42, [0xf7] = 0x68,
        [0xf8] = 0x41, [0xf9] = 0x99, [0xfa] = 0x2d, [0xfb] = 0x0f,
        [0xfc] = 0xb0, [0xfd] = 0x54, [0xfe] = 0xbb, [0xff] = 0x16
    };
    return s_box[byte];
}

static inline uint8_t inv_s_box_replace(uint8_t byte) {
    static const uint8_t inv_s_box[256] = {
        [0x00] = 0x52, [0x01] = 0x09, [0x02] = 0x6a, [0x03] = 0xd5,
        [0x04] = 0x30, [0x05] = 0x36, [0x06] = 0xa5, [0x07] = 0x38,
        [0x08] = 0xbf, [0x09] = 0x40, [0x0a] = 0xa3, [0x0b] = 0x9e,
        [0x0c] = 0x81, [0x0d] = 0xf3, [0x0e] = 0xd7, [0x0f] = 0xfb,
        [0x10] = 0x7c, [0x11] = 0xe3, [0x12] = 0x39, [0x13] = 0x82,
        [0x14] = 0x9b, [0x15] = 0x2f, [0x16] = 0xff, [0x17] = 0x87,
        [0x18] = 0x34, [0x19] = 0x8e, [0x1a] = 0x43, [0x1b] = 0x44,
        [0x1c] = 0xc4, [0x1d] = 0xde, [0x1e] = 0xe9, [0x1f] = 0xcb,
        [0x20] = 0x54, [0x21] = 0x7b, [0x22] = 0x94, [0x23] = 0x32,
        [0x24] = 0xa6, [0x25] = 0xc2, [0x26] = 0x23, [0x27] = 0x3d,
        [0x28] = 0xee, [0x29] = 0x4c, [0x2a] = 0x95, [0x2b] = 0x0b,
        [0x2c] = 0x42, [0x2d] = 0xfa, [0x2e] = 0xc3, [0x2f] = 0x4e,
        [0x30] = 0x08, [0x31] = 0x2e, [0x32] = 0xa1, [0x33] = 0x66,
        [0x34] = 0x28, [0x35] = 0xd9, [0x36] = 0x24, [0x37] = 0xb2,
        [0x38] = 0x76, [0x39] = 0x5b, [0x3a] = 0xa2, [0x3b] = 0x49,
        [0x3c] = 0x6d, [0x3d] = 0x8b, [0x3e] = 0xd1, [0x3f] = 0x25,
        [0x40] = 0x72, [0x41] = 0xf8, [0x42] = 0xf6, [0x43] = 0x64,
        [0x44] = 0x86, [0x45] = 0x68, [0x46] = 0x98, [0x47] = 0x16,
        [0x48] = 0xd4, [0x49] = 0xa4, [0x4a] = 0x5c, [0x4b] = 0xcc,
        [0x4c] = 0x5d, [0x4d] = 0x65, [0x4e] = 0xb6, [0x4f] = 0x92,
        [0x50] = 0x6c, [0x51] = 0x70, [0x52] = 0x48, [0x53] = 0x50,
        [0x54] = 0xfd, [0x55] = 0xed, [0x56] = 0xb9, [0x57] = 0xda,
        [0x58] = 0x5e, [0x59] = 0x15, [0x5a] = 0x46, [0x5b] = 0x57,
        [0x5c] = 0xa7, [0x5d] = 0x8d, [0x5e] = 0x9d, [0x5f] = 0x84,
        [0x60] = 0x90, [0x61] = 0xd8, [0x62] = 0xab, [0x63] = 0x00,
        [0x64] = 0x8c, [0x65] = 0xbc, [0x66] = 0xd3, [0x67] = 0x0a,
        [0x68] = 0xf7, [0x69] = 0xe4, [0x6a] = 0x58, [0x6b] = 0x05,
        [0x6c] = 0xb8, [0x6d] = 0xb3, [0x6e] = 0x45, [0x6f] = 0x06,
        [0x70] = 0xd0, [0x71] = 0x2c, [0x72] = 0x1e, [0x73] = 0x8f,
        [0x74] = 0xca, [0x75] = 0x3f, [0x76] = 0x0f, [0x77] = 0x02,
        [0x78] = 0xc1, [0x79] = 0xaf, [0x7a] = 0xbd, [0x7b] = 0x03,
        [0x7c] = 0x01, [0x7d] = 0x13, [0x7e] = 0x8a, [0x7f] = 0x6b,
        [0x80] = 0x3a, [0x81] = 0x91, [0x82] = 0x11, [0x83] = 0x41,
        [0x84] = 0x4f, [0x85] = 0x67, [0x86] = 0xdc, [0x87] = 0xea,
        [0x88] = 0x97, [0x89] = 0xf2, [0x8a] = 0xcf, [0x8b] = 0xce,
        [0x8c] = 0xf0, [0x8d] = 0xb4, [0x8e] = 0xe6, [0x8f] = 0x73,
        [0x90] = 0x96, [0x91] = 0xac, [0x92] = 0x74, [0x93] = 0x22,
        [0x94] = 0xe7, [0x95] = 0xad, [0x96] = 0x35, [0x97] = 0x85,
        [0x98] = 0xe2, [0x99] = 0xf9, [0x9a] = 0x37, [0x9b] = 0xe8,
        [0x9c] = 0x1c, [0x9d] = 0x75, [0x9e] = 0xdf, [0x9f] = 0x6e,
        [0xa0] = 0x47, [0xa1] = 0xf1, [0xa2] = 0x1a, [0xa3] = 0x71,
        [0xa4] = 0x1d, [0xa5] = 0x29, [0xa6] = 0xc5, [0xa7] = 0x89,
        [0xa8] = 0x6f, [0xa9] = 0xb7, [0xaa] = 0x62, [0xab] = 0x0e,
        [0xac] = 0xaa, [0xad] = 0x18, [0xae] = 0xbe, [0xaf] = 0x1b,
        [0xb0] = 0xfc, [0xb1] = 0x56, [0xb2] = 0x3e, [0xb3] = 0x4b,
        [0xb4] = 0xc6, [0xb5] = 0xd2, [0xb6] = 0x79, [0xb7] = 0x20,
        [0xb8] = 0x9a, [0xb9] = 0xdb, [0xba] = 0xc0, [0xbb] = 0xfe,
        [0xbc] = 0x78, [0xbd] = 0xcd, [0xbe] = 0x5a, [0xbf] = 0xf4,
        [0xc0] = 0x1f, [0xc1] = 0xdd, [0xc2] = 0xa8, [0xc3] = 0x33,
        [0xc4] = 0x88, [0xc5] = 0x07, [0xc6] = 0xc7, [0xc7] = 0x31,
        [0xc8] = 0xb1, [0xc9] = 0x12, [0xca] = 0x10, [0xcb] = 0x59,
        [0xcc] = 0x27, [0xcd] = 0x80, [0xce] = 0xec, [0xcf] = 0x5f,
        [0xd0] = 0x60, [0xd1] = 0x51, [0xd2] = 0x7f, [0xd3] = 0xa9,
        [0xd4] = 0x19, [0xd5] = 0xb5, [0xd6] = 0x4a, [0xd7] = 0x0d,
        [0xd8] = 0x2d, [0xd9] = 0xe5, [0xda] = 0x7a, [0xdb] = 0x9f,
        [0xdc] = 0x93, [0xdd] = 0xc9, [0xde] = 0x9c, [0xdf] = 0xef,
        [0xe0] = 0xa0, [0xe1] = 0xe0, [0xe2] = 0x3b, [0xe3] = 0x4d,
        [0xe4] = 0xae, [0xe5] = 0x2a, [0xe6] = 0xf5, [0xe7] = 0xb0,
        [0xe8] = 0xc8, [0xe9] = 0xeb, [0xea] = 0xbb, [0xeb] = 0x3c,
        [0xec] = 0x83, [0xed] = 0x53, [0xee] = 0x99, [0xef] = 0x61,
        [0xf0] = 0x17, [0xf1] = 0x2b, [0xf2] = 0x04, [0xf3] = 0x7e,
        [0xf4] = 0xba, [0xf5] = 0x77, [0xf6] = 0xd6, [0xf7] = 0x26,
        [0xf8] = 0xe1, [0xf9] = 0x69, [0xfa] = 0x14, [0xfb] = 0x63,
        [0xfc] = 0x55, [0xfd] = 0x21, [0xfe] = 0x0c, [0xff] = 0x7d,
    };
    return inv_s_box[byte];
}
#endif
