#ifndef AES256_H
#define AES256_H

#include <stdint.h>

// 4 32-bit columns in an AES state
#define Nb 4
// 14 rounds in AES-256 cipher
#define Nr 14
// 8 words in AES-256 key
#define Nk 8

#define BLOCK_SIZE (4 * Nb)
#define KEY_SIZE (4 * Nk)
#define NO_XOR ((void *)-1)

extern void aes256_ecb_enc(const uint8_t *, const uint8_t *, uint8_t *, int);
extern void aes256_ecb_dec(const uint8_t *, const uint8_t *, uint8_t *, int);
extern void aes256_cbc_enc(const uint8_t *, const uint8_t *, const uint8_t *,
                           uint8_t *, int);
extern void aes256_cbc_dec(const uint8_t *, const uint8_t *, const uint8_t *,
                           uint8_t *, int);
extern void aes256_ctr(const uint8_t *, uint32_t, const uint8_t *,
                       const uint8_t *, uint8_t *, int);
#endif
