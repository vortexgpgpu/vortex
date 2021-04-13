// From: https://github.gatech.edu/aadams80/cs7290-algos

#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>

// The formula in the spec for determining the number of padding zeroes
// k is to find the minimum nonnegative k such that
//
//     l + 1 + k = 448 (mod 512)
//
// where l is the number of bits of the data and k is padding bits after
// the 1 separating the padding bits from the data. This strange-looking
// formula makes sure that we choose the number of padding bits k such
// that the entire message is aligned on 512 bits. A big endian 64-bit
// field follows this padding and holds l, hence why they chose 448: 64
// + 448 = 512.
//
// But we are operating on bytes in this implementation, not bits, so we
// can simplify this formula by including that 8-byte length field and
// using bytes (here, l' and k' are in units of bytes; k' does not
// include the byte for 0b10000000, which is the +1 below):
//
//     l' + 1 + k' + 8 = 0       (mod 64)
//         l' + k' + 9 = 0       (mod 64)
//                  k' = -9 - l' (mod 64)
//                  k' = 55 - l' (mod 64)
//
// This is a little better but still l' could be very large, leading to
// confusing behavior thanks to the % operator in C, which can actually
// give negative answers when the LHS is negative. Let us introduce l'',
// which is the smallest nonnegative integer such that l'' = l' (mod
// 64). Then we have:
//
//              k' = 55 - l'' (mod 64)
//
// However, l'' could still be as large as 63, leading to negative
// numbers again. We can address this with:
//
//              k' = 119 - l'' (mod 64)
//
// This gives rise to the C expression we use below to find the padding
// bytes k' given data bytes = l':
//
//     padding_bytes = (119 - (data_bytes % 64)) % 64

#define PADDING_BYTES(data_bytes) ((119 - ((data_bytes) % 64)) % 64)
#define PADDED_SIZE_BYTES(data_bytes) ((data_bytes) + 1 + PADDING_BYTES(data_bytes) + 8)
#define DIGEST_BYTES 32

extern void sha256(uint8_t *, uint32_t, uint8_t *);

#endif
