#ifndef _COMMON_H_
#define _COMMON_H_

#include "aes256.h"

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef enum {
    AES_OP_ECB_ENC,
    AES_OP_ECB_DEC,
    AES_OP_CBC_ENC,
    AES_OP_CBC_DEC,
    AES_OP_CTR_ENC,
    AES_OP_CTR_DEC,
    // Just key expansion, no ciphers
    AES_OP_KEY_ENC,
    AES_OP_KEY_DEC,
    // Number of different operations
    AES_OP_COUNT
} aes_op_type_t;

struct kernel_arg_t {
  uint32_t num_tasks;
  uint32_t nblocks;
  uint32_t aes_op_type;
  uint32_t key_ptr;
  uint32_t in_ptr;
  uint32_t out_ptr;
  uint8_t key[KEY_SIZE];
  uint8_t iv[BLOCK_SIZE];
} __attribute__((packed));

#endif
