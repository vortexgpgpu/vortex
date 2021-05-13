#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <string.h>
#include "common.h"
#include "aes256.h"

struct extended_arg {
    struct kernel_arg_t *host_arg;
    uint32_t round_keys[Nb * (Nr + 1)];
};

static void kernel_body(int task_id, const void* _ext_arg) {
    const struct extended_arg *ext_arg = _ext_arg;
    struct kernel_arg_t* _arg = (struct kernel_arg_t*)(ext_arg->host_arg);
    uint32_t nblocks = _arg->nblocks;
    uint8_t* in_ptr = (uint8_t*)_arg->in_ptr;
    uint8_t* out_ptr = (uint8_t*)_arg->out_ptr;
    aes_op_type_t aes_op_type = _arg->aes_op_type;

    uint32_t start_block_idx = task_id * nblocks;
    uint32_t offset = start_block_idx * BLOCK_SIZE;

    switch (aes_op_type) {
        case AES_OP_ECB_ENC:
            aes256_ecb_enc(in_ptr + offset, ext_arg->round_keys, out_ptr + offset, nblocks);
            break;
        case AES_OP_ECB_DEC:
            aes256_ecb_dec(in_ptr + offset, ext_arg->round_keys, out_ptr + offset, nblocks);
            break;
        case AES_OP_CBC_DEC:
            {
                const uint8_t *iv_choices[] = {_arg->iv, in_ptr + offset - BLOCK_SIZE};
                const uint8_t *iv = iv_choices[!!offset];
                aes256_cbc_dec(iv, in_ptr + offset, ext_arg->round_keys, out_ptr + offset, nblocks);
            }
            break;
        case AES_OP_CTR_ENC:
        case AES_OP_CTR_DEC:
            aes256_ctr(_arg->iv, start_block_idx, in_ptr + offset, ext_arg->round_keys,
                       out_ptr + offset, nblocks);
            break;
        default:
            // No worries
            ;
    }
}

void main() {
    struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
    struct extended_arg ext_arg;
    ext_arg.host_arg = arg;

    int inv_mix_cols = arg->aes_op_type == AES_OP_ECB_DEC
                       || arg->aes_op_type == AES_OP_CBC_DEC
                       || arg->aes_op_type == AES_OP_KEY_DEC;
    aes256_key_exp((const uint32_t *)arg->key, ext_arg.round_keys, inv_mix_cols);

    if (arg->aes_op_type == AES_OP_KEY_ENC || arg->aes_op_type == AES_OP_KEY_DEC) {
        // Just do key expansion and exit
        return;
    } else if (arg->aes_op_type == AES_OP_CBC_ENC) {
        // CBC encryption is not parallelizable
        uint32_t nblocks = arg->nblocks * arg->num_tasks;
        aes256_cbc_enc(arg->iv, (uint8_t *)arg->in_ptr, ext_arg.round_keys, (uint8_t *)arg->out_ptr, nblocks);
    } else {
        vx_spawn_tasks(arg->num_tasks, kernel_body, &ext_arg);
    }
}
