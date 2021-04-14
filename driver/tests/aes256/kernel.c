#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <string.h>
#include "common.h"
#include "aes256.h"

void kernel_body(int task_id, const void* arg) {
    struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
    uint32_t nblocks = _arg->nblocks;
    uint8_t* key_ptr = (uint8_t*)_arg->key_ptr;
    uint8_t* indec_ptr = (uint8_t*)_arg->indec_ptr;
    uint8_t* inenc_ptr = (uint8_t*)_arg->inenc_ptr;
    uint8_t* outdec_ptr = (uint8_t*)_arg->outdec_ptr;
    uint8_t* outenc_ptr = (uint8_t*)_arg->outenc_ptr;

    uint32_t offset = task_id * nblocks * BLOCK_SIZE;
    aes256enc(indec_ptr + offset, key_ptr, outenc_ptr + offset, nblocks);
    aes256dec(inenc_ptr + offset, key_ptr, outdec_ptr + offset, nblocks);
}

void main() {
    struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
    vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}
