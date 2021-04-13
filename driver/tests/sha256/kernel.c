#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"
#include "sha256.h"

void kernel_body(int task_id, const void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t nmsg    = _arg->nmsg;
	uint32_t msgsize = _arg->msgsize;
	uint8_t* msg_ptr = (uint8_t*)_arg->msg_ptr;
	uint8_t* digest_ptr  = (uint8_t*)_arg->digest_ptr;
	
	uint32_t offset = task_id * nmsg;
    uint32_t padded_msgsize = PADDED_SIZE_BYTES(msgsize);

	for (uint32_t i = 0; i < nmsg; ++i) {
        uint8_t *msg = msg_ptr + (offset + i) * padded_msgsize;
        uint8_t *digest = digest_ptr + (offset + i) * DIGEST_BYTES;
        sha256(msg, msgsize, digest);
	}
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}
