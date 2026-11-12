#ifndef _COMMON_H_
#define _COMMON_H_

#include <flash_attn_args.h>

// The tensor core has no fp32 multiplier, so the inputs are tf32, fp16 or
// bf16; it accumulates in fp32, which softmax also needs.
#ifndef ITYPE
#define ITYPE tf32
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Valid for WGMMA: 8, 16, 32
#ifndef WGMMA_NRC
#define WGMMA_NRC 8
#endif

#endif
