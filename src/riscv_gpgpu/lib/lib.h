
#ifndef __RISCV_GP_
#define __RISCV_GP_

#include "queue.h"

#define WSPAWN          asm __volatile__(".word 0x3006b"::);
#define CLONE           asm __volatile__(".word 0x3506b":::);
#define JALRS           asm __volatile__(".word 0x1bfe0eb":::"s10")
#define ECALL           asm __volatile__(".word 0x00000073")
#define JMPRT           asm __volatile__(".word 0x5406b")


#define FUNC void (func)(unsigned, unsigned)
void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, unsigned *, unsigned *, unsigned *);
void reschedule_warps(void);

unsigned   get_wid();
unsigned * get_1st_arg(void);
unsigned * get_2nd_arg(void);
unsigned * get_3rd_arg(void);
void       initiate_stack();



#endif

