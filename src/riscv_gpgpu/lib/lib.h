
#ifndef __RISCV_GP_
#define __RISCV_GP_

#include "queue.h"

#define WSPAWN          asm __volatile__(".word 0x3006b"::);
#define CLONE           asm __volatile__(".word 0x3506b":::);
#define JALRS           asm __volatile__(".word 0x1bfe0eb":::"s10");
#define ECALL           asm __volatile__(".word 0x00000073");
#define JMPRT           asm __volatile__(".word 0x5406b");
#define SPLIT           asm __volatile__(".word 0xf206b");
#define P_JUMP          asm __volatile__(".word 0x1ff706b");
#define JOIN            asm __volatile__(".word 0x306b");


// #define __if(val) { \

// 		register unsigned   p asm("t5") = val; \
// 		register unsigned * e asm("t6") = &&ELSE; \
// 		SPLIT; \
// 		P_JUMP; \

	
// }

// #define __else asm __volatile__("j AFTER"); \
// 			   ELSE: asm __volatile__("nop");

// #define __end_if AFTER: JOIN;


#define FUNC void (func)(unsigned, unsigned)
void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, unsigned *, unsigned *, unsigned *);
void reschedule_warps(void);

unsigned * get_1st_arg(void);
unsigned * get_2nd_arg(void);
unsigned * get_3rd_arg(void);
void       sleep(int);


#endif

