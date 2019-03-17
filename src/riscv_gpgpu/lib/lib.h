
#ifndef __RISCV_GP_
#define __RISCV_GP_
#include <stdbool.h>
#include "queue.h"

#define WSPAWN          asm __volatile__(".word 0x3006b"::);
#define CLONE           asm __volatile__(".word 0x3506b":::);
#define JALRS           asm __volatile__(".word 0x1bfe0eb":::"s10");
#define ECALL           asm __volatile__(".word 0x00000073");
#define JMPRT           asm __volatile__(".word 0x5406b");
#define SPLIT           asm __volatile__(".word 0xf206b");
#define P_JUMP          asm __volatile__(".word 0x1ff707b");
#define JOIN            asm __volatile__(".word 0x306b");


#define __if(val)  bool temp = !val; \
		register unsigned   p asm("t5") = temp; \
		register void * e asm("t6") = &&ELSE; \
		SPLIT; \
		P_JUMP; \


#define __else register void * w asm("t3") =  &&AFTER; \
			   asm __volatile__("jr t3"); \
			   ELSE: asm __volatile__("nop");

#define __end_if AFTER:\
			   JOIN;


static char * hextoa[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"};
static bool done[] = {false, false, false, false, false, false, false};

static int main_sp[1];

#define FUNC void (func)(unsigned, unsigned)
void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, void *, void *, void *);
void reschedule_warps(void);
void int_print(unsigned);
void wait_for_done(unsigned);

void * get_1st_arg(void);
void * get_2nd_arg(void);
void * get_3rd_arg(void);
void       sleep(int);


#endif

