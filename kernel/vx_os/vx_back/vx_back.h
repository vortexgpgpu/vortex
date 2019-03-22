

#pragma once

#include <stdbool.h>
#include "../vx_util/queue.h"

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

static int done[] = {0, 0, 0, 0, 0, 0, 0};

static int main_sp[1];

unsigned context[32];
void  vx_save_context(void);
void  vx_load_context(void);


#define FUNC void (func)(unsigned, unsigned)

unsigned vx_available_warps(void);
unsigned vx_available_threads(void);


void vx_createThreads(unsigned, unsigned, unsigned, void *, unsigned);
void        vx_wspawn(unsigned, unsigned, unsigned, void *, unsigned);
void   vx_spawnWarps(unsigned num_Warps, unsigned num_threads, FUNC, void *);
void   vx_schedule_warps(void);
void   vx_reschedule_warps(void);
void   vx_wait_for_warps(unsigned);
void * vx_get_arg_struct(void);


