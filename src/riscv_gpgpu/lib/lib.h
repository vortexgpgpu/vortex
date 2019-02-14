
#ifndef __RISCV_GP_
#define __RISCV_GP_


#define WID_CSR  0x00E
#define FUNC_CSR 0x00F

#define SET_WID(val)    asm __volatile__("csrw 0x00e,%0"::"r"(val));
#define GET_WID(ret)   asm __volatile__("csrr %0,0x00e":"=r"(ret));

#define SET_FUNC(val)   asm __volatile__("csrw 0x00f,%0"::"r"(val));
#define GET_FUNC(ret)   asm __volatile__("csrr %0,0x00f":"=r"(ret));


#define WSPAWN          asm __volatile__(".word 0x3006b"::);
#define CLONE           asm __volatile__(".word 0x3506b":::"t1");
#define JALRS           asm __volatile__(".word 0x1bfe0eb":::"s10")
#define ECALL           asm __volatile__(".word 0x00000073")


#define FUNC void (func)(unsigned)
void createWarps(unsigned num_Warps, unsigned num_threads, FUNC);
unsigned get_tid(void);
void initiate_stack(void);



#endif

