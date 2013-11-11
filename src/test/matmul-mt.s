/*******************************************************************************
 Harptools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* Matrix multiply: find matrix product */
.def THREADS 2

.align 4096
.perm x
.entry
.global
entry: ldi %r0, matrix_a;
       ldi %r1, #3;
       jali %r5, matgen;

       ldi %r0, matrix_b;
       ldi %r1, #3;
       jali %r5, matgen;

       ldi %r0, matrix_a;
       ldi %r1, matrix_b;
       ldi %r2, matrix_r;
       ldi %r3, #3;
       jali %r5, matmul;

       ldi %r0, #64;
       ldi %r1, matrix_r;
ploop: ld %r7, %r1, #0;
       jali %r5, printfloat;
       subi %r0, %r0, #1;
       addi %r1, %r1, __WORD;
       rtop @p0, %r0;
 @p0 ? jmpi ploop;

       trap;

/* Generate a random 2^Nx2^N matrix at %r0, with N in %r1 */
matgen: ldi %r2, #0;
        st %r5, %r2, retaddr;
        ldi %r2, #1;
        shl %r2, %r2, %r1;
        shl %r2, %r2, %r1;
        ori %r3, %r0, #0;

mgloop: jali %r5, randf;
        st %r0, %r3, #0;
        addi %r3, %r3, __WORD;
        subi %r2, %r2, #1;
        rtop @p0, %r2;
  @p0 ? jmpi mgloop;
        
        ldi %r2, #0;
        ld %r5, %r2, retaddr;
        jmpr %r5;        

/* Write the matrix product of square matrix at (%r0) and (%r1) to (%r2). The
   size of these matrices is 2^Nx2^N, where N = %r3 */

matmul: ori %r22, %r5, #0;
        ldi %r4, #1;
        ldi %r10, (`__WORD); /* ` is the log base 2 operator */
        shl %r4, %r4, %r3;
        add %r10, %r10, %r3;
        ldi %r14, #1;
        shl %r14, %r14, %r10;
        shl %r17, %r14, %r3;

        divi %r17, %r17, THREADS; /* Spawn threads */
        divi %r24, %r4, THREADS;
        ori %r18, %r0, #0;
        ori %r19, %r2, #0;
        ldi %r20, #0;
sloop:  add %r0, %r0, %r17;
        add %r2, %r2, %r17;
        addi %r20, %r20, #1;
        subi %r21, %r20, THREADS;
        rtop @p0, %r21;
  @p0 ? clone %r20;

        ori %r0, %r18, #0;
        ori %r2, %r19, #0;

  @p0 ? jmpi sloop;

        ldi %r20, THREADS;
        jalis %r5, %r20, matmulthd;  

        jmpr %r22;

/* One thread of matrix multiplication. Expected register values at start:
 *   %r0 - matrix a pointer (plus offset)
 *   %r1 - matrix b pointer
 *   %r2 - destination matrix pointer (plus offset)
 *   %r24 - row count
 */
matmulthd: ldi %r9, #0; /* result row: %r9 */
rloop:     ldi %r6, #0; /* result col: %r6 */
cloop:     shli %r16, %r6, (`__WORD);
           shl %r15, %r9, %r10;

           add %r11, %r15, %r0;
           add %r12, %r16, %r1;
           ldi %r13, #0;

           ldi %r8, #0 /* dot prod position: %r8 */
iloop:     ld %r7, %r11, #0;
           ld %r23, %r12, #0;
           fmul %r7, %r7, %r23
           fadd %r13, %r13, %r7;

           addi %r8, %r8, #1;
           addi %r11, %r11, __WORD;
           add %r12, %r12, %r14;
           sub %r7, %r8, %r4;
           rtop @p0, %r7;
     @p0 ? jmpi iloop;

           add %r15, %r15, %r16;
           add %r15, %r15, %r2;
           st %r13, %r15, #0;

           addi %r6, %r6, #1;
           sub %r7, %r6, %r4;
           rtop @p0, %r7;
     @p0 ? jmpi cloop;

           addi %r9, %r9, #1;
           sub %r7, %r9, %r24;
           rtop @p0, %r7;
     @p0 ? jmpi rloop;

           jmprt %r5;

.perm rw
.align 4096

matrix_a: .space 64
matrix_b: .space 64
matrix_r: .space 64

retaddr: .word 0
