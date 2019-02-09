/*******************************************************************************
 Harptools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* Divergent branch: test immediate postdominator branch divergence support. */
.def THREADS 8

.align 4096
.perm x
.entry
.global
entry:
       ldi %r0, #1
       ldi %r1, THREADS
tc_loop: clone %r0
       
       addi %r0, %r0, #1
       sub %r2, %r1, %r0
       rtop @p0, %r2
 @p0 ? jmpi tc_loop

       ldi %r0, #0
       jalis %r5, %r1, dthread;

       ldi %r0, #0
       ldi %r1, (__WORD * THREADS)

ploop: ld %r7, %r0, RESULT
       jali %r5, printdec
       
       addi %r0, %r0, __WORD
       sub %r7, %r1, %r0
       rtop @p0, %r7
 @p0 ? jmpi ploop

       trap;


dthread: shli  %r10, %r0 , #3
         ld    %r11, %r10, Array1
         ld    %r12, %r10, Array2

         subi  %r13, %r0, #4
         isneg @p0, %r13
   @p0 ? split
   @p0 ? jmpi SUBT
         add %r14, %r11, %r12
         jmpi after
SUBT:    sub %r14, %r11, %r12
after:   join

         st %r14, %r10, RESULT

         jmprt %r5;

.align 4096
Array1:
  .word 1 5 10 0
  .word 3 1 1 2
Array2:
  .word 0 2 2 0
  .word 5 0 1 1
RESULT: .space 512
