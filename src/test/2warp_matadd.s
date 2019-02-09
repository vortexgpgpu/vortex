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
entry: ldi %r20, #1
       ldi %r21, Begin
       wspawn %r20, %r21, %r20
       ldi %r20, #0
Begin: ldi %r0, #1
       ldi %r1, THREADS
tc_loop: clone %r0
       
       addi %r0, %r0, #1
       sub %r2, %r1, %r0
       rtop @p0, %r2
 @p0 ? jmpi tc_loop

       ldi %r0, #0
       jalis %r5, %r1, dthread;

       ldi %r25, #55
       ldi %r26, #1
       bar %r25, %r26

       subi %r20, %r20, #1

       iszero @p0, %r20

  @p0 ? trap;

       ldi %r0, #0
       ldi %r1, (__WORD * THREADS)
       shli %r1, %r1, #1

ploop: ld %r7, %r0, RESULT
       jali %r5, printdec
       
       addi %r0, %r0, __WORD
       sub %r7, %r1, %r0
       rtop @p0, %r7
 @p0 ? jmpi ploop

       trap;


dthread: shli  %r15, %r20 , #6
         shli  %r10, %r0 , #3
         add   %r10, %r10, %r15
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
  .word 1 5 10 0 3 1 1 2
  .word 8 7 8 7 5 7 7 9
Array2:
  .word 0 2 2 0 5 0 1 1
  .word 4 2 2 0 3 2 3 2
RESULT: .space 512
