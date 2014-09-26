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
sloop: clone %r0
       
       addi %r0, %r0, #1
       sub %r2, %r1, %r0
       rtop @p0, %r2
 @p0 ? jmpi sloop

       ldi %r0, #0
       jalis %r5, %r1, dthread;

       ldi %r0, #0
       ldi %r1, (__WORD * THREADS)

ploop: ld %r7, %r0, array
       jali %r5, printdec
       
       addi %r0, %r0, __WORD
       sub %r7, %r1, %r0
       rtop @p0, %r7
 @p0 ? jmpi ploop

       trap;


dthread: ldi %r1, #10
         ldi %r2, #0

loop:    andi %r3, %r0, #1
         rtop @p1, %r3
   @p1 ? split
   @p1 ? jmpi else
         add %r2, %r2, %r0
         jmpi after
else:    sub %r2, %r2, %r0
after:   join

         subi %r1, %r1, #1
         rtop @p0, %r1
   @p0 ? jmpi loop

         shli %r4, %r0, (`__WORD)
         st %r2, %r4, array

         jmprt %r5;

.align 4096
array: .space 4096
