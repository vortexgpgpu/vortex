/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* sieve of eratosthanes: Find some primes. */
.def SIZE 0x1000

.align 4096
.perm x
.entry
.global
entry: ldi %r0, array_a;
       ldi %r1, array_b;
       ldi %r2, #10;
       jali %r5, dotprod

       ori %r7, %r0, #0;
       jali %r5, printfloat;

       trap;
 
/* Return in r0 dot product of vectors of real values pointed to by r0 and r1,
   length in r2 */
dotprod: ldi %r3, #0;
         st %r5, %r3, retaddr;
dploop:  ld %r4, %r0, #0;
         ld %r6, %r1, #0;
         subi %r2, %r2, #1;
         addi %r0, %r0, __WORD;
         addi %r1, %r1, __WORD;
         rtop @p0, %r2;
         fmul %r4, %r4, %r6;
         fadd %r3, %r3, %r4;
   @p0 ? jmpi dploop;
         ori %r0, %r3, #0;
         ldi %r3, retaddr;
         ld %r5, %r3, #0;
         jmpr %r5;

.align 4096
.perm rw
array_a: .word 1.0 2.0 3.0 0.5 1.0 1.5 0.33 0.67 1.0 1.33
array_b: .word 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1f
retaddr: .word 0


