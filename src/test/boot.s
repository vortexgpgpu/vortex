/* Bootstrap program for HARP. */

.perm x
.entry

boot: ldi   %r5, kernEnt;
      skep  %r5;

/*      ldi   %r0, #1;
      ldi   %r1, #033;
      ldi %r2, __WORD;
      muli %r2, %r2, #8;
      subi %r2, %r2, #1;
      shl %r0, %r0, %r2;
      tlbadd %r0, %r0, %r1; */

      ei;

      ldi   %r5, entry;
      jmpru %r5;

.perm x
/* The Kernel Entry Point / Interrupt service routine. */
kernEnt:  subi %r0, %r0, #1;
          rtop @p0, %r0;
    @p0 ? jmpi kernEnt1; /* If it's not page not found, try again. */

          ldi %r0, #077; /* Just map virt to phys, any address. */
          tlbadd %r1, %r1, %r0;
          reti;
    
kernEnt1: subi %r0, %r0, #7; /* If it's not console input, halt.*/
          rtop @p0, %r0;
    @p0 ? halt;
    
          ldi %r8, #1;
          ldi %r1, __WORD;
          muli %r1, %r1, #8;
          subi %r1, %r1, #1;
          shl %r8, %r8, %r1;
          
          ld %r0, %r8, #0;
          subi %r1, %r0, #0x71
          rtop @p0, %r1
          notp @p0, @p0
    @p0 ? halt; /* If it's 'q', halt. */
          st %r0, %r8, #0;
          ldi %r0, #0xa;
          st %r0, %r8, #0;
          reti;


