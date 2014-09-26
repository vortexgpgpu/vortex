/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* Simple example. */

.align 4096
.perm x
.entry
.global

entry:       ldi %r0, wentry
             ldi %r7, hello2
             /* wspawn %r0, %r7 */
             ldi %r0, hello1

wentry:      ori %r7, %r0, #0
             jali %r5, puts

             trap; /* All traps currently cause a halt. */

.perm rw

hello1:
.byte 0x22
.string "Harp!\" is how a harp seal says hello!\n"

hello2:
 .string "This is a string for another thread!\n"    
