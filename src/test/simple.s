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
entry:       ldi %r7, hello
             jali %r5, puts
             ldi  %r0, #2; /* i = 2 */

             trap; /* All traps currently cause a halt. */

.perm rw /* TODO: How should I write section permissions? */
/* TODO: String literals! */
.string hello "\"Harp!\" is how a harp seal says hello!\n"
