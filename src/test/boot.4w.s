.perm x

.entry
boot:	ldi %r5 kernEnt;
	skep %r5 ;
	ldi %r0 #0x1;
	ldi %r1 #0x18;
	ldi %r2 #0x4;
	muli %r2 %r2 #0x8;
	subi %r2 %r2 #0x1;
	shl %r0 %r0 %r2 ;
	tlbadd %r0 %r0 %r1 ;
	ei ;
	ldi %r5 entry;
	jmpru %r5 ;
kernEnt:	subi %r0 %r0 #0x8;
	rtop @p0 %r0 ;
	@p0 ? reti ;
	ldi %r8 #0x1;
	ldi %r1 #0x4;
	muli %r1 %r1 #0x8;
	subi %r1 %r1 #0x1;
	shl %r8 %r8 %r1 ;
	ld %r0 %r8 #0x0;
	subi %r1 %r0 #0x71;
	rtop @p0 %r1 ;
	notp @p0 @p0 ;
	@p0 ? halt ;
	st %r0 %r8 #0x0;
	ldi %r0 #0xa;
	st %r0 %r8 #0x0;
	reti ;
entry:	ldi %r7 hello;
	jali %r5 puts;
	jmpi entry;
.perm rw
.word hello 0x6c6c6548
.word __anonWord0 0x41202c6f
.word __anonWord1 0x6e616c74
.word __anonWord2 0xa216174
.word __anonWord3 0x0
