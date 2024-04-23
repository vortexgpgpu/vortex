# Generated automatically from makefile.in by configure.
#
# Makefile for Meschach via autoconf
#
# Copyright (C) David Stewart & Zbigniew Leyk 1993
#
# $Id: makefile.in,v 1.4 1994/03/14 01:24:06 des Exp $
#

srcdir = .
VPATH = .

CC = gcc

DEFS = -DHAVE_CONFIG_H
LIBS =  -lm
RANLIB = ranlib


CFLAGS = -O2 -ffast-math -fexpensive-optimizations


.c.o:
	$(CC) -c $(CFLAGS) $(DEFS) $<

SHELL = /bin/sh
MES_PAK = mesch12b
TAR = tar
SHAR = stree -u
ZIP = zip -r -l
FLIST = FILELIST

###############################

LIST1 = copy.o err.o matrixio.o memory.o vecop.o matop.o pxop.o \
	submat.o init.o otherio.o machine.o matlab.o ivecop.o version.o \
	meminfo.o memstat.o
LIST2 = lufactor.o bkpfacto.o chfactor.o qrfactor.o solve.o hsehldr.o \
	givens.o update.o norm.o hessen.o symmeig.o schur.o svd.o fft.o \
	mfunc.o bdfactor.o
LIST3 = sparse.o sprow.o sparseio.o spchfctr.o splufctr.o \
	spbkp.o spswap.o iter0.o itersym.o iternsym.o
ZLIST1 = zmachine.o zcopy.o zmatio.o zmemory.o zvecop.o zmatop.o znorm.o \
	 zfunc.o 
ZLIST2 = zlufctr.o zsolve.o zmatlab.o zhsehldr.o zqrfctr.o \
         zgivens.o  zhessen.o zschur.o

# they are no longer supported
# if you use them add oldpart to all and sparse
OLDLIST = conjgrad.o lanczos.o arnoldi.o

ALL_LISTS = $(LIST1) $(LIST2) $(LIST3) $(ZLIST1) $(ZLIST2) $(OLDLIST)

HBASE = err.h meminfo.h machine.h matrix.h

HLIST = $(HBASE) iter.h matlab.h matrix2.h  oldnames.h sparse.h \
	sparse2.h  zmatrix.h zmatrix2.h

TORTURE = torture.o sptort.o ztorture.o memtort.o itertort.o \
	 mfuntort.o iotort.o

OTHERS = dmacheps.c extras.c fmacheps.c maxint.c  makefile.in \
	 README configure configure.in machine.h.in copyright \
	 tutorial.c tutadv.c rk4.dat ls.dat makefile $(FLIST)


# Different configurations
# the dependencies **between** the parts are for dmake
all:  part1 part2 part3 zpart1 zpart2 
part2: part1
part3: part2
basic: part1 part2
sparse: part1 part2 part3 
zpart2: zpart1
complex: part1 part2 zpart1 zpart2


$(LIST1): $(HBASE)
part1: $(LIST1)
	ar ru libmeschach.a $(LIST1)
	$(RANLIB) libmeschach.a

$(LIST2): $(HBASE) matrix2.h
part2: $(LIST2)
	ar ru libmeschach.a $(LIST2)
	$(RANLIB) libmeschach.a

$(LIST3): $(HBASE) sparse.h sparse2.h
part3: $(LIST3)
	ar ru libmeschach.a $(LIST3)
	$(RANLIB) libmeschach.a

$(ZLIST1): $(HBASDE) zmatrix.h
zpart1: $(ZLIST1)
	ar ru libmeschach.a $(ZLIST1)
	$(RANLIB) libmeschach.a

$(ZLIST2): $(HBASE) zmatrix.h zmatrix2.h 
zpart2: $(ZLIST2)
	ar ru libmeschach.a $(ZLIST2)
	$(RANLIB) libmeschach.a

$(OLDLIST): $(HBASE) sparse.h sparse2.h 
oldpart: $(OLDLIST)
	ar ru libmeschach.a $(OLDLIST)
	$(RANLIB) libmeschach.a



#######################################

tar:
	- /bin/rm -f $(MES_PAK).tar
	chmod 644 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(OTHERS) $(HLIST)  `echo $(TORTURE) | sed -e 's/\.o/.c/g'` 
	chmod 755 configure
	$(MAKE) list
	$(TAR) cvf $(MES_PAK).tar \
	 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(HLIST)  $(OTHERS) \
	`echo $(TORTURE) | sed -e 's/\.o/.c/g'` \
	MACHINES DOC

# use this only for PC machines	
msdos-zip:
	- /bin/rm -f $(MES_PAK).zip
	chmod 644 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(OTHERS) $(HLIST) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` 
	chmod 755 configure
	$(MAKE) list
	$(ZIP)  $(MES_PAK).zip \
	 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(HLIST)  $(OTHERS) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` \
	MACHINES DOC


fullshar:
	- /bin/rm -f $(MES_PAK).shar;
	chmod 644 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(OTHERS) $(HLIST) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` 
	chmod 755 configure
	$(MAKE) list
	$(SHAR) `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(HLIST)  $(OTHERS) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` \
	MACHINES DOC > $(MES_PAK).shar

shar:
	- /bin/rm -f meschach1.shar meschach2.shar meschach3.shar \
	meschach4.shar oldmeschach.shar meschach0.shar 
	chmod 644 `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	$(OTHERS) $(HLIST) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` 
	chmod 755 configure	
	$(MAKE) list
	$(SHAR) `echo $(LIST1) | sed -e 's/\.o/.c/g'` > meschach1.shar
	$(SHAR) `echo $(LIST2) | sed -e 's/\.o/.c/g'` > meschach2.shar
	$(SHAR) `echo $(LIST3) | sed -e 's/\.o/.c/g'` > meschach3.shar	
	$(SHAR) `echo $(ZLIST1) | sed -e 's/\.o/.c/g'` \
	  `echo $(ZLIST2) | sed -e 's/\.o/.c/g'` > meschach4.shar
	$(SHAR) `echo $(OLDLIST) | sed -e 's/\.o/.c/g'` > oldmeschach.shar
	$(SHAR) $(OTHERS) `echo $(TORTURE) | sed -e 's/\.o/.c/g'` \
	  $(HLIST)  DOC MACHINES  > meschach0.shar

list:
	/bin/rm -f $(FLIST)
	ls -lR `echo $(ALL_LISTS) | sed -e 's/\.o/.c/g'` \
	`echo $(TORTURE) | sed -e 's/\.o/.c/g'` \
	$(HLIST) $(OTHERS) MACHINES DOC \
	|awk '/^$$/ {print};/^[-d]/ {printf("%s %s   %10d %s %s %s %s\n", \
	 $$1,$$2,$$5,$$6,$$7,$$8,$$9)}; /^[^-d]/ {print}' \
	> $(FLIST)



clean:
	/bin/rm -f *.o core asx5213a.mat iotort.dat 

cleanup:
	/bin/rm -f *.o core asx5213a.mat iotort.dat *.a *.dll *.def

realclean:
	/bin/rm -f *.o core asx5213a.mat iotort.dat *.a *.dll *.def
	/bin/rm -f torture.exe sptort.exe ztorture.exe memtort.exe \
                   itertort.exe mfuntort.exe iotort.exe
	/bin/rm -f maxint.exe macheps.exe
alltorture: torture sptort ztorture memtort itertort mfuntort iotort

torture:torture.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o torture torture.o \
	libmeschach.a $(LIBS)
sptort:sptort.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o sptort sptort.o \
	libmeschach.a $(LIBS)
memtort: memtort.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o memtort memtort.o \
	libmeschach.a $(LIBS)
ztorture:ztorture.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o ztorture ztorture.o \
	libmeschach.a $(LIBS)
itertort: itertort.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o itertort itertort.o \
	libmeschach.a $(LIBS)

iotort: iotort.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o iotort iotort.o \
	libmeschach.a $(LIBS)
mfuntort: mfuntort.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o mfuntort mfuntort.o \
	libmeschach.a $(LIBS)
tstmove: tstmove.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o tstmove tstmove.o \
	libmeschach.a $(LIBS)
tstpxvec: tstpxvec.o libmeschach.a
	$(CC) $(CFLAGS) $(DEFS) -o tstpxvec tstpxvec.o \
	libmeschach.a $(LIBS)

OBJ=$(LIST1) $(LIST2) $(LIST3) $(ZLIST1) $(ZLIST2)
DLLTOOL= dlltool
DLLWRAP= dllwrap

DLL_NAME = meschach.dll
DLL_EXP_DEF = libmeschach_dll.def
DLL_EXP_LIB = libmeschach_dll.a


DLL_LDFLAGS   = -L .
DLL_LDLIBS    =  -lm
DLL_CFLAGS =  --verbose --export-all-symbols
DLLWRAP_FLAGS =  --output-def $(DLL_EXP_DEF) \
                 --implib $(DLL_EXP_LIB)  \
                 --driver-name $(CC)   #--no-delete

$(DLL_NAME) $(DLL_EXP_DEF) $(DLL_EXP_LIB): $(OBJ)
	$(DLLWRAP) $(DLL_CFLAGS) $(DLLWRAP_FLAGS) -o $(DLL_NAME) \
        $(OBJ) $(DLL_LDFLAGS) $(DLL_LDLIBS)



dll: $(DLL_NAME)