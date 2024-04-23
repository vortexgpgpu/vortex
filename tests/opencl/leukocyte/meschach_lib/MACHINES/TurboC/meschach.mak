# This directory contains a makefile for Borland C++.
# It was written by Andrew Gockel (contact information below).
# Use at own risk.  This is provided as part of the standard Meschach
# distribution to give the library the widest possible use.
# However, problems with the makefile should be directed to the author,
# not the developers of Meschach (David Stewart and Zbigniew Leyk).
# 
# No representations are made concerning the fitness of this software for any
# particular purpose.

# Borland C++ V4 Makefile
#
# Saturday, 14 October, 1995
#
# Andrew Gockel
# 123 Settlement Road
# THE GAP, QLD., 4061
# AUSTRALIA
#
# Email
# INTERNET:andrew@kittyhawk.aero.rmit.edu.au
# CIS:100245.1253@compuserve.com
# MSN:Andrew_Gockel@msn.com
#
# c:\meschach\meschach.mak
#
.AUTODEPEND


#
# Borland C++ tools
#
IMPLIB  = Implib
BCC     = Bcc +BccW16.cfg 
TLINK   = TLink
TLIB    = TLib
BRC     = Brc
TASM    = Tasm
#
# IDE macros
#


#
# Options
#
IDE_LFLAGS =  -LD:\BC4\LIB
IDE_RFLAGS =  -ID:\BC4\INCLUDE
LLATW16_meschachdlib =  -Twe
RLATW16_meschachdlib =  -31
BLATW16_meschachdlib = 
LEAT_meschachdlib = $(LLATW16_meschachdlib)
REAT_meschachdlib = $(RLATW16_meschachdlib)
BEAT_meschachdlib = $(BLATW16_meschachdlib)

#
# Dependency List
#
Dep_meschach = \
   meschach.lib

meschach : BccW16.cfg $(Dep_meschach)
  echo MakeNode meschach

Dep_meschachdlib = \
   bdfactor.obj\
   bkpfacto.obj\
   chfactor.obj\
   copy.obj\
   err.obj\
   fft.obj\
   givens.obj\
   hessen.obj\
   hsehldr.obj\
   init.obj\
   iter0.obj\
   iternsym.obj\
   itersym.obj\
   ivecop.obj\
   lufactor.obj\
   machine.obj\
   matlab.obj\
   matop.obj\
   matrixio.obj\
   meminfo.obj\
   memory.obj\
   memstat.obj\
   mfunc.obj\
   norm.obj\
   otherio.obj\
   pxop.obj\
   qrfactor.obj\
   schur.obj\
   solve.obj\
   sparse.obj\
   sparseio.obj\
   spbkp.obj\
   spchfctr.obj\
   splufctr.obj\
   sprow.obj\
   spswap.obj\
   submat.obj\
   svd.obj\
   symmeig.obj\
   update.obj\
   vecop.obj\
   version.obj\
   zcopy.obj\
   zfunc.obj\
   zgivens.obj\
   zhessen.obj\
   zhsehldr.obj\
   zlufctr.obj\
   zmachine.obj\
   zmatio.obj\
   zmatlab.obj\
   zmatop.obj\
   zmemory.obj\
   znorm.obj\
   zqrfctr.obj\
   zschur.obj\
   zsolve.obj\
   zvecop.obj

meschach.lib : $(Dep_meschachdlib)
  $(TLIB) $< $(IDE_BFLAGS) $(BEAT_meschachdlib) @&&|
 -+bdfactor.obj&
-+bkpfacto.obj&
-+chfactor.obj&
-+copy.obj&
-+err.obj&
-+fft.obj&
-+givens.obj&
-+hessen.obj&
-+hsehldr.obj&
-+init.obj&
-+iter0.obj&
-+iternsym.obj&
-+itersym.obj&
-+ivecop.obj&
-+lufactor.obj&
-+machine.obj&
-+matlab.obj&
-+matop.obj&
-+matrixio.obj&
-+meminfo.obj&
-+memory.obj&
-+memstat.obj&
-+mfunc.obj&
-+norm.obj&
-+otherio.obj&
-+pxop.obj&
-+qrfactor.obj&
-+schur.obj&
-+solve.obj&
-+sparse.obj&
-+sparseio.obj&
-+spbkp.obj&
-+spchfctr.obj&
-+splufctr.obj&
-+sprow.obj&
-+spswap.obj&
-+submat.obj&
-+svd.obj&
-+symmeig.obj&
-+update.obj&
-+vecop.obj&
-+version.obj&
-+zcopy.obj&
-+zfunc.obj&
-+zgivens.obj&
-+zhessen.obj&
-+zhsehldr.obj&
-+zlufctr.obj&
-+zmachine.obj&
-+zmatio.obj&
-+zmatlab.obj&
-+zmatop.obj&
-+zmemory.obj&
-+znorm.obj&
-+zqrfctr.obj&
-+zschur.obj&
-+zsolve.obj&
-+zvecop.obj
|

bdfactor.obj :  bdfactor.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ bdfactor.c

bkpfacto.obj :  bkpfacto.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ bkpfacto.c

chfactor.obj :  chfactor.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ chfactor.c

copy.obj :  copy.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ copy.c

err.obj :  err.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ err.c

fft.obj :  fft.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ fft.c

givens.obj :  givens.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ givens.c

hessen.obj :  hessen.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ hessen.c

hsehldr.obj :  hsehldr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ hsehldr.c

init.obj :  init.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ init.c

iter0.obj :  iter0.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ iter0.c

iternsym.obj :  iternsym.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ iternsym.c

itersym.obj :  itersym.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ itersym.c

ivecop.obj :  ivecop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ ivecop.c

lufactor.obj :  lufactor.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ lufactor.c

machine.obj :  machine.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ machine.c

matlab.obj :  matlab.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ matlab.c

matop.obj :  matop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ matop.c

matrixio.obj :  matrixio.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ matrixio.c

meminfo.obj :  meminfo.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ meminfo.c

memory.obj :  memory.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ memory.c

memstat.obj :  memstat.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ memstat.c

mfunc.obj :  mfunc.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ mfunc.c

norm.obj :  norm.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ norm.c

otherio.obj :  otherio.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ otherio.c

pxop.obj :  pxop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ pxop.c

qrfactor.obj :  qrfactor.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ qrfactor.c

schur.obj :  schur.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ schur.c

solve.obj :  solve.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ solve.c

sparse.obj :  sparse.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ sparse.c

sparseio.obj :  sparseio.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ sparseio.c

spbkp.obj :  spbkp.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ spbkp.c

spchfctr.obj :  spchfctr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ spchfctr.c

splufctr.obj :  splufctr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ splufctr.c

sprow.obj :  sprow.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ sprow.c

spswap.obj :  spswap.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ spswap.c

submat.obj :  submat.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ submat.c

svd.obj :  svd.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ svd.c

symmeig.obj :  symmeig.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ symmeig.c

update.obj :  update.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ update.c

vecop.obj :  vecop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ vecop.c

version.obj :  version.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ version.c

zcopy.obj :  zcopy.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zcopy.c

zfunc.obj :  zfunc.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zfunc.c

zgivens.obj :  zgivens.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zgivens.c

zhessen.obj :  zhessen.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zhessen.c

zhsehldr.obj :  zhsehldr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zhsehldr.c

zlufctr.obj :  zlufctr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zlufctr.c

zmachine.obj :  zmachine.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zmachine.c

zmatio.obj :  zmatio.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zmatio.c

zmatlab.obj :  zmatlab.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zmatlab.c

zmatop.obj :  zmatop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zmatop.c

zmemory.obj :  zmemory.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zmemory.c

znorm.obj :  znorm.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ znorm.c

zqrfctr.obj :  zqrfctr.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zqrfctr.c

zschur.obj :  zschur.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zschur.c

zsolve.obj :  zsolve.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zsolve.c

zvecop.obj :  zvecop.c
  $(BCC)   -P- -c $(CEAT_meschachdlib) -o$@ zvecop.c

# Compiler configuration file
BccW16.cfg : 
   Copy &&|
-R
-v
-vi
-X-
-H
-ID:\BC4\INCLUDE
-H=meschach.csm
-ml
-WS
| $@


