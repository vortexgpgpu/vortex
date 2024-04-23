

				     
			     Meschach Library
			       Version 1.2b


			     David E. Stewart
			(david.stewart@anu.edu.au)

				    and

			       Zbigniew Leyk
			(zbigniew.leyk@anu.edu.au)

		      School of Mathematical Sciences
		      Australian National University
			     Canberra ACT 0200
				 Australia


		      [last revised: 6th April, 1994]


			      1. INTRODUCTION

   The Meschach Library is a numerical library of C routines for performing
calculations on matrices and vectors. It is intended for solving systems of
linear equations (dense and sparse), solve least squares problems,
computing eigenvalues and eigenvectors, etc. We do not claim that it
contains every useful algorithm in numerical linear algebra, but it does
provide a basis on which more advanced algorithms can be built. The library
is for people who know something about the C programming language,
something of how to solve the numerical problem they are faced with but do
not want to have the hassle of building all the necessary routines from the
scratch. The library is not a loose collection of numerical routines but it
comprises a coherent system. The current version is enhanced with many
features comparing with previous versions. Since the memory requirements
are nontrivial for large problems we have paid more attention to
allocation/deallocation of memory.

   The source code is available to be perused, used and passed on without
cost, while ensuring that the quality of the software is not compromised.
The software is copyrighted; however, the copyright agreement follows in
the footsteps of the Free Software Foundation in preventing abuse that
occurs with totally public domain software.

   Detailed instructions for installing Meschach are contained below.

   Pronunciation: if in doubt, say "me-shark".  This is close enough.
Don't ask us "Why call it that?"  Have a look at the quote at the front of
the manual.


			      2. AVAILABILITY

    The authors make this code openly available to others, in the hope that
it will prove to be a useful tool.  We ask only that:

* If you publish results obtained using Meschach, please consider
  acknowledging the source of the code.

* If you discover any errors in the code, please promptly communicate them
  to the authors.

    We also suggest that you send email to the authors identifying yourself
as a user of Meschach; this will enable the authors to notify you of any
corrections/improvements in Meschach.



			     3. HOW TO GET IT

   There are several different forms in which you might receive Meschach.
To provide a shorthand for describing collections of files, the Unix
convention of putting alternative letters in [...] will be used.  (So,
fred[123] means the collection fred1, fred2 and fred3.)  Meschach is
available over Internet/AARnet via netlib, or at the anonymous ftp site
thrain.anu.edu.au in the directory pub/meschach.  There are five .shar
files: meschach[01234].shar (which contain the library itself),
meschach0.shar (which contains basic documentation and machine dependent
files for a number of machines).  Of the meschach[1234].shar files, only
meschach[12].shar are needed for the basic Meschach library; the third
.shar file contains the sparse matrix routines, and the the fourth contains
the routines for complex numbers, vectors and matrices.  There is also a
README file that you should get from meschach0.shar.

   If you need the old iterative routines, the file oldmeschach.shar
contains the files conjgrad.c, arnoldi.c and lanczos.c.

   To get the library from netlib,

mail netlib@research.att.com
send all from c/meschach

   There are a number of other netlib sites which mirror the main netlib
sites.  These include netlib@ornl.gov (Oak Ridge, TN, USA), netlib@nac.no
(Oslo, Norway), ftp.cs.uow.edu.au (Wollongong, Australia; ftp only),
netlib@nchc.edu.tw (Taiwan), elib.zib-berlin.de (Berlin, Germany; ftp
only).  (For anonymous ftp sites the directory containing the Meschach
.shar files is pub/netlib/c/meschach or similar, possibly depending on the
site.)

   Meschach is available in other forms on thrain.anu.edu.au by ftp in the
directory pub/meschach.  It is available as a .tar file (mesch12a.tar for
version 1.2a), or as a collection of .shar files, or as a .zip file.  The
.tar and .zip versions each contain the entire contents of the Meschach
library.

   There is a manual called "Meschach: Matrix Computations in C" which has
been published by

	Centre for Mathematics and its Applications
	School of Mathematical Sciences
	Australian National University
	Canberra, ACT 0200
	Australia

and costs A$30 (about US$22) + postage/handling.  You can order it by
writing there or you can send email messages to one of us
(david.stewart@anu.edu.au or zbigniew.leyk@anu.edu.au) and we can pass it
on.

   If you don't have any money, as a stop gap you can get the **OLD**
manual, although it is out of date, by anonymous ftp from

	thrain.anu.edu.au : /pub/meschach/version1.1b/bookdvi.tar [.Z or .gz]

In addition, don't forget that the distribution includes a DOC directory
which contains tutorial.txt and fnindex.txt which are respectively, the
tutorial chapter (text version) and the function index (text version).



			      4. INSTALLATION

			    a) On Unix machines

   To extract the files from the .shar files, put them all into a suitable
directory and use

  sh <file>.shar

to expand the files.  (Use one sh command per file; sh *.shar will not work
in general.)

   For the .tar file, use

  tar xvf mesch12a.tar

and for the .zip file use

  unzip mesch12a.zip

   On a Unix system you can use the configure script to set up the
machine-dependent files.  The script takes a number of options which are
used for installing different subsets of the full Meschach.  For the basic
system, which requires only meschach[012].shar, use

  configure
  make basic
  make clean

   For including sparse operations, which requires meschach[0123].shar, use

  configure --with-sparse
  make sparse
  make clean

  For including complex operations, which requires meschach[0124].shar, use

  configure --with-complex
  make complex
  make clean

   For including everything, which requires meschach[01234].shar, use

  configure --with-all
  make all
  make clean

  To compile the complete library in single precision (with Real equivalent
to float), add the --with-float option to configure, use

  configure --with-all --with-float
  make all
  make clean


   Some Unix-like systems may have some problems with this due to bugs or
incompatibilities in various parts of the system.  To check this use make
torture and run torture.  In this case use the machine-dependent files from
the machines directory.  (This is the case for RS/6000 machines, the -O
switch results in failure of a routine in schur.c.  Compiling without the
-O switch results in correct results.)

   If you have problems using configure, or you use a non-Unix system,
check the MACHINES directory (generated by meschach0.shar) for your
machine, operating system and/or compiler.  Save the machine dependent
files makefile, machine.c and machine.h.  Copy those files from the
directory for your machine to the directory where the source code is.

   To link into a program prog.c, compile it using

  cc -o prog_name prog.c ....(source files).... meschach.a -lm


   This code has been mostly developed on the University of Queensland,
Australia's Pyramid 9810 running BSD4.3.  Initial development was on a
Zilog Zeus Z8000 machine running Zeus, a Unix workalike operating system.
Versions have also been successfully used on various Unix machines
including Sun 3's, IBM RT's, SPARC's and an IBM RS/6000 running AIX.  It
has also been compiled on an IBM AT clone using Quick C.  It has been
designed to compile under either Kernighan and Richie, (Edition 1) C and
under ANSI C.  (And, indeed, it has been compiled in both ANSI C and
non-ANSI C environments.)


			  b) On non-Unix machines

   First look in the machines directory for your system type.  If it is
there, then copy the machine dependent files machine.h, makefile (and
possibly machine.c) to the Meschach directory.

   If your machine type is not there, then you will need to either compile
``by hand'', or construct your own makefile and possibly machine.h as well.
The machine-dependent files for various systems should be used as a
starting point, and the ``vanilla'' version of machine.h should be used.
Information on the machine-dependent files follows in the next three
subsections.

   On an IBM PC clone, the source code would be on a floppy disk. Use

  xcopy a:* meschach

to copy it to the meschach directory.  Then ``cd meschach'', and then
compile the source code.  Different compilers on MSDOS machines will
require different installation procedures.  Check the directory meschach
for the appropriate ``makefile'' for your compiler.  If your compiler is
not listed, then you should try compiling it ``by hand'', modifying the
machine-dependent files as necessary.

   Worst come to worst, for a given C compiler, execute
		<C compiler name> *.c
on MS-DOS machines. For example,
		tcc *.c
for Turbo C, and
		msc *.c
for Microsoft C, or if you are using Quick C,
		qcl *.c
and of course
		cc *.c
for the standard Unix compiler.

   Once the object files have been generated, you will need to combine them
into a library. Consult your local compiler's manual for details of how to
do this.

   When compiling programs/routines that use Meschach, you will need to
have access the the header files in the INCLUDE directory. The INCLUDE
directory's contents can be copied to the directory where the
programs/routines are compiled.

   The files in the DOC directory form a very brief form of documentation
on the the library routines in Meschach. See the printed documentation for
more comprehensive documentation of the Meschach routines.  This can be
obtained from the authors via email.

   The files and directories created by the machines.shar shell archive
contain the files machine.c machine.h and makefile for a particular
machine/operating system/compiler where they need to be different.  Copy
the files in the appropriate directory for your machine/operating
system/compiler to the directory with the Meschach source before compiling.



			       c)  makefile


   This is setup by using the configure script on a Unix system, based on
the makefile.in file.  However, if you want to modify how the library is
compiled, you are free to change the makefile.

   The most likely change that you would want to make to this file is to
change the line

  CFLAGS = -O

to suit your particular compiler.

  The code is intended to be compilable by both ANSI and non-ANSI
compilers.

   To achieve this portability without sacrificing the ANSI function
prototypes (which are very useful for avoiding problems with passing
parameters) there is a token ANSI_C which must be #define'd in order to
take full advantage of ANSI C.  To do this you should do all compilations
with

  #define ANSI_C 1

   This can also be done at the compilation stage with a -DANSI_C flag.
Again, you will have to use the -DANSI_C flag or its equivalent whenever
you compile, or insert the line

  #define ANSI_C 1

in machine.h, to make full use of ANSI C with this matrix library.


			       d)  machine.h

   Like makefile this is normally set up by the configure script on Unix
machines.  However, for non-Unix systems, or if you need to set some things
``by hand'', change machine.h.

   There are a few quantities in here that should be modified to suit your
particular compiler.  Firstly, the macros MEM_COPY() and MEM_ZERO() need to
be correctly defined here.  The original library was compiled on BSD
systems, and so it originally relied on bcopy() and bzero().

   In machine.h you will find the definitions for using the standard ANSI C
library routines:

  /*--------------------ANSI C--------------------*/
  #include        <stddef.h>
  #include        <string.h>
  #define	MEM_COPY(from,to,size)  memmove((to),(from),(size))
  #define	MEM_ZERO(where,size)    memset((where),'\0',(size))

   Delete or comment out the alternative definitions and it should compile
correctly.  The source files containing memmove() and/or memset() are
available by anonymous ftp from some ftp sites (try archie to discover 
them). The files are usually called memmove.c or memset.c.
Some ftp sites which currently (Jan '94) have a version of these files are
munnari.oz.au (in Australia), ftp.uu.net, gatekeeper.dec.com (USA), and
unix.hensa.ac.uk (in the UK).  The directory in which you will find
memmove.c and memset.c typically looks like .../bsd-sources/lib/libc/...

   There are two further machine-dependent quantities that should be set.
These are machine epsilon or the unit roundoff for double precision
arithmetic, and the maximum value produced by the rand() routine, which is
used in rand_vec() and rand_mat().


   The current definitions of these are

  #define	MACHEPS	2.2e-16
  #define	MAX_RAND 2.147483648e9

   The value of MACHEPS should be correct for all IEEE standard double
precision arithmetic.

   However, ANSI C's <float.h> contains #define'd quantities DBL_EPSILON
and RAND_MAX, so if you have an ANSI C compiler and headers, replace the
above two lines of machine.h with

  #include <float.h>
  /* for Real == float */
  #define MACHEPS DBL_EPSILON
  #define MAX_RAND RAND_MAX

   The default value given for MAX_RAND is 2^31 , as the Pyramid 9810 and
the SPARC 2's both have 32 bit words.  There is a program macheps.c which
is included in your source files which computes and prints out the value of
MACHEPS for your machine.

   Some other macros control some aspects of Meschach.  One of these is
SEGMENTED which should be #define'd if you are working with a machine or
compiler that does not allow large arrays to be allocated.  For example,
the most common memory models for MS-DOS compilers do not allow more than
64Kbyte to be allocated in one block.  This limits square matrices to be no
more than 9090 .  Inserting #define SEGMENTED 1 into machine.h will mean
that matrices are allocated a row at a time.



			      4. SAMPLE TESTS

    There are several programs for checking Meschach called torture
(source: torture.c) for the dense routines, sptort (source: sptort.c) for
the sparse routines, ztorture (source ztorture.c) for a complex version of
torture, memtort (source memtort.c) for memory allocation/deallocation,
itertort (source itertort.c) for iterative methods, mfuntort (source
mfuntort.c) for computing powers of dense matrices, iotort (source
iotort.c) for I/O routines.  These can be compiled using make by "make
torture", "make sptort", etc.  The programs are part of meschach0.shar.


			     5. OTHER PROBLEMS

   Meschach is not a commercial package, so we do not guarantee that
everything will be perfect or will install smoothly.  Inevitably there will
be unforeseen problems. If you come across any bugs or inconsistencies, please
let us know.  If you need to modify the results of the configure script, or
need to construct your own machine.h and makefile's, please send them to
us.  A number of people sent us the machine dependent files for Meschach 1.1,
but with the use of configure, and the new information needed for version
1.2, these machine dependent files don't have quite the right information.
Hopefully, though, they are redundant.  Non-Unix platforms at present
require ``manual'' installation.  Because of the variety of platforms
(MS-DOS, Macintosh, VAX/VMS, Prime, Amiga, Atari, ....) this is left up to
the users of these platforms.  We hope that you can use the distibutable
machine-dependent files as a starting point for this task.

   If you have programs or routines written using Meschach v.1.1x, you
should put the statement

   #include "oldnames.h"

at the beginning of your files.  This is because a large number of the
names of the routines have been changed (e.g. "get_vec()" has become
"v_get()").  This will enable you to use the old names, although all of the
error messages etc., will use the new names.  Also note that the new
iterative routines have a very different calling sequence.  If you need the
old iterative routines, they are in oldmeschach.shar.

   If you wish to let us know what you have done, etc., our email
addresses are

			 david.stewart@anu.edu.au
			 zbigniew.leyk@anu.edu.au

    Good luck!
 

			      ACKNOWLEDGMENTS


    Many people have helped in various ways with ideas and suggestions.
Needless to say, the bugs are all ours!  But these people should be thanked
for their encouragement etc.  These include a number of people at
University of Queensland: Graeme Chandler, David De Wit, Martin Sharry,
Michael Forbes, Phil Kilby, John Holt, Phil Pollett and Tony Watts.  At the
Australian National University: Mike Osborne, Steve Roberts, Margaret Kahn
and Teresa Leyk.  Karen George of the University of Canberra has been a
source of both ideas and encouragement.  Email has become significant part
of work, and many people have pointed out bugs, inconsistencies and
improvements to Meschach by email.  These people include Ajay Shah of the
University of Southern California, Dov Grobgeld of the Weizmann Institute,
John Edstrom of the University of Calgary, Eric Grosse, one of the netlib
organisers, Ole Saether of Oslo, Norway, Alfred Thiele and Pierre
Asselin of Carnegie-Mellon Univeristy, Daniel Polani of the University of
Mainz, Marian Slodicka of Slovakia, Kaifu Wu of Pomona, Hidetoshi
Shimodaira of the University of Tokyo, Eng Siong of Edinburgh, Hirokawa Rui
of the University of Tokyo, Marko Slyz of the University of Michigan, and
Brook Milligan of the University of Texas.  This list is only partial, and
there are many others who have corresponded with us on details about
Meschach and the like.  Finally our thanks go to all those that have had to
struggle with compilers and other things to get Meschach to work.

				     



