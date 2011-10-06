################################################################################
# HARPtools by Chad D. Kersey, Summer 2011                                     #
################################################################################
CXXFLAGS=-g -fPIC

LIB_OBJS=args.o obj.o mem.o core.o instruction.o enc.o util.o lex.yy.o

all: harptool libharplib.so libharplib.a

# Use -static so we don't have to install the library in order to just run
# Harptool.
harptool: harptool.o libharplib.a
	$(CXX) -static -o $@ harptool.o -L. -lharplib  -pthread

libharplib.so: $(LIB_OBJS)
	$(CXX) -shared -o $@ $(LIB_OBJS)

libharplib.a: $(LIB_OBJS)
	ar rcs $@ $(LIB_OBJS)

args.o : args.cpp include/args.h
enc.o : enc.cpp include/types.h include/util.h include/enc.h include/archdef.h include/instruction.h
harptool.o : harptool.cpp include/types.h include/core.h include/enc.h include/instruction.h include/mem.h include/obj.h include/archdef.h include/args.h include/help.h
instruction.o : instruction.cpp include/instruction.h include/obj.h
obj.o : obj.cpp include/types.h include/obj.h include/util.h include/asm-tokens.h
util.o : util.cpp include/types.h include/util.h
mem.o : mem.cpp include/types.h include/util.h include/mem.h
core.o : core.cpp include/types.h include/util.h include/mem.h include/archdef.h

lex.yy.cc: scanner.lex
	flex scanner.lex

clean:
	rm -f *~ \#* *.o *.a *.so include/*~ include/\#* harptool
