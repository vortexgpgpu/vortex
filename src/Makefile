################################################################################
# HARPtools by Chad D. Kersey, Summer 2011                                     #
################################################################################
CXXFLAGS ?= -std=c++11 -fPIC -O3 # -g -DUSE_DEBUG=3 -DPRINT_ACTIVE_THREADS
LDLIBS ?= -pthread
PREFIX ?= /usr/local

LIB_OBJS=args.o obj.o mem.o core.o instruction.o enc.o util.o

all: harptool libharplib.so libharplib.a #libqsim-harp.so

# Use -static so we don't have to install the library in order to just run
# Harptool.
harptool: harptool.o libharplib.a
	$(CXX) $(LDFLAGS) -o $@ harptool.o libharplib.a $(LDLIBS)

libharplib.so: $(LIB_OBJS)
	$(CXX) -shared -o $@ $(LIB_OBJS) $(LDLIBS)

libharplib.a: $(LIB_OBJS)
	ar rcs $@ $(LIB_OBJS)

args.o : args.cpp include/args.h
enc.o : enc.cpp include/types.h include/util.h include/enc.h include/archdef.h\
                include/instruction.h
harptool.o : harptool.cpp include/types.h include/core.h include/enc.h \
             include/instruction.h include/mem.h include/obj.h \
             include/archdef.h include/args.h include/help.h include/debug.h
instruction.o : instruction.cpp include/instruction.h include/obj.h \
                include/core.h include/debug.h include/asm-tokens.h
obj.o : obj.cpp include/types.h include/obj.h include/util.h \
        include/asm-tokens.h include/debug.h include/instruction.h
util.o : util.cpp include/types.h include/util.h
mem.o : mem.cpp include/types.h include/util.h include/mem.h include/debug.h \
        include/core.h
core.o : core.cpp include/types.h include/util.h include/mem.h \
         include/debug.h include/archdef.h include/core.h

#QSIM_CXXFLAGS=-DEMU_INSTRUMENTATION
#
#libqsim-harp.so: args.cpp enc.cpp instruction.cpp obj.cpp util.cpp mem.cpp \
#                 core.cpp qsim-harp.cpp lex.yy.o include/qsim-harp.h \
#                 include/types.h include/core.h include/util.h include/enc.h \
#                 include/archdef.h include/instruction.h include/asm-tokens.h \
#                 include/mem.h
#	$(CXX) $(CXXFLAGS) $(QSIM_CXXFLAGS) -shared -o $@ $^


install:
	cp libharplib.so $(PREFIX)/lib
	cp harptool $(PREFIX)/bin
	mkdir -p $(PREFIX)/include/harp
	cp include/* $(PREFIX)/include/harp

clean:
	rm -f *~ \#* *.o *.a *.so include/*~ include/\#* harptool
