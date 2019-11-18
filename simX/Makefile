################################################################################
# HARPtools by Chad D. Kersey, Summer 2011                                     #
################################################################################
CXXFLAGS ?= -std=c++11 -fPIC -O3 # -g -DUSE_DEBUG=3 -DPRINT_ACTIVE_THREADS

LIB_OBJS=simX.cpp args.cpp mem.cpp core.cpp instruction.cpp enc.cpp util.cpp


INCLUDE=-I. -I../rtl/shared_memory -I../rtl/cache -I../rtl/interfaces -Isimulate -I../rtl
FILE=cache_simX.v
COMP=--compiler gcc
LIB=
CF=-CFLAGS '-std=c++11 -fPIC -O3'

LIGHTW=-Wno-UNOPTFLAT
DEB=--trace --prof-cfuncs -DVL_DEBUG=1 
EXE=--exe $(LIB_OBJS)

all: simX

# simX:
# 	g++ $(LDFLAGS) $(LIB_OBJS) -o simX.run


simX:
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF) $(LIGHTW) $(DEB)
	(cd obj_dir && make -j -f Vcache_simX.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

clean:
	rm -rf *~ \#* *.o *.a *.so include/*~ include/\#* simX.run obj_dir
