################################################################################
# HARPtools by Chad D. Kersey, Summer 2011                                     #
################################################################################

CXXFLAGS ?= -std=c++11 -fPIC -O3 -Wall -Wextra -pedantic -DUSE_DEBUG=3 -DPRINT_ACTIVE_THREADS
#CXXFLAGS ?= -std=c++11 -fPIC -g -O0 -Wall -Wextra -pedantic -DUSE_DEBUG=3 -DPRINT_ACTIVE_THREADS

LIB_OBJS=simX.cpp args.cpp mem.cpp core.cpp instruction.cpp enc.cpp util.cpp


INCLUDE=-I../hw/old_rtl -I../hw/old_rtl/interfaces -I../hw/old_rtl/cache -I../hw/old_rtl/shared_memory
FILE=cache_simX.v
COMP=--compiler gcc
LIB=

CF=-CFLAGS '-std=c++11 -fPIC -O3 -Wall -Wextra -pedantic -I../../hw'
#CF=-CFLAGS '-std=c++11 -fPIC -O0 -g -Wall -Wextra -pedantic'

LIGHTW=-Wno-UNOPTFLAT -Wno-WIDTH
DEB=--trace -DVL_DEBUG=1 
EXE=--exe $(LIB_OBJS)

all: simX

# simX:
# 	g++ $(LDFLAGS) $(LIB_OBJS) -o simX.run


simX:
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF) $(LIGHTW) $(DEB)
	(cd obj_dir && make -j -f Vcache_simX.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

clean:
	rm -rf *~ \#* *.o *.a *.so include/*~ include/\#* simX.run obj_dir
