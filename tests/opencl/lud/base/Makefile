# Makefile for LU Decomposition
#
#

# DEBUG can be set to YES to include debugging info, or NO otherwise
DEBUG          := YES

# PROFILE can be set to YES to include profiling info, or NO otherwise
PROFILE        := NO

# ------------  name of the executable  ----------------------------------------
EXECUTABLE      := lud_base

# ------------  list of all source files  --------------------------------------
SOURCES         := lud.c lud_base.c ../common/common.c 

# ------------  compiler  ------------------------------------------------------
CC              := gcc
CXX             := g++

# ------------  compiler flags  ------------------------------------------------
DEBUG_CFLAGS    := -Wall -O0 -g
RELEASE_CFLAGS  := -Wall -O3 
COMMON_CFLAGS   := 

# ------------  compiler defs  ------------------------------------------------
DEBUG_CDEFS    := -D_DEBUG_RESULT
RELEASE_CDEFS  := 
COMMON_CDEFS   := 

# ------------  linker flags  --------------------------------------------------
DEBUG_LDFLAGS    := -g
RELEASE_LDFLAGS  := 
COMMON_LDFLAGS   := 

ifeq (YES, ${DEBUG})
  CFLAGS       := ${DEBUG_CFLAGS} ${DEBUG_CDEFS}
  CXXFLAGS     := ${DEBUG_CXXFLAGS} ${DEBUG_CDEFS}
  LDFLAGS      := ${DEBUG_LDFLAGS}
else
  CFLAGS       := ${RELEASE_CFLAGS} ${RELEASE_CDEFS}
  CXXFLAGS     := ${RELEASE_CXXFLAGS} ${RELEASE_CDEFS}
  LDFLAGS      := ${RELEASE_LDFLAGS}
endif

CFLAGS         := ${CFLAGS} ${COMMON_CFLAGS}
CXXFLAGS       := ${CXXFLAGS} ${COMMON_CFLAGS}
LDFLAGS        := ${LDFLAGS} ${COMMON_LDFLAGS}

ifeq (YES, ${PROFILE})
  CFLAGS       := ${CFLAGS}   -pg -O3
  CXXFLAGS     := ${CXXFLAGS} -pg -O3
  LDFLAGS      := ${LDFLAGS}  -pg
endif

# ------------  additional system include directories  -------------------------
GLOBAL_INC_DIR  =

# ------------  private include directories  -----------------------------------
LOCAL_INC_DIR   = ../common

# ------------  system libraries  (e.g. -lm )  ---------------------------------
SYS_LIBS        = 

# ------------  additional system library directories  -------------------------
GLOBAL_LIB_DIR  =

# ------------  additional system libraries  -----------------------------------
GLOBAL_LIBS     =

# ------------  private library directories  -----------------------------------
LOCAL_LIB_DIR   = 

# ------------  private libraries  (e.g. libxyz.a )  ---------------------------
LOCAL_LIBS      =


C_SOURCES       = $(filter     %.c, $(SOURCES))
CPP_SOURCES     = $(filter-out %.c, $(SOURCES))
ALL_INC_DIR     = $(addprefix -I, $(LOCAL_INC_DIR) $(GLOBAL_INC_DIR))
ALL_LIB_DIR     = $(addprefix -L, $(LOCAL_LIB_DIR) $(GLOBAL_LIB_DIR))
GLOBAL_LIBSS    = $(addprefix $(GLOBAL_LIB_DIR)/, $(GLOBAL_LIBS))
LOCAL_LIBSS     = $(addprefix $(LOCAL_LIB_DIR)/, $(LOCAL_LIBS))
ALL_CFLAGS      = $(CFLAGS) $(ALL_INC_DIR)
ALL_LFLAGS      = $(LDFLAGS) $(ALL_LIB_DIR)
BASENAMES       = $(basename $(SOURCES))

# ------------  generate the names of the object files  ------------------------
OBJECTS         = $(addsuffix .o,$(BASENAMES))
 
# ------------  make the executable (the default goal)  ------------------------
$(EXECUTABLE):	$(OBJECTS)
ifeq ($(strip $(CPP_SOURCES)),)
	$(CC)  $(ALL_LFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LOCAL_LIBSS) $(GLOBAL_LIBSS) $(SYS_LIBS)
else
	$(CXX) $(ALL_LFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LOCAL_LIBSS) $(GLOBAL_LIBSS) $(SYS_LIBS)
endif

# ------------  make the objects  ----------------------------------------------
%.o:		%.c
				$(CC)  -c $(ALL_CFLAGS) $< -o $@

%.o:		%.cc
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

%.o:		%.cpp
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

%.o:		%.C
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

clean:
	-rm --force $(EXECUTABLE) $(OBJECTS) *~

.PHONY: clean

# ==============================================================================
# vim: set tabstop=2: set shiftwidth=2:
