CXXFLAGS += -std=c++11 -O3 -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I../../runtime -I../../hw

CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread

SRCS = vortex.cpp ../common/vx_utils.cpp

PROJECT = libvortex.so

all: $(PROJECT)

$(PROJECT): $(SRCS) 
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(PROJECT) obj_dir