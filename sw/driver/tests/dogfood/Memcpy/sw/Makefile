include ../../common/sw/common_include.mk

# Primary test name
TEST = cci_hello

# Build directory
OBJDIR = obj
CFLAGS += -I./$(OBJDIR)
CPPFLAGS += -I./$(OBJDIR)

# Files and folders
SRCS = $(TEST).c
OBJS = $(addprefix $(OBJDIR)/,$(patsubst %.c,%.o,$(SRCS)))

# Targets (build only $(TEST)_ase by default)
all: $(TEST) $(TEST)_ase

# AFU info from JSON file, including AFU UUID
AFU_JSON_INFO = $(OBJDIR)/afu_json_info.h

$(AFU_JSON_INFO): ../hw/rtl/$(TEST).json | objdir
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

$(OBJS): $(AFU_JSON_INFO)

$(TEST): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) $(FPGA_LIBS)

$(TEST)_ase: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) $(ASE_LIBS)

$(OBJDIR)/%.o: %.c | objdir
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TEST) $(TEST)_ase $(OBJDIR)

objdir:
	@mkdir -p $(OBJDIR)

.PHONY: all clean
