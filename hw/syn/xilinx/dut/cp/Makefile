PROJECT = VX_cp_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

# VX_cp_core_top (hw/unittest/cp_core) wraps the Command Processor core.
# Its package/interfaces live in hw/rtl/cp; the wrapper itself in unittest/cp_core.
RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu
RTL_INCLUDE += -I$(RTL_DIR)/cp
RTL_INCLUDE += -I$(UNITTEST_DIR)/cp_core
