#include "warp.h"

using namespace vortex;

void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, uint32_t vsew, uint32_t vl, uint32_t vmask);
void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, uint32_t vsew, uint32_t vl, uint32_t vmask);